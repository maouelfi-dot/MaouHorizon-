//! Mabar skins: players on one hotspot or LAN see one another's skins and capes in offline mode.
//!
//! A server in offline mode sends no skin for a player, so every phone tells the others its own:
//!
//! * [`Share`] serves the phone's skin and cape PNGs over HTTP on a port of its own and
//!   announces `name`, that port and the images' SHA-1 by UDP broadcast to every network the
//!   phone is on (the hotspot's included), every two seconds.
//! * [`Peers`] listens for those announcements. [`Peers::look_of`] gives the URLs a player's
//!   images can be fetched from, which the game then downloads as it does an Ely.by skin.
//!
//! The address in a URL is the announcement's sender, never one it names, so a peer can only
//! point the game at its own phone. Images are named by their SHA-1: a changed skin is a new URL.

use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream, UdpSocket},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use super::discovery::local_networks;

/// The UDP port announcements go to (not the game's LAN port, 4445, nor Maou Launcher's).
pub const BEACON_PORT: u16 = 24446;
const MAGIC: &str = "MABAR-SKIN 1 ";
const INTERVAL: Duration = Duration::from_secs(2);
/// A peer is kept this long after its last announcement, so a skin does not flicker back to the
/// default one when the hotspot drops a few packets.
const PEER_TTL: Duration = Duration::from_secs(600);
/// Larger images are neither served nor believed: a 64x64 skin is a few kilobytes.
const MAX_IMAGE: usize = 256 * 1024;
const MAX_NAME: usize = 32;

/// What this phone shows of its player.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Look {
    pub name: String,
    /// The skin as a PNG.
    pub skin: Option<Vec<u8>>,
    pub slim: bool,
    /// The cape as a PNG (64x32).
    pub cape: Option<Vec<u8>>,
}

fn sha1_hex(data: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    Sha1::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn is_hash(text: &str) -> bool {
    text.len() == 40
        && text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The announcement of `look` served on `port`.
fn beacon(look: &Look, port: u16) -> String {
    let hash = |image: &Option<Vec<u8>>| image.as_deref().map(sha1_hex);
    let body = serde_json::json!({
        "n": look.name,
        "p": port,
        "s": hash(&look.skin),
        "m": look.slim,
        "c": hash(&look.cape),
    });
    format!("{MAGIC}{body}")
}

/// What an announcement says of a player.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Peer {
    ip: Ipv4Addr,
    port: u16,
    skin: Option<String>,
    slim: bool,
    cape: Option<String>,
}

/// A valid announcement from `from`: the player's name (lower case) and what it says. Anything
/// else, including a hash that is not 40 hex digits or a port a user cannot hold, is dropped.
fn parse_beacon(text: &str, from: Ipv4Addr) -> Option<(String, Peer)> {
    let body: serde_json::Value = serde_json::from_str(text.strip_prefix(MAGIC)?).ok()?;
    let name = body.get("n")?.as_str()?;
    if name.is_empty() || name.chars().count() > MAX_NAME || name.chars().any(char::is_control) {
        return None;
    }
    let port = u16::try_from(body.get("p")?.as_u64()?).ok()?;
    if port < 1024 {
        return None;
    }
    let hash = |key: &str| match body.get(key) {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(v) => v.as_str().filter(|h| is_hash(h)).map(|h| Some(h.to_owned())),
    };
    Some((
        name.to_lowercase(),
        Peer {
            ip: from,
            port,
            skin: hash("s")?,
            slim: body.get("m").and_then(|m| m.as_bool()).unwrap_or(false),
            cape: hash("c")?,
        },
    ))
}

/// Where a player's images are: URLs of the phone that announced them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerLook {
    pub skin: Option<String>,
    pub slim: bool,
    pub cape: Option<String>,
}

fn url(peer: &Peer, kind: &str, hash: &str) -> String {
    format!("http://{}:{}/{kind}/{hash}.png", peer.ip, peer.port)
}

/// Whether `address` (`host` or `host:port`) is a private or link-local IPv4 address: a LAN,
/// where Mabar skins are shared, not a server on the internet.
pub fn is_private_address(address: &str) -> bool {
    let host = address.rsplit_once(':').map_or(address, |(host, _)| host);
    host.parse::<Ipv4Addr>()
        .is_ok_and(|ip| ip.is_private() || ip.is_link_local())
}

/// A skin or cape as a PNG, from its RGBA pixels.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Option<Vec<u8>> {
    if rgba.len() != (width as usize) * (height as usize) * 4 {
        return None;
    }
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .ok()?
        .write_image_data(rgba)
        .ok()?;
    Some(out)
}

/// Listens for announcements until dropped.
pub struct Peers {
    peers: Arc<Mutex<HashMap<String, (Peer, Instant)>>>,
    stop: Arc<AtomicBool>,
}

impl Peers {
    /// Bind the announcement port; without it nobody is ever heard.
    pub fn start() -> Self {
        let peers: Arc<Mutex<HashMap<String, (Peer, Instant)>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        if let Ok(socket) = UdpSocket::bind(("0.0.0.0", BEACON_PORT)) {
            let _ = socket.set_read_timeout(Some(Duration::from_millis(250)));
            let (peers, stop) = (peers.clone(), stop.clone());
            let _ = std::thread::Builder::new()
                .name("horizon-mabar-skins-listen".into())
                .spawn(move || {
                    let mut buf = [0u8; 1024];
                    while !stop.load(Ordering::Relaxed) {
                        let Ok((n, SocketAddr::V4(from))) = socket.recv_from(&mut buf) else {
                            continue;
                        };
                        let text = String::from_utf8_lossy(&buf[..n]);
                        if let Some((name, peer)) = parse_beacon(&text, *from.ip())
                            && let Ok(mut peers) = peers.lock()
                        {
                            peers.insert(name, (peer, Instant::now()));
                        }
                    }
                });
        }
        Self { peers, stop }
    }

    /// Where `name`'s skin and cape are, when its phone has announced them.
    pub fn look_of(&self, name: &str) -> Option<PeerLook> {
        let peers = self.peers.lock().ok()?;
        let (peer, seen) = peers.get(&name.to_lowercase())?;
        if seen.elapsed() > PEER_TTL {
            return None;
        }
        Some(PeerLook {
            skin: peer.skin.as_deref().map(|h| url(peer, "skin", h)),
            slim: peer.slim,
            cape: peer.cape.as_deref().map(|h| url(peer, "cape", h)),
        })
    }
}

impl Drop for Peers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Hearing the others and, once this phone has a skin or cape of its own, sharing it: one
/// running piece for a session on a LAN.
pub struct Mabar {
    peers: Peers,
    share: Option<Share>,
    sent: Option<Look>,
    checked: Option<Instant>,
}

impl Mabar {
    pub fn start() -> Self {
        Self {
            peers: Peers::start(),
            share: None,
            sent: None,
            checked: None,
        }
    }

    /// Tell the LAN this phone's look, `look` asked for at most once a second (making it takes
    /// a PNG encode) and sent only when it changed. Nothing is shared without an image.
    pub fn update(&mut self, look: impl FnOnce() -> Look) {
        if self.checked.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
            return;
        }
        self.checked = Some(Instant::now());
        let look = look();
        if self.sent.as_ref() == Some(&look) {
            return;
        }
        if look.skin.is_none() && look.cape.is_none() {
            self.share = None;
        } else if let Some(share) = &self.share {
            share.set(look.clone());
        } else {
            match Share::start(look.clone()) {
                Ok(share) => self.share = Some(share),
                Err(e) => {
                    log::warn!("Horizon: Mabar skins could not be shared: {e}");
                    return;
                }
            }
        }
        self.sent = Some(look);
    }

    /// Where `name`'s skin and cape are, when its phone has announced them.
    pub fn look_of(&self, name: &str) -> Option<PeerLook> {
        self.peers.look_of(name)
    }
}

/// Serves this phone's images and announces them until dropped.
pub struct Share {
    look: Arc<Mutex<Look>>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Share {
    pub fn start(look: Look) -> io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", 0))?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();
        let look = Arc::new(Mutex::new(look));
        let stop = Arc::new(AtomicBool::new(false));
        let serve = {
            let (look, stop) = (look.clone(), stop.clone());
            std::thread::Builder::new()
                .name("horizon-mabar-skins-serve".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => answer(stream, &look),
                            Err(_) => std::thread::sleep(Duration::from_millis(50)),
                        }
                    }
                })?
        };
        let announce = {
            let (look, stop) = (look.clone(), stop.clone());
            std::thread::Builder::new()
                .name("horizon-mabar-skins-announce".into())
                .spawn(move || announce_loop(&look, port, &stop))?
        };
        Ok(Self {
            look,
            stop,
            threads: vec![serve, announce],
        })
    }

    /// What is shown from now on.
    pub fn set(&self, look: Look) {
        if let Ok(mut current) = self.look.lock() {
            *current = look;
        }
    }
}

impl Drop for Share {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn announce_loop(look: &Mutex<Look>, port: u16, stop: &AtomicBool) {
    let mut sockets: HashMap<Ipv4Addr, UdpSocket> = HashMap::new();
    while !stop.load(Ordering::Relaxed) {
        let message = look.lock().map(|l| beacon(&l, port)).unwrap_or_default();
        let networks = local_networks();
        sockets.retain(|ip, _| networks.iter().any(|(n, _)| n == ip));
        for (ip, broadcast) in networks {
            let socket = match sockets.entry(ip) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let Ok(s) = UdpSocket::bind((ip, 0)) else {
                        continue;
                    };
                    let _ = s.set_broadcast(true);
                    e.insert(s)
                }
            };
            if !message.is_empty() {
                let _ = socket.send_to(
                    message.as_bytes(),
                    SocketAddr::V4(SocketAddrV4::new(broadcast, BEACON_PORT)),
                );
            }
        }
        let mut slept = Duration::ZERO;
        while slept < INTERVAL && !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
            slept += Duration::from_millis(50);
        }
    }
}

/// One HTTP request: `GET /skin/<sha1>.png` or `/cape/<sha1>.png` of the current look.
fn answer(mut stream: TcpStream, look: &Mutex<Look>) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let mut request = [0u8; 1024];
    let mut len = 0;
    while len < request.len() && !request[..len].windows(2).any(|w| w == b"\r\n") {
        match stream.read(&mut request[len..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => len += n,
        }
    }
    let line = String::from_utf8_lossy(&request[..len]);
    let path = line
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("GET "))
        .and_then(|l| l.split(' ').next())
        .unwrap_or("");
    let body = look.lock().ok().and_then(|look| {
        let (kind, name) = path.strip_prefix('/')?.split_once('/')?;
        let hash = name.strip_suffix(".png")?;
        let image = match kind {
            "skin" => look.skin.as_ref()?,
            "cape" => look.cape.as_ref()?,
            _ => return None,
        };
        (image.len() <= MAX_IMAGE && sha1_hex(image) == hash).then(|| image.clone())
    });
    let _ = match body {
        Some(body) => stream
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .as_bytes(),
            )
            .and_then(|()| stream.write_all(&body)),
        None => stream.write_all(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn look() -> Look {
        Look {
            name: "Maou".into(),
            skin: Some(b"skin-bytes".to_vec()),
            slim: true,
            cape: Some(b"cape-bytes".to_vec()),
        }
    }

    #[test]
    fn an_announcement_names_the_images_by_their_sha1() {
        let text = beacon(&look(), 30000);
        let (name, peer) = parse_beacon(&text, Ipv4Addr::new(192, 168, 43, 5)).unwrap();
        assert_eq!(name, "maou");
        assert_eq!(peer.port, 30000);
        assert!(peer.slim);
        // FIPS 180 test vector for the hash, then the announcement carrying it.
        assert_eq!(sha1_hex(b"abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(peer.skin, Some(sha1_hex(b"skin-bytes")));
        assert_eq!(peer.cape, Some(sha1_hex(b"cape-bytes")));
        assert_eq!(peer.ip, Ipv4Addr::new(192, 168, 43, 5));
    }

    #[test]
    fn an_announcement_without_images_is_valid() {
        let text = beacon(
            &Look {
                name: "Steve".into(),
                ..Default::default()
            },
            4000,
        );
        let (_, peer) = parse_beacon(&text, Ipv4Addr::LOCALHOST).unwrap();
        assert_eq!((peer.skin, peer.cape), (None, None));
    }

    #[test]
    fn bad_announcements_are_dropped() {
        let from = Ipv4Addr::LOCALHOST;
        let good = format!("{MAGIC}{{\"n\":\"A\",\"p\":2000,\"s\":null,\"m\":false,\"c\":null}}");
        assert!(parse_beacon(&good, from).is_some());
        for bad in [
            "junk".to_owned(),
            format!("{MAGIC}not json"),
            // A port below 1024, and a name that is empty, too long or has control characters.
            format!("{MAGIC}{{\"n\":\"A\",\"p\":80}}"),
            format!("{MAGIC}{{\"n\":\"\",\"p\":2000}}"),
            format!("{MAGIC}{{\"n\":\"{}\",\"p\":2000}}", "x".repeat(33)),
            format!("{MAGIC}{{\"n\":\"a\\nb\",\"p\":2000}}"),
            // A hash that could climb out of the path or is not 40 lower-case hex digits.
            format!("{MAGIC}{{\"n\":\"A\",\"p\":2000,\"s\":\"../../x\"}}"),
            format!("{MAGIC}{{\"n\":\"A\",\"p\":2000,\"c\":\"{}\"}}", "G".repeat(40)),
            format!("{MAGIC}{{\"n\":\"A\",\"p\":70000}}"),
        ] {
            assert!(parse_beacon(&bad, from).is_none(), "{bad}");
        }
    }

    #[test]
    fn private_addresses_are_lans() {
        for yes in ["192.168.43.1", "192.168.43.1:25565", "10.0.0.7", "172.16.5.5:1", "169.254.1.1"] {
            assert!(is_private_address(yes), "{yes}");
        }
        for no in ["8.8.8.8", "8.8.8.8:25565", "play.example.com", "example.com:25565", "172.32.0.1", ""] {
            assert!(!is_private_address(no), "{no}");
        }
    }

    #[test]
    fn a_png_round_trips() {
        let rgba: Vec<u8> = (0..64 * 32 * 4).map(|i| (i % 251) as u8).collect();
        let bytes = encode_png(64, 32, &rgba).unwrap();
        let mut reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap();
        let mut out = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut out).unwrap();
        assert_eq!((info.width, info.height), (64, 32));
        assert_eq!(&out[..info.buffer_size()], &rgba[..]);
        assert!(encode_png(64, 64, &rgba).is_none());
    }

    fn get(address: SocketAddr, path: &str) -> (String, Vec<u8>) {
        let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        write!(stream, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut reply = Vec::new();
        let _ = stream.read_to_end(&mut reply);
        let split = reply.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        (
            String::from_utf8_lossy(&reply[..split]).lines().next().unwrap().to_owned(),
            reply[split + 4..].to_vec(),
        )
    }

    /// Two phones on one network: one shares, the other hears it and fetches the images.
    #[test]
    fn a_shared_look_is_heard_and_fetched() {
        let peers = Peers::start();
        let share = Share::start(look()).unwrap();
        let mut heard = None;
        for _ in 0..60 {
            if let Some(found) = peers.look_of("MAOU") {
                heard = Some(found);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let heard = heard.expect("the announcement was not heard");
        assert!(heard.slim);
        let skin = heard.skin.unwrap();
        let cape = heard.cape.unwrap();
        let address = |url: &str| -> (SocketAddr, String) {
            let rest = url.strip_prefix("http://").unwrap();
            let (host, path) = rest.split_once('/').unwrap();
            (host.parse().unwrap(), format!("/{path}"))
        };
        let (addr, path) = address(&skin);
        let (status, body) = get(addr, &path);
        assert!(status.contains("200"), "{status}");
        assert_eq!(body, b"skin-bytes");
        let (addr, path) = address(&cape);
        assert_eq!(get(addr, &path).1, b"cape-bytes");
        // Another hash, another kind and a path that climbs are all refused.
        let wrong = format!("/skin/{}.png", "0".repeat(40));
        assert!(get(addr, &wrong).0.contains("404"));
        assert!(get(addr, "/../etc/passwd").0.contains("404"));
        // A changed skin is announced under its new hash.
        share.set(Look {
            skin: Some(b"new-skin".to_vec()),
            ..look()
        });
        let mut changed = false;
        for _ in 0..60 {
            if peers.look_of("maou").and_then(|l| l.skin).is_some_and(|u| u != skin) {
                changed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(changed, "the new skin was not announced");
    }
}
