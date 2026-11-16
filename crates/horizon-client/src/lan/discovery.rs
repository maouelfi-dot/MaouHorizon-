//! LAN game discovery: LanServerPinger announces a hosted world on the
//! multicast group 224.0.2.60:4445 every 1.5 seconds as
//! `[MOTD]text[/MOTD][AD]port[/AD]`, and LanServerDetection collects those
//! announcements for the multiplayer screen.
use std::{
    collections::HashMap,
    io,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

pub const GROUP: Ipv4Addr = Ipv4Addr::new(224, 0, 2, 60);
pub const PORT: u16 = 4445;
/// LanServerPinger sleeps 1500 ms between announcements.
const INTERVAL: Duration = Duration::from_millis(1500);

/// LanServerPinger.createPingString.
pub fn ping_string(motd: &str, port: u16) -> String {
    ping_string_text(motd, &port.to_string())
}
/// LanServerPinger.createPingString with the address as text.
pub fn ping_string_text(motd: &str, address: &str) -> String {
    format!("[MOTD]{motd}[/MOTD][AD]{address}[/AD]")
}
/// String.indexOf(needle, from) on a Rust string (byte offsets).
fn index_of(s: &str, needle: &str, from: usize) -> Option<usize> {
    s.get(from..)?.find(needle).map(|i| i + from)
}
/// LanServerPinger.parseMotd; "missing no" when there is none.
pub fn parse_motd(s: &str) -> String {
    let Some(start) = s.find("[MOTD]") else {
        return "missing no".to_owned();
    };
    match index_of(s, "[/MOTD]", start + "[MOTD]".len()) {
        Some(end) => s[start + "[MOTD]".len()..end].to_owned(),
        None => "missing no".to_owned(),
    }
}
/// LanServerPinger.parseAddress: the text between [AD] and [/AD] after the
/// only [/MOTD], as announced (vanilla does not check it is a port).
pub fn parse_address(s: &str) -> Option<String> {
    let end_motd = s.find("[/MOTD]")?;
    if index_of(s, "[/MOTD]", end_motd + "[/MOTD]".len()).is_some() {
        return None;
    }
    let start = index_of(s, "[AD]", end_motd + "[/MOTD]".len())?;
    let end = index_of(s, "[/AD]", start + "[AD]".len())?;
    Some(s[start + "[AD]".len()..end].to_owned())
}
/// LanServerDetection.LanServerList.addServer: an announcement from `ip`.
/// A new address joins the list; a known one keeps its first MOTD. True
/// when the list changed.
pub fn add_server(games: &mut Vec<LanGame>, ping: &str, ip: &str) -> bool {
    let motd = parse_motd(ping);
    let Some(address) = parse_address(ping) else {
        return false;
    };
    let address = format!("{ip}:{address}");
    if games.iter().any(|g| g.address == address) {
        return false;
    }
    games.push(LanGame { motd, address });
    true
}

/// The subnet broadcast address of `ip` on a network with `netmask`.
pub fn broadcast_address(ip: Ipv4Addr, netmask: Ipv4Addr) -> Ipv4Addr {
    Ipv4Addr::from(u32::from(ip) | !u32::from(netmask))
}

/// The IPv4 networks this device is on, loopback and link-local left out: `(address, broadcast)`.
/// A phone's hotspot is one of them while it is on, and the multicast group alone does not
/// reach it (the group goes out of the default network only), so announcements also go to each
/// network's broadcast address (Maou Horizon's "Mabar").
pub fn local_networks() -> Vec<(Ipv4Addr, Ipv4Addr)> {
    let Ok(interfaces) = if_addrs::get_if_addrs() else {
        return Vec::new();
    };
    interfaces
        .into_iter()
        .filter_map(|interface| match interface.addr {
            if_addrs::IfAddr::V4(v4) if !v4.ip.is_loopback() && !v4.ip.is_link_local() => {
                let broadcast = v4
                    .broadcast
                    .unwrap_or_else(|| broadcast_address(v4.ip, v4.netmask));
                Some((v4.ip, broadcast))
            }
            _ => None,
        })
        .collect()
}

/// Announces one hosted world until dropped.
pub struct Pinger {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Pinger {
    pub fn start(motd: String, port: u16) -> io::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", 0))?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let message = ping_string(&motd, port);
        let thread = std::thread::Builder::new()
            .name("horizon-lan-ping".into())
            .spawn(move || {
                let target = SocketAddr::V4(SocketAddrV4::new(GROUP, PORT));
                // One socket per network, bound to its own address so the announcement
                // names that address: the one a guest on that network can reach.
                let mut per_network: HashMap<Ipv4Addr, UdpSocket> = HashMap::new();
                while !flag.load(Ordering::Relaxed) {
                    let _ = socket.send_to(message.as_bytes(), target);
                    let networks = local_networks();
                    per_network.retain(|ip, _| networks.iter().any(|(n, _)| n == ip));
                    for (ip, broadcast) in networks {
                        let network_socket = match per_network.entry(ip) {
                            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                            std::collections::hash_map::Entry::Vacant(e) => {
                                let Ok(s) = UdpSocket::bind((ip, 0)) else {
                                    continue;
                                };
                                let _ = s.set_broadcast(true);
                                e.insert(s)
                            }
                        };
                        let _ = network_socket.send_to(
                            message.as_bytes(),
                            SocketAddr::V4(SocketAddrV4::new(broadcast, PORT)),
                        );
                    }
                    let mut slept = Duration::ZERO;
                    while slept < INTERVAL && !flag.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(50));
                        slept += Duration::from_millis(50);
                    }
                }
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Pinger {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A world announced on the local network.
#[derive(Clone, Debug, PartialEq)]
pub struct LanGame {
    pub motd: String,
    pub address: String,
}
type Games = Arc<Mutex<Vec<LanGame>>>;

/// Listens for announcements until the last clone is dropped.
#[derive(Clone, Debug)]
pub struct Scanner {
    inner: Arc<ScanInner>,
}
#[derive(Debug)]
struct ScanInner {
    games: Games,
    stop: Arc<AtomicBool>,
    /// How many games `take_found` has handed out.
    taken: Mutex<usize>,
}
impl Drop for ScanInner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
impl Scanner {
    /// Bind the discovery port; without it the list simply stays empty.
    pub fn start() -> Self {
        let games: Games = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        // Constructing multiplayer screens in unit tests must not open the LAN.
        if !cfg!(test)
            && !super::test_loopback()
            && let Ok(socket) = bind_listener()
        {
            let (games, stop) = (games.clone(), stop.clone());
            let _ = std::thread::Builder::new()
                .name("horizon-lan-scan".into())
                .spawn(move || listen(socket, games, stop));
        }
        Self {
            inner: Arc::new(ScanInner {
                games,
                stop,
                taken: Mutex::new(0),
            }),
        }
    }
    #[cfg(test)]
    pub(crate) fn for_tests(games: Vec<LanGame>) -> Self {
        Self {
            inner: Arc::new(ScanInner {
                games: Arc::new(Mutex::new(games)),
                stop: Arc::new(AtomicBool::new(false)),
                taken: Mutex::new(0),
            }),
        }
    }
    /// The worlds announced since the scan started, in order (vanilla's
    /// list keeps them while the multiplayer screen is open).
    pub fn games(&self) -> Vec<LanGame> {
        self.inner.games.lock().unwrap().clone()
    }
    /// `LanServerList.takeDirty` with `updateNetworkServers`' new count: the
    /// games found since the last call, with their place in [`games`](Self::games).
    pub fn take_found(&self) -> Vec<(usize, LanGame)> {
        let games = self.inner.games.lock().unwrap();
        let mut taken = self.inner.taken.lock().unwrap();
        let found = games.iter().cloned().enumerate().skip(*taken).collect();
        *taken = games.len();
        found
    }
}
fn bind_listener() -> io::Result<UdpSocket> {
    let socket = UdpSocket::bind(("0.0.0.0", PORT))?;
    socket.join_multicast_v4(&GROUP, &Ipv4Addr::UNSPECIFIED)?;
    // Also on each network (a hotspot's included); broadcasts reach the bound port as they are.
    for (ip, _) in local_networks() {
        let _ = socket.join_multicast_v4(&GROUP, &ip);
    }
    socket.set_read_timeout(Some(Duration::from_millis(250)))?;
    Ok(socket)
}
fn listen(socket: UdpSocket, games: Games, stop: Arc<AtomicBool>) {
    let mut buf = [0u8; 1024];
    while !stop.load(Ordering::Relaxed) {
        let Ok((n, from)) = socket.recv_from(&mut buf) else {
            continue;
        };
        let text = String::from_utf8_lossy(&buf[..n]);
        add_server(&mut games.lock().unwrap(), &text, &from.ip().to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `takeDirty`: each world found is handed out once, in order.
    #[test]
    fn found_worlds_are_taken_once() {
        let scanner = Scanner {
            inner: Arc::new(ScanInner {
                games: Arc::default(),
                stop: Arc::default(),
                taken: Mutex::new(0),
            }),
        };
        let found = |s: &Scanner| -> Vec<usize> { s.take_found().iter().map(|f| f.0).collect() };
        assert!(found(&scanner).is_empty());
        add_server(
            &mut scanner.inner.games.lock().unwrap(),
            &ping_string("A", 1),
            "10.0.0.2",
        );
        add_server(
            &mut scanner.inner.games.lock().unwrap(),
            &ping_string("B", 2),
            "10.0.0.3",
        );
        assert_eq!(found(&scanner), [0, 1]);
        assert!(found(&scanner).is_empty());
        add_server(
            &mut scanner.inner.games.lock().unwrap(),
            &ping_string("C", 3),
            "10.0.0.4",
        );
        assert_eq!(scanner.take_found()[0].1.motd, "C");
    }

    #[test]
    fn the_broadcast_address_fills_the_host_bits() {
        let mask = Ipv4Addr::new(255, 255, 255, 0);
        assert_eq!(
            broadcast_address(Ipv4Addr::new(192, 168, 43, 1), mask),
            Ipv4Addr::new(192, 168, 43, 255)
        );
        assert_eq!(
            broadcast_address(Ipv4Addr::new(10, 1, 2, 3), Ipv4Addr::new(255, 255, 0, 0)),
            Ipv4Addr::new(10, 1, 255, 255)
        );
    }

    #[test]
    fn local_networks_leave_out_loopback() {
        assert!(local_networks().iter().all(|(ip, _)| !ip.is_loopback()));
    }

    #[test]
    fn ping_string_round_trips() {
        let s = ping_string("Steve - World", 25565);
        assert_eq!(s, "[MOTD]Steve - World[/MOTD][AD]25565[/AD]");
        assert_eq!(parse_motd(&s), "Steve - World");
        assert_eq!(parse_address(&s).as_deref(), Some("25565"));
        assert_eq!(parse_address("[AD]1[/AD]"), None);
        assert_eq!(parse_address("[/MOTD][AD]x[/AD][/MOTD]"), None);
        assert_eq!(parse_motd("junk"), "missing no");
        let mut games = Vec::new();
        assert!(add_server(&mut games, &s, "10.0.0.2"));
        assert!(!add_server(
            &mut games,
            &ping_string("Other", 25565),
            "10.0.0.2"
        ));
        assert_eq!(games[0].motd, "Steve - World");
    }
}
