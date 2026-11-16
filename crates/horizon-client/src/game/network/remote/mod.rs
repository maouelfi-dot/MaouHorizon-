//! Multiplayer: the game driven by a server connection instead of the local
//! simulation. Mirrors vanilla's ClientPacketListener (chunks, entities,
//! inventory, health, time, weather, chat, respawn and dimension changes),
//! LocalPlayer.sendPosition (movement packets) and MultiPlayerGameMode
//! (block breaking with sequence numbers, use-item, attacks, container
//! clicks). The player still moves with the local physics, as the vanilla
//! client predicts its own movement; the server stays authoritative for
//! everything else.
mod attack_target;
mod recipe_properties;

use super::remote_chunks::{ChunkDecoder, ChunkFormat, Next};
use super::render_tick::{mob_tracking_body, player_tracking_body};
use super::{Dim, Game, Mining, Screen, level_sounds::LevelSound};
use crate::game::api_values::TryIntoApi;
use horizon_client::actions::Action;
use horizon_client::first_person::{Hand, SwingAnimation};
use horizon_client::gamepad::ServerStick;
use horizon_client::{
    Result,
    blocks::Fluid,
    chunk::Chunk,
    containers::{BlockEntity, Furnace, FurnaceKind},
    entities::{Experience, ExperienceOrb, ItemEntity},
    hud::HudBatch,
    items::ItemStack,
    lan::data::{BIOME_REGISTRY, client_biomes},
    menu::Input,
    mobs::{Mob, MobType, Who},
    motion::{MotionWorld, PlayerMotion},
    net::{Event, Session},
    particles::TrackingTarget,
    random_source,
    save::SaveDir,
    screen_effects,
    sprint::Keys,
    streaming::{ChunkGenerator, Proto, Streaming},
    tab_list::SPECTATOR,
    world::World,
};
use horizon_core::{geometry::Vec3, random::RandomSource};
use horizon_protocol::{
    entity_data::{ITEM_ENTITY_ITEM, Value},
    item::{Hashed, Stack},
    packets::{
        Empty,
        play::{self, Clientbound as C, Serverbound as S},
    },
    registry::{Builtin, DimensionType, Dynamic, SyncedNames, ceil_log2},
    rsa::RsaPublicKey,
    secure_chat::{self, Signature},
    text::{Language, plain},
};
use mod_api::InteractionKey;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{event::MouseButton, keyboard::KeyCode};
use {
    horizon_core::nbt::Tag,
    horizon_mod_runtime::attachments::{Holder, ResidentAttachments},
    horizon_world::{
        level::InteractionResult,
        registry::BlockStateId,
        stepped_interpolation::{Path, PosRot, Start, SteppedInterpolation},
    },
};

mod client_level_tick;

type BuiltinCache = Vec<(std::path::PathBuf, Arc<Builtin>)>;
static BUILTIN: std::sync::Mutex<BuiltinCache> = std::sync::Mutex::new(Vec::new());

/// The prepared resources' `registries.json`, parsed once per root: the
/// file never changes while the game runs, and parsing it took about 3 ms
/// of the frame that joins a server (or opens a single-player world).
/// `None` when it is missing or invalid (not cached: a fixed install can
/// join again).
pub(super) fn builtin_registries(root: &std::path::Path) -> Option<Arc<Builtin>> {
    let mut cache = BUILTIN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, builtin)) = cache.iter().find(|(cached, _)| cached == root) {
        return Some(builtin.clone());
    }
    let builtin = Arc::new(Builtin::find(root).and_then(|p| Builtin::load(&p).ok())?);
    cache.push((root.to_owned(), builtin.clone()));
    Some(builtin)
}

/// `ChatType`'s two decorations: how a message shows in the chat, and what
/// the narrator says of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Decoration {
    Chat,
    Narration,
}

/// `ChatTypeDecoration.decorate` of a chat type's `chat` or `narration`
/// entry: its translation, with the sender, the target or the message as its
/// parameters, in its style. A chat type the registry lacks decorates as
/// `minecraft:chat` (`chat.type.text`, narrated `chat.type.text.narrate`).
fn chat_type_decorate(
    chat_type: Option<&Tag>,
    decoration: Decoration,
    bound: &play::ChatTypeBound,
    content: &serde_json::Value,
) -> serde_json::Value {
    use horizon_client::scoreboard::tag_to_json;
    let (field, default_key) = match decoration {
        Decoration::Chat => ("chat", "chat.type.text"),
        Decoration::Narration => ("narration", "chat.type.text.narrate"),
    };
    let decoration = chat_type.and_then(|t| t.get(field));
    let key = match decoration.and_then(|d| d.get("translation_key")) {
        Some(Tag::String(s)) => s.to_string_lossy(),
        _ => default_key.into(),
    };
    let parameter = |name: &str| match name {
        "sender" => tag_to_json(&bound.name),
        "target" => bound
            .target
            .as_ref()
            .map_or_else(|| serde_json::Value::from(""), tag_to_json),
        _ => content.clone(),
    };
    let with: Vec<serde_json::Value> = match decoration.and_then(|d| d.get("parameters")) {
        Some(Tag::List(list)) => list
            .iter()
            .map(|p| match p {
                Tag::String(s) => parameter(&s.to_string_lossy()),
                _ => serde_json::Value::from(""),
            })
            .collect(),
        _ => vec![parameter("sender"), parameter("content")],
    };
    let mut component = match decoration.and_then(|d| d.get("style")) {
        Some(style @ Tag::Compound(_)) => tag_to_json(style),
        _ => serde_json::json!({}),
    };
    component["translate"] = key.into();
    component["with"] = with.into();
    component
}

/// A text component the server sent, for the HUD and the chat. Legacy `Â§` codes in its strings
/// (which plugins still send) style the text as the font's StringDecomposer does.
pub(super) fn remote_text(tag: &Tag) -> horizon_client::text::Text {
    use horizon_client::scoreboard::{comp_text, tag_to_json};
    comp_text(&tag_to_json(tag)).resolve_legacy()
}

/// `Player.getVoicePitch`: a constant 1.0
/// (`javap -p -c -constants net.minecraft.world.entity.player.Player`:
/// `0: fconst_1  1: freturn`), so a player's own hurt and death sounds are
/// neither raised for a baby nor jittered - `LivingEntity.getVoicePitch`, which
/// the mobs port, does both. It takes no random, so it draws nothing from the
/// level random either.
pub(super) fn player_voice_pitch() -> f32 {
    1.0
}

/// The hurt sound of a `ClientboundDamageEventPacket`'s damage type, by its
/// synchronized registry id: `Player.getHurtSound` picks the type's effects.
fn hurt_sound_for(
    registries: &Dynamic,
    damage: &horizon_world::damage_source::Catalog,
    source_type: i32,
) -> &'static str {
    if let Some(data) = registries.data("minecraft:damage_type", source_type) {
        let effects = data.get("effects").and_then(|t| match t {
            Tag::String(effects) => effects.to_string().ok(),
            _ => None,
        });
        return horizon_client::health::hurt_sound(effects.as_deref().unwrap_or("hurt"));
    }
    registries
        .name("minecraft:damage_type", source_type)
        .and_then(|name| damage.id(name).and_then(|id| damage.definition(id)))
        .map_or(horizon_client::health::hurt_sound("hurt"), |d| {
            horizon_client::health::hurt_sound(&d.effects)
        })
}

/// The live entity a bound sound names; absence means that object was removed.
fn entity_sound_state(
    id: i32,
    player: Option<(i32, [f64; 3], bool)>,
    mob: Option<horizon_client::sound_engine::SoundEntityState>,
    tracked: &HashMap<i32, Tracked>,
    dragon: Option<(i32, [f64; 3])>,
) -> Option<horizon_client::sound_engine::SoundEntityState> {
    use horizon_client::sound_engine::SoundEntityState;
    if let Some((player_id, position, silent)) = player
        && player_id == id
    {
        return Some(SoundEntityState {
            position,
            silent,
            removed: false,
        });
    }
    let mut state = mob.or_else(|| {
        tracked.get(&id).map(|t| SoundEntityState {
            position: t.position,
            silent: t.silent,
            removed: false,
        })
    })?;
    // The dragon's Mob is metadata storage; only Dragon follows its movement.
    if let Some((dragon_id, position)) = dragon
        && dragon_id == id
    {
        state.position = position;
    }
    Some(state)
}

/// `ClientboundOpenScreenPacket.title` as a container screen draws it: the
/// component's styles, and the `Â§` codes a literal holds (`StringDecomposer`).
fn menu_title(tag: &Tag) -> horizon_client::text::Text {
    remote_text(tag).resolve_legacy()
}

/// A player message Chat Delay holds back, its signature and trust level,
/// what the narrator says of it once it shows, and who sent it in which
/// chat type, for mods told when it shows (`ClientChatReceivedEvent`).
struct QueuedChat {
    line: horizon_client::text::Text,
    signature: Option<Signature>,
    trust: ChatTrust,
    narration: String,
    /// The player's UUID, 0 for disguised chat (`Util.NIL_UUID`).
    sender: u128,
    /// The chat type's network ID (`ChatType.Bound.chatType`).
    chat_type: i32,
}

/// `ChatTrustLevel`; a modified message keeps the signed content it was
/// changed from.
pub(super) enum ChatTrust {
    Secure,
    Modified(String),
    NotSecure,
    /// `GuiMessageTag.chatError`: the line `ChatListener.handleChatMessageError` adds.
    Error,
}
impl ChatTrust {
    /// `ChatTrustLevel.createTag`.
    fn tag(&self) -> Option<horizon_client::chat::Tag> {
        match self {
            Self::Secure => None,
            Self::Modified(original) => Some(horizon_client::chat::Tag::modified(original)),
            Self::NotSecure => Some(horizon_client::chat::Tag::not_secure()),
            Self::Error => Some(horizon_client::chat::Tag::chat_error()),
        }
    }
}

/// Streaming needs a generator; a server world never generates.
struct NoTerrain;
impl ChunkGenerator for NoTerrain {
    type Worker = ();
    fn worker(&self) {}
    fn proto(&self, _: &mut (), _: i32, _: i32) -> Proto {
        Box::new(())
    }
    fn finish(&self, _: Proto) -> Chunk {
        Chunk::empty(0, 0)
    }
}

/// A server entity the client tracks (position as last sent).
pub(super) struct Tracked {
    pub(super) kind: String,
    pub(super) position: [f64; 3],
    /// `Entity.getPositionCodec`'s base: the position the server last
    /// reported (AddEntity, MoveEntity, EntityPositionSync), which a move
    /// packet's delta counts from. A TeleportEntity moves the entity but not
    /// this base, so the server's next delta still lands where it means.
    codec_base: [f64; 3],
    silent: bool,
    /// `Entity.vehicle`: the entity it rides (`SetPassengers`), whatever
    /// either one's type.
    vehicle: Option<u32>,
}

/// `Entity.startRiding` / `removePassenger` for a `SetPassengers` packet: the
/// listed entities ride `vehicle`, the ones that rode it and are no longer
/// listed got off. Every tracked entity counts, whatever its type, so a
/// server's item display or marker on a player hides that player's name tag
/// as vanilla's `isVehicle` does (#5666).
fn seat_passengers(entities: &mut HashMap<i32, Tracked>, vehicle: u32, passengers: &[i32]) {
    for (id, t) in entities.iter_mut() {
        if passengers.contains(id) {
            t.vehicle = Some(vehicle);
        } else if t.vehicle == Some(vehicle) {
            t.vehicle = None;
        }
    }
}
/// Smooth an entity toward a new position over three ticks (InterpolationHandler).
#[derive(Clone, Copy)]
/// `LinearInterpolationHandler`'s target: a position and rotation (yRot,
/// xRot) reached in `steps` ticks.
pub(super) struct Lerp {
    target: [f64; 3],
    rotation: Option<(f32, f32)>,
    steps: u32,
}

/// `NewMinecartBehavior.MinecartStep`, including the server's rail pitch.
#[derive(Clone, Copy)]
struct CartStep {
    position: Vec3,
    movement: Vec3,
    yaw: f32,
    pitch: f32,
    weight: f32,
}

/// The client's state of a minecart on `NewMinecartBehavior`: the steps the
/// server sent (`lerpSteps`), the ones being played over 3 ticks
/// (`currentLerpSteps`) and where the cart was when they began (`oldLerp`).
pub(super) struct CartLerp {
    pending: Vec<CartStep>,
    current: Vec<CartStep>,
    total_weight: f64,
    delay: i32,
    old: CartStep,
}

impl CartLerp {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            current: Vec::new(),
            total_weight: 0.0,
            delay: 0,
            old: CartStep {
                position: Vec3::ZERO,
                movement: Vec3::ZERO,
                yaw: 0.0,
                pitch: 0.0,
                weight: 0.0,
            },
        }
    }

    /// `NewMinecartBehavior.getCartLerpPosition`, `getCartLerpMovements` and
    /// `getCartLerpYRot` / `getCartLerpXRot` at `partial`: the step the weights
    /// put this moment in, and how far along it from the one before.
    fn at(&self, partial: f32) -> (Vec3, Vec3, f32, f32) {
        let alpha = ((3 - self.delay) as f32 + partial) / 3.0;
        let mut count_up = 0.0f32;
        let mut in_step = 1.0f32;
        let mut found = None;
        for (i, s) in self.current.iter().enumerate() {
            if s.weight <= 0.0 {
                continue;
            }
            count_up += s.weight;
            if f64::from(count_up) >= self.total_weight * f64::from(alpha) {
                let before = count_up - s.weight;
                in_step = ((f64::from(alpha) * self.total_weight - f64::from(before))
                    / f64::from(s.weight)) as f32;
                found = Some(i);
                break;
            }
        }
        let index = found.unwrap_or(self.current.len() - 1);
        let current = self.current[index];
        let previous = if index > 0 {
            self.current[index - 1]
        } else {
            self.old
        };
        let t = f64::from(in_step);
        let lerp = |a: Vec3, b: Vec3| {
            Vec3::new(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
        };
        (
            lerp(previous.position, current.position),
            lerp(previous.movement, current.movement),
            previous.yaw + in_step * wrap_degrees(current.yaw - previous.yaw),
            previous.pitch + in_step * wrap_degrees(current.pitch - previous.pitch),
        )
    }

    /// `NewMinecartBehavior.lerpClientPositionAndRotation`: once the steps
    /// being played are used up, the queued ones become the current ones;
    /// the cart then sits where the steps put it at the end of this tick.
    /// False when it has no steps to play.
    fn tick(&mut self, c: &mut super::minecarts::Minecart) -> bool {
        self.delay -= 1;
        if self.delay <= 0 {
            self.old = CartStep {
                position: c.position,
                movement: c.velocity,
                yaw: c.yaw,
                pitch: c.pitch,
                weight: 0.0,
            };
            self.current.clear();
            if !self.pending.is_empty() {
                self.current.append(&mut self.pending);
                self.total_weight = self.current.iter().map(|s| f64::from(s.weight)).sum();
                self.delay = if self.total_weight == 0.0 { 0 } else { 3 };
            }
        }
        if self.current.is_empty() {
            return false;
        }
        let (position, movement, yaw, pitch) = self.at(1.0);
        c.position = position;
        c.velocity = movement;
        c.yaw = yaw;
        c.pitch = pitch;
        true
    }
}

/// `SteppedInterpolationHandler.doInterpolate`'s `setPos` and `setRot` on a
/// mob: its yaw kept continuous (the same turn as vanilla's, without the
/// wrap that would spin it in the frame between ticks), its pitch clamped.
fn place_interpolated(m: &mut Mob, at: PosRot) {
    m.position = Vec3::new(at.position[0], at.position[1], at.position[2]);
    m.yaw += wrap_degrees(at.yaw - m.yaw);
    m.head_pitch = (at.pitch % 360.0).clamp(-90.0, 90.0);
}

/// The packet a living entity's move came in.
#[derive(Clone, Copy, PartialEq)]
enum MovePacket {
    /// `handleMoveEntity`: always interpolated.
    Move,
    /// `handleEntityPositionSync`.
    PositionSync,
    /// `handleTeleportEntity` (`setValuesFromPositionPacket`).
    Teleport,
}

impl MovePacket {
    /// `tooBigToInterpolate`: a move to `to` of more than 64 blocks snaps.
    /// A position sync measures it from where the entity is, a teleport from
    /// where it is headed (`PositionMoveRotation.of`'s
    /// `getClientPositionAndRotation`).
    fn snaps(self, handler: &SteppedInterpolation, entity: PosRot, to: [f64; 3]) -> bool {
        let from = match self {
            MovePacket::Move => return false,
            MovePacket::PositionSync => entity.position,
            MovePacket::Teleport => handler.heading(entity).position,
        };
        let d = [0, 1, 2].map(|i| to[i] - from[i]);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > 4096.0
    }
}

/// A move `packet` to `to` for a living entity at `entity`, through its
/// handler (`interpolateTo`): where it is put at once, if anywhere. A move
/// too big to interpolate is a `snapTo`; vanilla leaves the handler running
/// and its next step shifts what is left by the jump
/// (`adjustInterpolationTargetFromDeltas`), which this handler doesn't port,
/// so the steps are dropped instead. That is the same for an entity that
/// wasn't moving, as a teleported one usually isn't.
fn stepped_move(
    handler: &mut SteppedInterpolation,
    packet: MovePacket,
    entity: PosRot,
    to: [f64; 3],
    path: Option<Path>,
    rotation: Option<(f32, f32)>,
) -> Option<PosRot> {
    if packet.snaps(handler, entity, to) {
        handler.cancel();
        let (yaw, pitch) = rotation.unwrap_or((entity.yaw, entity.pitch));
        return Some(PosRot {
            position: to,
            yaw,
            pitch,
        });
    }
    match handler.interpolate_to(entity, path, rotation) {
        Start::Snap(at) => Some(at),
        Start::Interpolating | Start::Unchanged => None,
    }
}

/// What the network thread delivered, in order: an event, or a chunk packet
/// handed to the decoder (with the format it decodes with), whose column
/// comes back from `ChunkDecoder::next`.
enum Received {
    Event(Event),
    Chunk(Arc<ChunkFormat>),
}

/// `Minecraft.services().profileKeySignatureValidator()`: the services' profile-key keys, which a
/// player's chat key must be signed by. Asked for once per connection to a server, off the game
/// thread.
pub(super) enum ServicesKeys {
    Pending,
    Ready(Vec<RsaPublicKey>),
    /// `canValidateProfileKeys()` is false: none could be had.
    Missing,
}
/// The signed-in account's `/minecraft/profile` fetch after joining.
enum OwnProfile {
    Idle,
    Fetching(std::sync::mpsc::Receiver<Option<horizon_client::skins::Textures>>),
    Done(Option<horizon_client::skins::Textures>),
}

pub(super) struct Remote {
    pub session: Session,
    pub brand: String,
    pub(super) recipe_transfer: bool,
    /// This connection's configured feature flags, carried through reconfiguration.
    pub(super) features: std::sync::Arc<[String]>,
    /// The server announced it runs mods' quiet commands.
    pub quiet_commands: bool,
    /// The server announced it carries mod messages.
    pub mod_net: bool,
    /// The synced attachments the server sent, by holder.
    pub attachments: HashMap<Holder, ResidentAttachments>,
    /// The server samples far terrain for us, this far at most (`lod_net`).
    pub far_terrain: Option<horizon_client::lod_net::Offer>,
    pub(super) builtin: Arc<Builtin>,
    lang: Arc<Language>,
    pub(super) registries: Arc<Dynamic>,
    /// The block, item and fluid tags of `registries` by name, as native mods read them
    /// (`install_remote_tags`).
    pub(super) tags: Option<horizon_client::items::Items>,
    /// The game's enchantments and the server's that it lacks (a data
    /// pack's), from their synced data; `None` when the server has none.
    pub(super) enchantments: Option<Arc<horizon_client::enchant::Enchantments>>,
    pub(super) joined: bool,
    /// `ServerReconfigScreen` is up since this tick.
    pub(super) reconfiguring: Option<u64>,
    /// `handleLogin` ran: vanilla's `Minecraft.player` exists from then on,
    /// while here the player is made at the first teleport.
    pub(super) logged_in: bool,
    status: String,
    pub(super) entity_id: i32,
    local_entity_silent: bool,
    /// `Minecraft.cameraEntity` when the server put the view on another
    /// entity (ClientboundSetCameraPacket); None while it is the player.
    pub(super) camera: Option<i32>,
    /// The item the player's use began with (`LivingEntity.useItem`).
    pub(super) use_item: Option<Arc<str>>,
    pub(super) dimension: DimensionType,
    /// The level's key (`minecraft:overworld`): where its far terrain is kept.
    pub(in crate::game) level: String,
    /// `level`'s key, as mods' handles name the level.
    pub(super) level_key: mod_api::DimensionKey,
    /// The level's facts from the dimension type this connection's server sent (`None` when it
    /// sent none the game takes): owned by the connection, so entering a level touches no other
    /// level's facts, the integrated server's included.
    pub(in crate::game) level_facts: Option<Arc<horizon_world::dimension_type::DimensionFacts>>,
    /// The sky, fog and cloud attributes (and timelines) of the level from the server's type, for a
    /// level of a mod or a pack (`None` for a built-in level, or when the type is refused).
    pub(in crate::game) level_environment: Option<horizon_client::environment::Environment>,
    /// Connection-owned facts, cached by level and the full named type identity.
    facts_cache: HashMap<
        String,
        (
            String,
            serde_json::Value,
            Arc<horizon_world::dimension_type::DimensionFacts>,
        ),
    >,
    /// `ClientPlayerNetworkEvent.LoggingOut` was fired for this connection: every way out
    /// of a world reports it once.
    logged_out: bool,
    /// The level's hashed seed (login, respawn): the folder of its kept chunks
    /// (`server_terrain::level_dir`).
    pub(super) hashed_seed: i64,
    /// What chunk packets decode with (the dimension's height and sky, the
    /// biome registry); replaced whenever one of them changes.
    pub(super) chunk_format: Arc<ChunkFormat>,
    /// Decodes chunk packets as they arrive, off the frame.
    decoder: ChunkDecoder,
    /// Options.serverRenderDistance: the server's view distance from the
    /// login or the last chunk-cache-radius packet.
    pub(super) view_distance: i32,
    /// `ClientLevel.serverSimulationDistance`: from the login or the last
    /// simulation-distance packet.
    pub(super) simulation_distance: i32,
    /// `BlockStatePredictionHandler.currentSequenceNr`: the level's, so a new level (a login,
    /// another dimension) counts from 0 again.
    sequence: i32,
    /// `BlockStatePredictionHandler.lastTeleportSequence`: the sequence of the last
    /// teleport, so a prediction made before it keeps its block but not its snap.
    last_teleport_sequence: i32,
    position_sender: horizon_client::player_sync::PositionSender,
    /// When the last tick ended to the server, and the last movement packet went: an old
    /// server behind ViaVersion (its anticheat) wants one every tick, so gaps are logged.
    tick_end_at: Option<Instant>,
    moved_at: Option<Instant>,
    /// When the play phase began (login or the end of a reconfiguration).
    play_began: Option<Instant>,
    /// This second's movement packets and ticks ended, from when: a second over 20 of
    /// either is logged (an old server's anticheat counts them against the time passed).
    rate: (Option<Instant>, u32, u32),
    /// The last movement packets sent: position, on ground, input flags and when. The
    /// first setback logs them, the path an anticheat refused.
    trail: std::collections::VecDeque<([f64; 3], bool, u8, Instant)>,
    /// Free stick direction (experimental, `stick_steer`): this tick's rotation the server sees
    /// in place of the view's, and when the player last hit, used or dropped something.
    steer_yaw: Option<f32>,
    interacted_at: std::cell::Cell<Option<Instant>>,
    /// The look a teleport's acknowledgement told, for `PositionSender::told_rotation`.
    told_rotation: std::cell::Cell<Option<(f32, f32)>>,
    /// Hits and uses waiting for a move with the view's look (`send`).
    held_back: std::cell::RefCell<Vec<S>>,
    /// The stick's mode here this tick ([`ServerStick`]): this session's choice, else the one
    /// remembered for the server (read once), else the setting.
    stick_mode: horizon_client::gamepad::ServerStick,
    stick_choice: Option<horizon_client::gamepad::ServerStick>,
    stick_remembered: Option<Option<horizon_client::gamepad::ServerStick>>,
    /// The setting as last seen here: changed while playing, it becomes the mode here.
    stick_setting: Option<horizon_client::gamepad::ServerStick>,
    /// Setbacks while the stick moved the player, for asking once a session how it should.
    stick_setbacks: std::collections::VecDeque<Instant>,
    stick_used_at: Option<Instant>,
    stick_asked: bool,
    /// key.swapOffhand presses since the last tick, which sends them (`remote_handle_input`).
    swap_presses: u32,
    last_input: u8,
    was_sprinting: bool,
    /// The slot the server last heard (`MultiPlayerGameMode.carriedIndex`): a cell, so `send` can
    /// send a new one first (`ensureHasSentCarriedItem`).
    carried: std::cell::Cell<Option<usize>>,
    was_shared_flags: u8,
    was_living_flags: u8,
    last_recorded_pose: Option<crate::pose::Pose>,
    /// The last ClientboundPlayerAbilitiesPacket. Vanilla's LocalPlayer
    /// holds its abilities from the login on; here the player is made at
    /// the first teleport, which takes them then. None after a respawn
    /// until the server sends them: the game mode's then
    /// (`GameType.updatePlayerAbilities`).
    pub(super) abilities: Option<play::PlayerAbilities>,
    loaded_sent: bool,
    /// Ticks the player has waited for the chunk under it (logged when long: the player
    /// cannot move meanwhile).
    waiting_ticks: u32,
    /// Server teleports of the player that took it back a short way (an anti-cheat's or the
    /// server's correction), logged so a player who "cannot walk" shows why.
    setbacks: u32,
    /// The player's attributes as the server last sent them (ClientboundUpdateAttributesPacket):
    /// movement speed, jump strength, gravity, step height... kept for a LocalPlayer made
    /// after the packet came, and across respawns.
    player_attributes: Vec<play::AttributeSnapshot>,
    /// The dimension of the client level the mods heard made
    /// (`native_client_level_load`), to be told dropped once.
    heard_level: Option<String>,
    /// `ClientPacketListener.levelLoadTracker`: the loading screen after a login
    /// or respawn, until the player's section is shown.
    pub(super) level_load: horizon_client::level_loading::LevelLoadTracker,
    /// The ClientInformation last sent (`Options.broadcastOptions` sends it
    /// again when a view distance, Chat, skin or Main Hand option changes).
    sent_info: horizon_protocol::packets::configuration::ClientInformation,
    /// BlockStatePredictionHandler: blocks changed ahead of the server (a predicted break),
    /// with what it remembers of each of them. Until the server acknowledges the
    /// sequence its block updates only replace the remembered state; then the block
    /// takes it.
    predictions: HashMap<[i32; 3], Prediction>,
    /// The entities the level held when its `LocalPlayer` was added
    /// (`ClientLevel.addEntity` on login or respawn): `ClientLevel.tickEntities`
    /// ticks in the order entities were added, so these tick before the
    /// player and every later one after it.
    ticked_before_player: std::collections::HashSet<u32>,
    /// Open container: (window id, menu name).
    pub(super) menu: ClientMenu,
    /// The crafting result slot of the open menu as the server last set it
    /// (`CraftingMenu.slotChangedCraftingGrid`): the client never computes it.
    pub(super) craft_result: Option<ItemStack>,
    /// The open menu's title as the server named it, with its styles.
    menu_title: Option<horizon_client::text::Text>,
    menu_title_snapshot: Option<Arc<mod_api::Text>>,
    state_id: i32,
    use_target: Option<[i32; 3]>,
    /// A mod menu's extra data, kept for the `OpenScreen` of its window that
    /// follows (`horizon:menu_extra`).
    menu_extra: Option<horizon_protocol::menu_extra::Message>,
    pub(super) entities: HashMap<i32, Tracked>,
    /// Per-tick remote entity scratch retained between frames.
    tick_fluids: Vec<Option<horizon_client::mobs::living::Fluids>>,
    tick_rain: Vec<bool>,
    tick_standings: Vec<Option<String>>,
    tick_water_fx: Vec<horizon_client::mobs::MobEvent>,
    /// Complete network riding graph, including invisible plugin seat entities.
    pub(super) passenger_vehicles: HashMap<u32, u32>,
    pub(super) lerps: HashMap<i32, Lerp>,
    /// The minecarts the server moves with `ClientboundMoveMinecartPacket`
    /// (`NewMinecartBehavior`), which have no interpolation handler.
    pub(super) cart_lerps: HashMap<i32, CartLerp>,
    /// Living entities' `SteppedInterpolationHandler`s.
    pub(super) stepped: HashMap<i32, SteppedInterpolation>,
    /// `LivingEntity.lerpHeadTo`: the head yaw and the ticks left to reach it.
    pub(super) head_lerps: HashMap<i32, (f32, u32)>,
    /// The happy ghast the player just got off, until the server's teleport
    /// puts the player on top of it (`place_ghast_under`).
    pub(super) ghast_dismount: Option<u32>,
    /// Whether the player is the first passenger of its vehicle, the one
    /// `getControllingPassenger` names.
    pub(super) first_passenger: bool,
    /// The player's place in its vehicle's `getPassengers`.
    pub(super) passenger_index: usize,
    /// Item entities waiting for their item data: where they are and their
    /// motion.
    pub(super) pending_items: HashMap<i32, ([f64; 3], Vec3)>,
    health: (f32, i32, f32),
    /// `LocalPlayer.flashOnSetHealth`: set by the first health the server
    /// sends; from then on a drop starts the hurt animation.
    flash_on_set_health: bool,
    game_mode: i32,
    pub chat: Vec<(String, Instant)>,
    /// Lines received since joining, for the control socket's events.
    pub chat_received: u64,
    /// Entity.DATA_CUSTOM_NAME and DATA_CUSTOM_NAME_VISIBLE as the server set them, by entity
    /// (holograms are armor stands with a visible name).
    pub(super) names: HashMap<i32, (Option<horizon_client::text::Text>, bool)>,
    /// `ChatListener.delayedMessageQueue`: player chat held back by Chat
    /// Delay, with the signature a deletion names it by and what the
    /// narrator says once it shows.
    chat_queue: VecDeque<QueuedChat>,
    /// `ClientPacketListener.messageSignatureCache`: the signatures packed
    /// ones index, as the server's copy for this connection holds them.
    pub(super) signatures: horizon_protocol::secure_chat::SignatureCache,
    /// `ChatListener.previousMessageTime`: when a message last showed, if one has.
    chat_previous: Option<Instant>,
    /// `ClientPacketListener.nextChatIndex`: the global index the next chat packet must carry.
    next_chat_index: i32,
    /// `serverEnforcesSecureChat`, from the login packet.
    enforces_secure_chat: bool,
    services_keys: Arc<std::sync::Mutex<ServicesKeys>>,
    /// The `RemoteChatSession.Data` each player's list entry last carried (INITIALIZE_CHAT).
    chat_sessions: HashMap<u128, Option<(u128, i64, Vec<u8>, Vec<u8>)>>,
    /// `PlayerInfo.messageValidator` of the players whose chat has been read.
    chat_validators: HashMap<u128, secure_chat::MessageValidator>,
    /// `ConnectScreen.lastNarration`: when "Joining the world" was last said.
    joining_narrated: Option<Instant>,
    /// The tree the chat colours, completes and parses with: the server's
    /// (ClientboundCommandsPacket) once it has sent one, else the game's own,
    /// plus the commands native mods own (`chat::chat_tree_for`). Rebuilt where
    /// that base changes, not at each keystroke.
    pub chat_tree: horizon_client::command::Dispatcher,
    /// Whether a `ClientboundCommandsPacket` has supplied the server's tree.
    /// The local fallback is for UI completion only; vanilla signs command
    /// arguments against an empty tree until the server sends one.
    pub(super) chat_tree_received: bool,
    /// The chat session, the last-seen window and the signature cache.
    pub(super) secure_chat: super::remote_chat::SecureChat,
    /// `ClientSuggestionProvider.customCompletionSuggestions`: the extra chat
    /// completions this server sent (ClientboundCustomChatCompletionsPacket).
    custom_completions: std::collections::BTreeSet<String>,
    players: HashMap<u128, String>,
    /// `ClientPacketListener.playerInfoMap` as player-list entries, and
    /// which of them are listed.
    tab_players: HashMap<u128, horizon_client::tab_list::TabPlayer>,
    /// Each listed profile's unpacked `textures` property.
    skin_textures: HashMap<u128, horizon_client::skins::Textures>,
    /// Active skin and cape from the services profile when the server sent none.
    own_profile: OwnProfile,
    /// The skin texture each profile's players show, once its download is ready.
    skins: HashMap<u128, String>,
    skins_revision: u64,
    published_skins: Option<(usize, u128, u64, u64)>,
    listed: std::collections::HashSet<u128>,
    /// `ClientboundLoginPacket.onlineMode`: the player list shows heads.
    pub(super) online_mode: bool,
    /// `ClientboundLoginPacket.levels`: the server's level keys, which
    /// `ClientPacketListener.levels()` hands to command suggestions.
    pub(super) levels: Vec<String>,
    /// Mabar skins while this session is on a LAN (see `own_skin.rs`), dropped with the session.
    pub(super) mabar: Option<horizon_client::lan::mabar_skins::Mabar>,
    /// Player entities in the level: entity id to profile UUID.
    pub(super) level_players: HashMap<i32, u128>,
    /// The profile name each player entity was created with (`RemotePlayer.getGameProfile`): it outlives the
    /// player-list entry, which servers remove to hide their NPCs from the list.
    level_player_names: HashMap<i32, String>,
    /// The `textures` each player entity found in the player list first
    /// (`AbstractClientPlayer.getPlayerInfo` keeps that `PlayerInfo`): its skin
    /// outlives the entry, which NPC plugins remove a moment after spawning.
    level_player_textures: HashMap<i32, horizon_client::skins::Textures>,
    /// Entities with shared flag 6 (`Entity.isCurrentlyGlowing` on a client); mobs keep
    /// theirs in `Mob::glowing`.
    pub(super) glowing: std::collections::HashSet<i32>,
    /// The player's own shared flag 5 (`Entity.isInvisible`) as the server
    /// last set it.
    pub(super) own_invisible: bool,
    /// The player's own shared flag 0 (`Entity.isOnFire`) as the server last
    /// set it: its fire (blaze fireballs, fire aspect, lightning) never
    /// reaches the window's own `fire_ticks` otherwise.
    own_on_fire: bool,
    /// `LivingEntity.DATA_ARROW_COUNT_ID` and `DATA_STINGER_COUNT_ID` of the
    /// own player, as the server synced them.
    pub(super) own_arrows: i32,
    pub(super) own_stingers: i32,
    /// `Player.DATA_SHOULDER_PARROT_LEFT` and `_RIGHT` of the own player.
    pub(super) own_shoulders: [Option<i32>; 2],
    /// The local player's profile UUID (always in the level).
    pub(super) profile: u128,
    /// The local player's profile name: its score holder and team member.
    pub(super) profile_name: String,
    /// `MultiPlayerGameMode.destroyBlockPos`, `.destroyProgress` and the face last sent: the block
    /// the break targets, which outlives the break itself.
    destroying: Option<([i32; 3], u8, f32)>,
    /// `MultiPlayerGameMode.destroyingItem`: the held item key the break began with.
    destroying_item: String,
    /// `MultiPlayerGameMode.isDestroying`: a break is running. `destroying` keeps the target after
    /// one ends (`destroyBlockPos` is written only when a break starts), so a dig resumed on the
    /// same block and item sends no new START and the server keeps its clock (#4124).
    is_destroying: bool,
    destroy_delay: u32,
    input: horizon_client::interaction_input::InteractionInput,
    /// Keys whose presses send a packet (swap hands, drop, a mount's
    /// inventory): like the click queue, they wait for the tick's
    /// `handleKeybinds` phase rather than going out between window events.
    pub(super) pending_keys: Vec<KeyCode>,
    /// Whether the main and off hand still wait for the server's answer to
    /// a SWAP_ITEM_WITH_OFFHAND (`remote_hand_slot_set`).
    swap_answer: [bool; 2],
    stop_mining: bool,
    placing: Option<([i32; 3], u64)>,
    /// `CommonPlayerSpawnInfo.isFlat`: the level is a superflat one.
    pub(super) flat: bool,
    death_message: Option<horizon_client::text::Text>,
    /// `LocalPlayer.showDeathScreen`: off with `immediate_respawn`.
    show_death_screen: bool,
    /// `Player.isReducedDebugInfo`: the server's `reduced_debug_info` rule.
    pub(super) reduced_debug_info: bool,
    /// `LocalPlayer.doLimitedCrafting`: whether limited crafting is enabled.
    pub(super) do_limited_crafting: bool,
    player_type: Option<Arc<MobType>>,
    /// Inventory, off hand and armour as last synchronised, for creative-mode
    /// slot updates (`creative_sync_stacks`).
    synced: Vec<Option<ItemStack>>,
    /// While a creative screen is open: the inventory menu's crafting grid, armour and off hand
    /// (slots 1 to 8 and 45) as last sent or received, which its clicks send when they change.
    creative_menu: Option<[Option<ItemStack>; 9]>,
    decode_errors: u32,
    /// Connection-owned `ClientClockManager` state, retained across dimensions.
    clocks: HashMap<i32, play::ClockState>,
    default_clock: Option<i32>,
    overworld_clock: Option<i32>,
    clock_game_time: i64,
    /// `ClientChunkCache.Storage`'s view centre (ClientboundSetChunkCacheCenterPacket).
    chunk_center: (i32, i32),
    /// Received events a frame left for the next one, in order, and when
    /// the oldest of them first waited (`poll_remote`).
    backlog: VecDeque<Received>,
    backlog_since: Option<Instant>,
    /// `ClientboundTransferPacket`'s target: when the connection ends the
    /// game connects here, the stored cookies carried over (`TransferState`).
    transfer: Option<String>,
    /// `ClientCommonPacketListenerImpl.serverLinks`: the links "Server
    /// Links..." in the pause menu lists (only `http`/`https` URIs kept).
    pub(super) server_links: Vec<horizon_protocol::packets::configuration::ServerLink>,
    /// `ClientPacketListener.getDebugQueryHandler`: the F3+I query waiting
    /// for the server's data.
    pub(super) debug_query: super::debug_keys::DebugQuery,
}

/// The time a frame spends on chunk packets before it leaves the rest of
/// the received events for the next frame. A batch of 25 to 60 columns
/// (0.1 ms each on an M4 Pro) all in one frame was a 3 to 8 ms frame every
/// time the player crossed a chunk border at 32 chunks; they decode on the
/// chunk decoder's thread now, and the frame only inserts them.
const CHUNK_PACKETS_PER_FRAME: Duration = Duration::from_millis(1);
/// Events that have waited this long are all handled in the next frame,
/// whatever inserting their columns costs, up to a column still decoding:
/// the packets after a slow machine's chunks don't wait on its frames.
const BACKLOG_LIMIT: Duration = Duration::from_millis(50);
/// What a frame spends on an overdue backlog's chunk packets on a phone: there one frame
/// taking the whole backlog after a teleport was a visible hitch (a 214 ms frame on the POCO
/// F6 Pro); four times a frame's usual share still drains it in a few frames.
const OVERDUE_CHUNK_PACKETS_PER_FRAME: Duration = Duration::from_millis(4);
/// A gap in the ticks or movement sent to the server worth a log line (an old server's
/// anticheat behind ViaVersion counts on one each tick, and a reminder every 20).
const GAP_LOGGED: Duration = Duration::from_millis(1100);

/// A movement packet in the log, when it came late or was the first. Apart from
/// `remote_send_state`: horizon-mixins refuses to hook a function that calls `log::info!`
/// (E_UNSUPPORTED_MACRO).
fn log_movement_gap(moved_at: &mut Option<Instant>, play_began: Option<Instant>) {
    let now = Instant::now();
    match moved_at.replace(now).map(|at| now - at) {
        Some(gap) if gap >= GAP_LOGGED => log::info!(
            "Horizon: {} ms between two movement packets",
            gap.as_millis()
        ),
        Some(_) => {}
        None => log::info!(
            "Horizon: the first movement packet {} ms after the play phase began",
            play_began.map_or(0, |at| (now - at).as_millis())
        ),
    }
}
/// How many movement packets the first setback's log line shows.
const TRAIL: usize = 12;
/// Free stick direction rests this long after a hit, a use or a drop: those go with the view's
/// rotation, and turning away at once after a hit looks like an aimbot to an anticheat.
const STEER_REST: Duration = Duration::from_secs(1);
/// The open menu name of a server's horse inventory, then its chest's
/// columns (the packet names no `MenuType`).
const MOUNT_MENU: &str = "horizon:mount";
/// ServerboundPlayerInputPacket's shift bit.
const SHIFT_INPUT: u8 = 1 << 5;

impl Remote {
    /// Count movement packets and ticks ended; a second with more than 20 of either is logged.
    fn count_rate(&mut self, moves: u32, ticks: u32) {
        let now = Instant::now();
        let (since, m, t) = &mut self.rate;
        let start = *since.get_or_insert(now);
        if now - start >= Duration::from_secs(1) {
            if *m > 21 || *t > 21 {
                log::info!(
                    "Horizon: {m} movement packets and {t} ticks ended in {} ms",
                    (now - start).as_millis()
                );
            }
            *since = Some(now);
            *m = 0;
            *t = 0;
        }
        *m += moves;
        *t += ticks;
    }
    /// A new ClientPacketListener after a reconfiguration: what the last one tracked is gone.
    fn reset_listener(&mut self) {
        self.joined = false;
        self.entity_id = 0;
        self.position_sender = Default::default();
        self.tick_end_at = None;
        self.moved_at = None;
        self.play_began = None;
        self.rate = (None, 0, 0);
        self.trail.clear();
        self.steer_yaw = None;
        self.last_input = 0;
        self.was_sprinting = false;
        self.carried.set(None);
        self.loaded_sent = false;
        self.predictions = Default::default();
        self.menu = ClientMenu::default();
        self.menu_title = None;
        self.state_id = 0;
        self.use_target = None;
        self.entities.clear();
        self.lerps.clear();
        self.stepped.clear();
        self.head_lerps.clear();
        self.pending_items.clear();
        self.flash_on_set_health = false;
        self.names.clear();
        self.players.clear();
        self.tab_players.clear();
        self.skin_textures.clear();
        self.skins.clear();
        self.listed.clear();
        self.level_players.clear();
        self.destroying = None;
        self.destroy_delay = 0;
        self.input = Default::default();
        self.stop_mining = false;
        self.placing = None;
        self.death_message = None;
        self.synced = vec![None; CREATIVE_SYNCED];
        self.clocks.clear();
        self.clock_game_time = 0;
        self.chunk_center = (0, 0);
    }
}

fn face_direction(face: [i32; 3]) -> u8 {
    match face {
        [0, -1, 0] => 0,
        [0, 1, 0] => 1,
        [0, 0, -1] => 2,
        [0, 0, 1] => 3,
        [-1, 0, 0] => 4,
        [1, 0, 0] => 5,
        _ => 1,
    }
}

/// A protocol number format as the scoreboard's.
fn number_format(f: &play::NumberFormat) -> horizon_client::scoreboard::NumberFormat {
    use horizon_client::scoreboard::{NumberFormat, tag_to_json};
    match f {
        play::NumberFormat::Blank => NumberFormat::Blank,
        play::NumberFormat::Styled(t) => NumberFormat::Styled(tag_to_json(t)),
        play::NumberFormat::Fixed(t) => NumberFormat::Fixed(tag_to_json(t)),
    }
}

impl Game<'_> {
    #[cfg(target_os = "android")]
    pub(super) fn touch_mining_progress(&self) -> f32 {
        self.remote
            .as_ref()
            .and_then(|r| r.destroying.map(|d| d.2))
            .or_else(|| self.mining.as_ref().map(|m| m.progress))
            .unwrap_or(0.)
    }

    /// A `Holder<SoundEvent>` off the wire: an inline event's location, or
    /// the name of a registered one.
    pub(super) fn remote_sound_name(
        &self,
        sound: &(i32, Option<(String, Option<f32>)>),
    ) -> Option<String> {
        match &sound.1 {
            Some((location, _)) => Some(location.clone()),
            None => self.remote.as_ref().and_then(|r| {
                r.builtin
                    .get("minecraft:sound_event")?
                    .name(sound.0)
                    .map(str::to_owned)
            }),
        }
    }
    pub(super) fn remote_sound_entity_state(
        &mut self,
        id: i32,
    ) -> Option<horizon_client::sound_engine::SoundEntityState> {
        let r = self.remote.as_ref()?;
        let player = self.player.as_ref().map(|p| {
            (
                r.entity_id,
                [p.position.x, p.position.y, p.position.z],
                r.local_entity_silent,
            )
        });
        // The local player is not in mobs, so avoid a guaranteed index miss.
        if player.is_some_and(|p| p.0 == id) {
            return entity_sound_state(id, player, None, &r.entities, None);
        }
        let mob = self.mob_index.get(&self.mobs, id as u32).map(|m| {
            horizon_client::sound_engine::SoundEntityState {
                position: [m.position.x, m.position.y, m.position.z],
                silent: m.silent,
                removed: m.removed,
            }
        });
        let dragon = self
            .dragon
            .as_ref()
            .map(|d| (d.id as i32, [d.position.x, d.position.y, d.position.z]));
        let mut state = entity_sound_state(id, player, mob, &r.entities, dragon)?;
        // Mob states already use the live position. In particular,
        // level_entity would find the dragon's stale Mob before its Dragon.
        if mob.is_some() || dragon.is_some_and(|d| d.0 == id) {
            return Some(state);
        }
        // Vehicles, projectiles and other simulated objects may have moved
        // since the last tracking packet.
        if let Some(position) = self
            .level_entity(id)
            .and_then(|target| self.target_position(target))
        {
            state.position = position;
        }
        Some(state)
    }
    /// The hurt sound the hurt player's own client plays from
    /// `ClientboundDamageEventPacket` (`LivingEntity.handleDamageEvent`):
    /// `Player.getHurtSound` is the damage type's effects, with the codec's
    /// `hurt` default when the type is unknown.
    pub(super) fn remote_hurt_sound(&self, source_type: i32) -> &'static str {
        match self.remote.as_ref() {
            Some(r) => hurt_sound_for(&r.registries, &self.health_resources.damage, source_type),
            None => horizon_client::health::hurt_sound("hurt"),
        }
    }
    /// Leave the title for a server (ConnectScreen): an empty world fed by the connection.
    pub(super) fn connect_server(&mut self, address: &str, name: &str) -> Result<()> {
        self.connect_server_with_cookies(address, name, None)
    }
    /// `connect_server` carrying a transfer's stored server cookies
    /// (`TransferState`); None on an ordinary join.
    pub(super) fn connect_server_with_cookies(
        &mut self,
        address: &str,
        name: &str,
        cookies: Option<horizon_client::net::Cookies>,
    ) -> Result<()> {
        self.connect_server_with_options(address, name, cookies, false, false)
    }
    /// `ConnectScreen.startConnecting`: keep LAN discovery's server kind through loading.
    pub(super) fn connect_server_with_options(
        &mut self,
        address: &str,
        name: &str,
        cookies: Option<horizon_client::net::Cookies>,
        lan: bool,
        realm: bool,
    ) -> Result<()> {
        super::world_loading::preload_server_loot(
            &self.options.resources,
            &self.res.core.item_data,
        );
        if self.defer_world_entry(super::PendingWorldEntry::Server {
            address: address.to_owned(),
            name: name.to_owned(),
            cookies: cookies.clone(),
            lan,
            realm,
        })? {
            return Ok(());
        }
        self.join_world_rest()?;
        self.client_logging_out();
        if let Err(error) = self.join_server(address, name, None, cookies, lan, realm) {
            log::warn!("Horizon: could not join server: {error}");
            let skip = self.settings.file.bool("skipMultiplayerWarning");
            match &mut self.title {
                Some(title) => {
                    title.skip_multiplayer_warning = skip;
                    title.failed(error);
                }
                None => {
                    let mut title = horizon_client::title::Title::new(
                        std::env::current_dir()?.join("saves"),
                        &self.options.resources,
                    )
                    .with_user(|| self.options.player_name());
                    title.skip_multiplayer_warning = skip;
                    title.failed(error);
                    self.show_title(title);
                }
            }
            self.capture(false);
        }
        Ok(())
    }
    pub(super) fn connect_remote_session(
        &mut self,
        address: &str,
        name: &str,
        local: Option<horizon_protocol::connection::Pipe>,
        cookies: Option<horizon_client::net::Cookies>,
        lan: bool,
        realm: bool,
    ) -> Result<()> {
        self.realm_session = realm;
        self.reset_tutorial();
        // Mabar skins: shared with the phones of a LAN, never of the internet.
        self.set_lan_peer(
            local.is_none() && horizon_client::lan::mabar_skins::is_private_address(address),
        );
        self.perf.world_opening();
        // A server's own biomes come over the connection; the last local
        // world's packed table must not name its ids. The window's own
        // server thread keeps the one it just set.
        if local.is_none() {
            self.world_biomes = None;
            self.world_biomes_base = None;
        }
        let root = self.options.resources.clone();
        let item_data = Arc::clone(&self.res.core.item_data);
        let builtin = builtin_registries(&root).ok_or("Resources are incomplete: registries.json is missing or invalid. Click \"Prepare again\" in the launcher (or run horizon-prepare), then try again.")?;
        let lang = Arc::new(horizon_client::net::load_language(&root));
        // Options.createInformation: what the server learns about this player,
        // sent again when one of those options changes.
        let info = self
            .settings
            .client_information(client_view_distance(self.settings.render_distance));
        if local.is_none() && !self.options.split_guest {
            mod_api::content::crash::set_detail(
                "Server",
                horizon_client::net::shown_address(address),
            );
        }
        // A signed-in account always plays under its own name on servers (an online-mode one
        // takes no other); single player's own server, the name its world opened with
        // (integrated.rs: a name chosen in Skin Customization goes first); a splitscreen player
        // signs in with its own account when it picked one.
        let auth = self.auth();
        let name = match (&auth, &local) {
            (Some(a), None) => a.name.as_str(),
            _ => name,
        };
        let user = auth.as_ref().map_or_else(
            || {
                (
                    name.to_owned(),
                    horizon_protocol::crypto::offline_uuid(name),
                )
            },
            |a| (a.name.clone(), a.uuid),
        );
        self.server_packs_connect(local.is_none().then_some(address), user);
        let session = match local {
            Some(pipe) => Session::connect_local(
                pipe,
                name,
                auth.clone(),
                info.clone(),
                lang.clone(),
                item_data.clone(),
            ),
            None => {
                self.extras_connecting(address, name, lan);
                match cookies {
                    Some(cookies) => Session::connect_transferred(
                        address,
                        name,
                        auth.clone(),
                        info.clone(),
                        lang.clone(),
                        cookies,
                        item_data.clone(),
                    ),
                    None => {
                        let connect = if lan {
                            Session::connect_lan
                        } else {
                            Session::connect
                        };
                        connect(
                            address,
                            name,
                            auth.clone(),
                            info.clone(),
                            lang.clone(),
                            item_data.clone(),
                        )
                    }
                }
            }
        };
        let player_type = super::perspective::player_type(&root);
        let mut streaming = Streaming::new(
            Arc::new(NoTerrain),
            self.worker_context(),
            self.res.threads(),
        );
        streaming.remote = true;
        streaming.radius = self.settings.render_distance;
        self.streaming = streaming;
        self.world = World::new(self.res.facts().air, 0);
        self.save = Arc::new(SaveDir::new(
            std::env::temp_dir().join("horizon-remote-unused"),
            horizon_client::save::LevelShape::OVERWORLD,
        ));
        self.inventory = vec![None; 36];
        self.offhand = None;
        self.armor = [None, None, None, None];
        self.craft_grid = vec![None; 9];
        self.craft_size = 2;
        self.remove_local_player_body();
        self.spawn = None;
        self.mobs.clear();
        self.clear_remote_objects();
        self.clear_remote_projectiles();
        self.creatures.projectiles.clear();
        self.dropped.clear();
        self.clear_client_particles();
        self.falling.clear();
        self.mining = None;
        self.container = None;
        self.carried = None;
        self.using = None;
        self.end_mod_use();
        self.clicks.miss_time = 0;
        // A new LocalPlayer has no effects: the server sends what it has
        // (ClientboundUpdateMobEffectPacket on placement).
        self.potions = Default::default();
        self.set_mode_id(0);
        self.experience = Experience {
            level: 0,
            progress: 0.0,
            total: 0,
        };
        // The connected client never opens a server world (single player also
        // joins here), so load the registries its argument parsers and local
        // suggestions need before receiving the server's command tree.
        self.ui.registries = Default::default();
        self.build_registries();
        self.ui.registries.permission = 0;
        self.ui.chat.clear();
        self.set_screen(Screen::Playing);
        self.hide_title();
        self.dim = Dim::Overworld;
        self.clear_lod();
        self.pending_uploads.clear();
        if let Some(r) = self.renderer.as_mut() {
            r.clear_sections(self.world.layout());
        }
        self.apply_settings();
        self.clear_waypoints();
        self.boss_overlay.reset();
        // The chat's tree until the server sends its own, or forever when it
        // sends none: the game's own, plus the commands native mods own.
        let chat_tree = self.chat_tree_for(None);
        // Minecraft.services() has the keys wherever the game is played, single player's own
        // server included: its guests' chat keys are checked against them.
        let services_keys = Arc::new(std::sync::Mutex::new(ServicesKeys::Pending));
        let slot = services_keys.clone();
        let _ = std::thread::Builder::new()
            .name("Services keys".into())
            .spawn(move || {
                use horizon_server_files::profiles;
                let keys = profiles::fetch_profile_key_keys(&profiles::discovery_url());
                *slot.lock().unwrap_or_else(|e| e.into_inner()) = match keys {
                    Ok(keys) if !keys.is_empty() => ServicesKeys::Ready(keys),
                    Ok(_) => ServicesKeys::Missing,
                    Err(why) => {
                        log::error!("Horizon: {why}");
                        ServicesKeys::Missing
                    }
                };
            });
        self.remote = Some(Remote {
            session,
            brand: "unknown".into(),
            recipe_transfer: false,
            features: ["minecraft:vanilla".to_string()].into(),
            quiet_commands: false,
            mod_net: false,
            logged_out: false,
            attachments: HashMap::new(),
            far_terrain: None,
            builtin,
            status: lang
                .get("connect.connecting")
                .cloned()
                .unwrap_or_else(|| "Connecting to the server...".into()),
            lang,
            registries: Arc::new(Dynamic::default()),
            tags: None,
            enchantments: None,
            joined: false,
            reconfiguring: None,
            logged_in: false,
            entity_id: 0,
            local_entity_silent: false,
            camera: None,
            use_item: None,
            dimension: DimensionType::vanilla("minecraft:overworld"),
            level: "minecraft:overworld".into(),
            level_key: mod_api::DimensionKey::OVERWORLD,
            level_facts: None,
            level_environment: None,
            facts_cache: Default::default(),
            hashed_seed: 0,
            chunk_format: Arc::new(ChunkFormat::new(
                &DimensionType::vanilla("minecraft:overworld"),
                Vec::new(),
                7,
            )),
            decoder: ChunkDecoder::new(),
            view_distance: self.settings.render_distance,
            simulation_distance: self.settings.simulation_distance,
            sequence: 0,
            last_teleport_sequence: -1,
            position_sender: Default::default(),
            tick_end_at: None,
            moved_at: None,
            play_began: None,
            rate: (None, 0, 0),
            trail: Default::default(),
            steer_yaw: None,
            interacted_at: Default::default(),
            told_rotation: Default::default(),
            held_back: Default::default(),
            stick_mode: Default::default(),
            stick_choice: None,
            stick_remembered: None,
            stick_setting: None,
            stick_setbacks: Default::default(),
            stick_used_at: None,
            stick_asked: false,
            swap_presses: 0,
            last_input: 0,
            was_sprinting: false,
            carried: std::cell::Cell::new(None),
            was_shared_flags: 0,
            was_living_flags: 0,
            last_recorded_pose: None,
            abilities: None,
            loaded_sent: false,
            waiting_ticks: 0,
            setbacks: 0,
            player_attributes: Vec::new(),
            heard_level: None,
            level_load: horizon_client::level_loading::LevelLoadTracker::new(0),
            sent_info: info,
            predictions: HashMap::new(),
            menu: ClientMenu::default(),
            ticked_before_player: std::collections::HashSet::new(),
            craft_result: None,
            menu_title: None,
            menu_title_snapshot: None,
            state_id: 0,
            use_target: None,
            menu_extra: None,
            entities: HashMap::new(),
            tick_fluids: Vec::new(),
            tick_rain: Vec::new(),
            tick_standings: Vec::new(),
            tick_water_fx: Vec::new(),
            passenger_vehicles: HashMap::new(),
            lerps: HashMap::new(),
            cart_lerps: HashMap::new(),
            stepped: HashMap::new(),
            head_lerps: HashMap::new(),
            ghast_dismount: None,
            first_passenger: false,
            passenger_index: 0,
            pending_items: HashMap::new(),
            health: (20.0, 20, 5.0),
            flash_on_set_health: false,
            game_mode: 0,
            chat: Vec::new(),
            chat_received: 0,
            names: HashMap::new(),
            chat_queue: VecDeque::new(),
            signatures: Default::default(),
            chat_previous: None,
            next_chat_index: 0,
            enforces_secure_chat: false,
            services_keys,
            chat_sessions: HashMap::new(),
            chat_validators: HashMap::new(),
            joining_narrated: None,
            chat_tree,
            chat_tree_received: false,
            secure_chat: Default::default(),
            custom_completions: Default::default(),
            players: HashMap::new(),
            tab_players: HashMap::new(),
            skin_textures: HashMap::new(),
            own_profile: OwnProfile::Idle,
            skins: HashMap::new(),
            skins_revision: 0,
            published_skins: None,
            listed: Default::default(),
            online_mode: false,
            levels: Vec::new(),
            mabar: None,
            level_players: HashMap::new(),
            level_player_names: HashMap::new(),
            level_player_textures: HashMap::new(),
            glowing: Default::default(),
            own_invisible: false,
            own_on_fire: false,
            own_arrows: 0,
            own_stingers: 0,
            own_shoulders: [None; 2],
            profile: 0,
            profile_name: String::new(),
            destroying: None,
            destroying_item: String::new(),
            is_destroying: false,
            destroy_delay: 0,
            input: Default::default(),
            pending_keys: Vec::new(),
            swap_answer: [false; 2],
            stop_mining: false,
            placing: None,
            flat: false,
            death_message: None,
            show_death_screen: true,
            reduced_debug_info: false,
            do_limited_crafting: false,
            player_type,
            synced: vec![None; CREATIVE_SYNCED],
            creative_menu: None,
            decode_errors: 0,
            clocks: HashMap::new(),
            default_clock: None,
            overworld_clock: None,
            clock_game_time: 0,
            chunk_center: (0, 0),
            backlog: VecDeque::new(),
            backlog_since: None,
            server_links: Vec::new(),
            debug_query: Default::default(),
            transfer: None,
        });
        // The level's TickRateManager follows the server's ticking packets.
        self.clock = horizon_client::clock::FixedStep::client();
        // The player's advancements and statistics are the server's.
        self.adv = Default::default();
        self.stats = Default::default();
        Ok(())
    }

    /// Prepare a client session and remote state to receive packets from a studio replay.
    pub(super) fn init_remote_for_replay(&mut self) -> Result<()> {
        self.realm_session = false;
        self.reset_tutorial();
        self.perf.world_opening();
        self.world_biomes = None;
        self.world_biomes_base = None;
        let root = self.options.resources.clone();
        let builtin = builtin_registries(&root).ok_or("Resources are incomplete")?;
        let lang = Arc::new(horizon_client::net::load_language(&root));
        let info = self
            .settings
            .client_information(client_view_distance(self.settings.render_distance));
        let auth = self.auth();
        let name = auth.as_ref().map_or("ReplayPlayer", |a| a.name.as_str());
        let user = auth.as_ref().map_or_else(
            || {
                (
                    name.to_owned(),
                    horizon_protocol::crypto::offline_uuid(name),
                )
            },
            |a| (a.name.clone(), a.uuid),
        );
        self.server_packs_connect(None, user);
        let session = Session::replay(name);
        let player_type = super::perspective::player_type(&root);
        let mut streaming = Streaming::new(
            Arc::new(NoTerrain),
            self.worker_context(),
            self.res.threads(),
        );
        streaming.remote = true;
        streaming.radius = self.settings.render_distance;
        self.streaming = streaming;
        self.world = World::new(self.res.facts().air, 0);
        self.save = Arc::new(SaveDir::new(
            std::env::temp_dir().join("horizon-replay-unused"),
            horizon_client::save::LevelShape::OVERWORLD,
        ));
        self.inventory = vec![None; 36];
        self.offhand = None;
        self.armor = [None, None, None, None];
        self.craft_grid = vec![None; 9];
        self.craft_size = 2;
        self.remove_local_player_body();
        self.spawn = None;
        self.mobs.clear();
        self.clear_remote_objects();
        self.clear_remote_projectiles();
        self.creatures.projectiles.clear();
        self.dropped.clear();
        self.particles.clear();
        self.pickups.clear();
        self.orbs.clear();
        self.falling.clear();
        self.mining = None;
        self.container = None;
        self.carried = None;
        self.using = None;
        self.end_mod_use();

        self.set_mode_id(0);
        self.experience = Experience {
            level: 0,
            progress: 0.0,
            total: 0,
        };
        self.ui.registries = Default::default();
        self.build_registries();
        self.ui.registries.permission = 0;
        self.ui.chat.clear();
        self.set_screen(Screen::Playing);
        self.title = None;
        self.dim = Dim::Overworld;
        self.clear_lod();
        self.pending_uploads.clear();
        if let Some(r) = self.renderer.as_mut() {
            r.clear_sections(self.world.layout());
        }
        self.apply_settings();
        self.clear_waypoints();
        self.boss_overlay.reset();

        let chat_tree = self.chat_tree_for(None);
        let services_keys = Arc::new(std::sync::Mutex::new(ServicesKeys::Ready(Vec::new())));

        self.remote = Some(Remote {
            names: HashMap::new(),
            tick_end_at: None,
            moved_at: None,
            play_began: None,
            rate: (None, 0, 0),
            trail: Default::default(),
            steer_yaw: None,
            interacted_at: Default::default(),
            told_rotation: Default::default(),
            held_back: Default::default(),
            stick_mode: Default::default(),
            stick_choice: None,
            stick_remembered: None,
            stick_setting: None,
            stick_setbacks: Default::default(),
            stick_used_at: None,
            stick_asked: false,
            swap_presses: 0,
            waiting_ticks: 0,
            setbacks: 0,
            player_attributes: Vec::new(),
            creative_menu: None,
            session,
            brand: "horizon-studio".into(),
            recipe_transfer: false,
            features: ["minecraft:vanilla".to_string()].into(),
            quiet_commands: false,
            mod_net: false,
            attachments: HashMap::new(),
            far_terrain: None,
            builtin,
            status: "Playing Studio Replay".into(),
            lang,
            registries: Arc::new(Dynamic::default()),
            tags: None,
            enchantments: None,
            joined: false,
            reconfiguring: None,
            logged_in: true,
            logged_out: false,
            local_entity_silent: false,
            entity_id: 0,
            camera: None,
            use_item: None,
            dimension: DimensionType::vanilla("minecraft:overworld"),
            level: "minecraft:overworld".into(),
            level_key: mod_api::DimensionKey::OVERWORLD,
            level_facts: None,
            level_environment: None,
            facts_cache: Default::default(),
            hashed_seed: 0,
            chunk_format: Arc::new(ChunkFormat::new(
                &DimensionType::vanilla("minecraft:overworld"),
                Vec::new(),
                7,
            )),
            decoder: ChunkDecoder::new(),
            view_distance: self.settings.render_distance,
            simulation_distance: self.settings.simulation_distance,
            sequence: 0,
            last_teleport_sequence: -1,
            position_sender: Default::default(),
            last_input: 0,
            was_sprinting: false,
            was_shared_flags: 0,
            was_living_flags: 0,
            last_recorded_pose: None,
            carried: std::cell::Cell::new(None),
            abilities: None,
            loaded_sent: false,
            heard_level: None,
            level_load: horizon_client::level_loading::LevelLoadTracker::new(0),
            sent_info: info,
            predictions: HashMap::new(),
            menu: ClientMenu::default(),
            ticked_before_player: std::collections::HashSet::new(),
            craft_result: None,
            menu_title: None,
            menu_title_snapshot: None,
            state_id: 0,
            use_target: None,
            menu_extra: None,
            entities: HashMap::new(),
            tick_fluids: Vec::new(),
            tick_rain: Vec::new(),
            tick_standings: Vec::new(),
            tick_water_fx: Vec::new(),
            passenger_vehicles: HashMap::new(),
            lerps: HashMap::new(),
            cart_lerps: HashMap::new(),
            stepped: HashMap::new(),
            head_lerps: HashMap::new(),
            ghast_dismount: None,
            first_passenger: false,
            passenger_index: 0,
            pending_items: HashMap::new(),
            health: (20.0, 20, 5.0),
            flash_on_set_health: false,
            game_mode: 0,
            chat: Vec::new(),
            chat_received: 0,
            chat_queue: VecDeque::new(),
            signatures: Default::default(),
            chat_previous: None,
            next_chat_index: 0,
            enforces_secure_chat: false,
            services_keys,
            chat_sessions: HashMap::new(),
            chat_validators: HashMap::new(),
            joining_narrated: None,
            chat_tree,
            chat_tree_received: false,
            secure_chat: Default::default(),
            custom_completions: Default::default(),
            players: HashMap::new(),
            tab_players: HashMap::new(),
            skin_textures: HashMap::new(),
            own_profile: OwnProfile::Idle,
            skins: HashMap::new(),
            skins_revision: 0,
            published_skins: None,
            listed: Default::default(),
            online_mode: false,
            levels: Vec::new(),
            mabar: None,
            level_players: HashMap::new(),
            level_player_names: HashMap::new(),
            level_player_textures: HashMap::new(),
            glowing: Default::default(),
            own_invisible: false,
            own_on_fire: false,
            own_arrows: 0,
            own_stingers: 0,
            own_shoulders: [None; 2],
            profile: 0,
            profile_name: String::new(),
            destroying: None,
            destroying_item: String::new(),
            is_destroying: false,
            destroy_delay: 0,
            input: Default::default(),
            pending_keys: Vec::new(),
            swap_answer: [false; 2],
            stop_mining: false,
            placing: None,
            flat: false,
            death_message: None,
            show_death_screen: true,
            reduced_debug_info: false,
            do_limited_crafting: false,
            player_type,
            synced: vec![None; CREATIVE_SYNCED],
            decode_errors: 0,
            clocks: HashMap::new(),
            default_clock: None,
            overworld_clock: None,
            clock_game_time: 0,
            chunk_center: (0, 0),
            backlog: VecDeque::new(),
            backlog_since: None,
            server_links: Vec::new(),
            debug_query: Default::default(),
            transfer: None,
        });
        self.clock = horizon_client::clock::FixedStep::client();
        self.adv = Default::default();
        self.stats = Default::default();
        Ok(())
    }

    /// Disconnect and return to the multiplayer screen (DisconnectedScreen).
    pub(super) fn leave_server(&mut self, reason: Option<String>) -> Result<()> {
        if reason.is_none() && self.integrated.as_ref().is_some_and(|s| !s.stopped()) {
            return self.begin_closing_save(false);
        }
        self.client_logging_out();
        self.finish_perf();
        if reason.is_none() {
            // Leaving on purpose: nothing for auto-reconnect to retry.
            self.extras_left();
        }
        self.clear_waypoints();
        self.hud_disconnected();
        // Vanilla's Hud also ticks on the title and connect screens, so the
        // action bar has run out by the next world; Horizon's ticks in one.
        self.titles.overlay_time = 0;
        // GameRenderer.resetData and narrator.clear (Minecraft.disconnect).
        self.item_activation.reset();
        // Kept mod block entity geometry belongs to the world leaving.
        self.kept_block_entities.get_mut().clear();
        self.mod_view_distances.take();
        self.clear_narrator();
        // updateLevelInEngines(null): every sound stops and the music manager
        // lets go of its track. Leaving a server comes straight here, not
        // through finish_close_world, so the biome's ambient loop would
        // otherwise play on over the title screen until the next world.
        self.ambience.clear(&mut self.audio);
        self.music.stop_playing(None, &mut self.audio);
        self.audio.stop_all();
        self.mod_sounds.clear();
        // A world is leaving either way: its icon attempt goes with it.
        self.world_icon = None;
        // Minecraft.disconnect: the client level goes.
        if self.remote.is_some() {
            self.forget_all_remote_columns();
        }
        if let Some(old) = self.remote.as_mut().and_then(|r| r.heard_level.take()) {
            self.native_client_level_unload(&old);
        }
        if self.remote.is_some() {
            self.native_client_disconnecting();
        }
        if let Some(mut r) = self.remote.take() {
            r.session.disconnect();
        }
        super::lan::forget_sync_data();
        self.gui.atlas.clear_object_skins();
        self.server_packs_disconnect();
        self.code_of_conduct = None;
        self.reset_tutorial();
        self.close_dialogs();
        // Gui.setScreen -> KeyMapping.releaseAll: the title or disconnect
        // screen releases every held key. Its release goes to that screen,
        // so a key kept here would still be down in the next world (#3246).
        self.keys.clear();
        self.left = false;
        self.right = false;
        // The shared world is closing: its invite code ends with it.
        self.friends_lan_closed();
        // Single player's server thread saves the world before the title shows.
        let single_player = match self.integrated.take() {
            Some(server) => {
                server.stop()?;
                true
            }
            None => false,
        };
        // A splitscreen player leaving says nothing of where P1 plays.
        if !self.options.split_guest {
            mod_api::content::crash::clear_detail("Server");
        }
        self.clock = Default::default();
        self.clear_remote_world();
        let saves = std::env::current_dir()?.join("saves");
        let mut title = horizon_client::title::Title::new(saves.clone(), &self.options.resources)
            .with_user(|| self.options.player_name());
        title.last_server = self.settings.file.string("lastServer");
        title.skip_multiplayer_warning = self.settings.file.bool("skipMultiplayerWarning");
        if single_player && reason.is_none() {
            self.show_title(title);
            self.capture(false);
            return Ok(());
        }
        title.mode = horizon_client::title::Mode::Multiplayer(match reason {
            Some(reason) => horizon_client::multiplayer::Screen::Disconnected {
                reason,
                retry: None,
                lines: std::cell::Cell::new(1),
            },
            None => horizon_client::multiplayer::Screen::list(&saves),
        });
        self.show_title(title);
        self.capture(false);
        Ok(())
    }

    /// What the connection's level and listener held (leaving, or reconfiguring).
    fn clear_remote_world(&mut self) {
        self.adv = Default::default();
        self.stats = Default::default();
        self.score.board = Default::default();
        self.streaming = Streaming::new(Arc::new(NoTerrain), self.worker_context(), 1);
        self.streaming.paused = true;
        self.reset_client_world(0);
        self.remove_local_player_body();
        // The vehicle went with the level: an id kept here would seat the
        // next login on an entity that is not there.
        self.riding = None;
        // Out of the server's bed: InBedChatScreen goes with the level.
        self.sleeping = None;
        self.sleep_timer = 0;
        self.clear_entities();
        self.score.board = Default::default();
        self.tab_list = Default::default();
        self.border = Default::default();
        self.set_screen(Screen::Playing);
        self.container = None;
        self.clear_lod();
        self.pending_uploads.clear();
        if let Some(r) = self.renderer.as_mut() {
            r.clear_sections(self.world.layout());
        }
    }
    /// `Hud.onDisconnected`, from `Minecraft.disconnect` and `clearClientLevel`
    /// (a proxy's reconfiguration too), but the chat: joining clears it and a
    /// reconfiguration keeps it (`ChatComponent.storeState`).
    fn hud_disconnected(&mut self) {
        self.tab_list.reset();
        self.boss_overlay.reset();
        self.adv.toasts.clear();
        self.debug_charts.reset();
        self.titles.clear();
        self.titles.reset_times();
    }

    pub(super) fn send(&self, packet: S) {
        if let Some(r) = &self.remote {
            let interaction = is_interaction(&packet);
            // MultiPlayerGameMode's attack, interact, useItem, useItemOn, releaseUsingItem, dropItem and
            // piercingAttack run ensureHasSentCarriedItem before their packet: a slot the number keys
            // chose this tick goes out first. A miss, a dig's start or a swap waits for the next tick.
            if let S::Attack(_) | S::Interact(_) | S::UseItem(_) | S::UseItemOn(_) = &packet {
                self.remote_send_carried();
            } else if let S::PlayerAction(a) = &packet
                && matches!(a.action, 4 | 5 | 6 | 8)
            {
                self.remote_send_carried();
            }
            if (interaction || matches!(packet, S::Punch(_))) && r.stick_mode == ServerStick::Free {
                // A hit, a use, a block placed or an item dropped goes where the player looks:
                // a server places stairs, throws and aims by the rotation it was last sent. While
                // that is the stick's steering (last tick's move), it waits for the next tick,
                // after this tick's move with the view's look (`stick_steer` rests after it): a
                // look of its own would be a tick more to an anticheat (Grim's TickTimer), the
                // move after it the same look twice (AimDuplicateLook). The swings after it wait
                // too; a block being broken does not (`waits_for_the_view`).
                if interaction {
                    r.interacted_at.set(Some(Instant::now()));
                }
                let mut held = r.held_back.borrow_mut();
                if waits_for_the_view(&packet, r.steer_yaw.is_some(), !held.is_empty()) {
                    held.push(packet);
                    return;
                }
            }
            self.send_play(packet);
        }
    }
    /// `Connection.send` for the game's play packets: a mod may cancel one
    /// (`client_packet_send`), and the server never hears of it.
    fn send_play(&self, packet: S) {
        if let Some(r) = &self.remote
            && !self.client_packet_send_canceled(&packet)
        {
            r.session.send(packet);
        }
    }
    /// The hits and uses held back while the stick steered (`send`), the tick after.
    fn send_held_back(&self) {
        if let Some(r) = &self.remote {
            for packet in r.held_back.take() {
                self.send_play(packet);
            }
        }
    }
    /// The stick's mode on the server this tick, when on one.
    pub(super) fn remote_stick_mode(&self) -> Option<ServerStick> {
        self.remote.as_ref().map(|r| r.stick_mode)
    }
    /// The stick's mode on this connection: Analog in single player and for a LAN host.
    fn update_stick_mode(&mut self) -> ServerStick {
        let setting = self.settings.controller.server_stick;
        let dir = Self::options_path()
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        let Some(r) = self.remote.as_mut() else {
            return ServerStick::Analog;
        };
        if r.session.address == "local" {
            r.stick_mode = ServerStick::Analog;
            return r.stick_mode;
        }
        let mut remembered = *r
            .stick_remembered
            .get_or_insert_with(|| horizon_client::stick_memory::get(&dir, &r.session.address));
        // The setting changed while playing here: it is this server's mode now, the one
        // remembered for it included.
        if r.stick_setting
            .replace(setting)
            .is_some_and(|seen| seen != setting)
        {
            r.stick_choice = Some(setting);
            if remembered.is_some() {
                horizon_client::stick_memory::set(&dir, &r.session.address, setting);
                remembered = Some(setting);
                r.stick_remembered = Some(remembered);
            }
            log::info!(
                "Horizon: the stick moves as {} on this server, from the setting",
                setting.name()
            );
        }
        r.stick_mode = r.stick_choice.or(remembered).unwrap_or(setting);
        r.stick_mode
    }
    /// A setback while the stick moved the player: four in six seconds, in a mode the server
    /// may refuse, and the player is asked once a session how the stick should move here.
    fn stick_setback(&mut self) {
        let now = Instant::now();
        let ask = self.settings.controller.stick_prompt;
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        if !r
            .stick_used_at
            .is_some_and(|at| now - at < Duration::from_secs(1))
        {
            return;
        }
        r.stick_setbacks.push_back(now);
        while r
            .stick_setbacks
            .front()
            .is_some_and(|at| now - *at > Duration::from_secs(6))
        {
            r.stick_setbacks.pop_front();
        }
        if r.stick_setbacks.len() < 4
            || r.stick_asked
            || !ask
            || r.stick_mode == ServerStick::Keys
            || r.stick_choice.is_some()
            || r.stick_remembered.flatten().is_some()
        {
            return;
        }
        r.stick_asked = true;
        log::info!("Horizon: the server keeps pulling the stick's moves back; asking the player");
        self.ask_stick_mode();
    }
    /// The question: which mode, remembered for this server or not, asked again or not.
    fn ask_stick_mode(&mut self) {
        let t = |key: &str| Tag::String(horizon_client::lang::get_or(key, key).as_str().into());
        let choice = |key: &str, mode: ServerStick| {
            Tag::compound([
                ("label", t(key)),
                ("width", Tag::Int(300)),
                (
                    "action",
                    Tag::compound([
                        ("type", Tag::String("minecraft:dynamic/custom".into())),
                        ("id", Tag::String("starling:server_stick".into())),
                        (
                            "additions",
                            Tag::compound([("mode", Tag::String(mode.name().into()))]),
                        ),
                    ]),
                ),
            ])
        };
        let toggle = |key: &str, label: &str, initial: bool| {
            Tag::compound([
                ("type", Tag::String("minecraft:boolean".into())),
                ("key", Tag::String(key.into())),
                ("label", t(label)),
                ("initial", Tag::Byte(i8::from(initial))),
            ])
        };
        let dialog = Tag::compound([
            ("type", Tag::String("minecraft:multi_action".into())),
            ("title", t("starling.stick_prompt.title")),
            ("pause", Tag::Byte(0)),
            ("columns", Tag::Int(1)),
            (
                "body",
                Tag::compound([
                    ("type", Tag::String("minecraft:plain_message".into())),
                    ("contents", t("starling.stick_prompt.body")),
                    ("width", Tag::Int(300)),
                ]),
            ),
            (
                "inputs",
                Tag::List(vec![
                    toggle("remember", "starling.stick_prompt.remember", true),
                    toggle("dont_ask", "starling.stick_prompt.dont_ask", false),
                ]),
            ),
            (
                "actions",
                Tag::List(vec![
                    choice("starling.stick_prompt.free", ServerStick::Free),
                    choice("starling.stick_prompt.keys", ServerStick::Keys),
                    choice("starling.stick_prompt.analog", ServerStick::Analog),
                ]),
            ),
        ]);
        self.show_local_dialog(&dialog);
    }
    /// A click in one of the game's own dialogs (`starling:` actions).
    pub(super) fn starling_dialog_click(&mut self, id: &str, payload: Option<&Tag>) {
        // The link confirmation's Open (`confirm_link`).
        if id == "starling:open_link" {
            if let Some(Tag::String(url)) = payload.and_then(|p| p.get("url")) {
                self.open_link(&url.to_string_lossy());
            }
            return;
        }
        if id != "starling:server_stick" {
            log::warn!("Horizon: no such dialog action {id}");
            return;
        }
        let field = |key: &str| payload.and_then(|p| p.get(key));
        let mode = match field("mode") {
            Some(Tag::String(s)) => ServerStick::parse(&s.to_string_lossy()),
            _ => return,
        };
        let on = |key: &str| matches!(field(key), Some(Tag::Byte(b)) if *b != 0);
        let path = Self::options_path();
        let dir = path
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        if let Some(r) = self.remote.as_mut() {
            r.stick_choice = Some(mode);
            if on("remember") {
                horizon_client::stick_memory::set(&dir, &r.session.address, mode);
                r.stick_remembered = Some(Some(mode));
            }
        }
        if on("dont_ask") {
            self.settings.controller.stick_prompt = false;
            self.settings.save(&path);
        }
        log::info!(
            "Horizon: the stick moves as {} on this server{}",
            mode.name(),
            if on("remember") { ", remembered" } else { "" }
        );
    }
    /// Free stick direction (the controller setting, experimental): on a server the stick
    /// moves where it points by turning the rotation the server sees toward it and holding
    /// forward, a move the keys make. Not with keys held, nor while mining, using, riding or
    /// gliding, nor right after a hit or a use, which go with the view's rotation.
    fn stick_steer(&self, stick: Option<[f32; 2]>) -> Option<f32> {
        let r = self.remote.as_ref()?;
        let [left, forward] = stick?;
        let keys = [Action::Forward, Action::Left, Action::Back, Action::Right];
        let p = self.player.as_ref()?;
        if left.hypot(forward) <= 0.3
            || keys.iter().any(|k| self.keys.held(*k))
            || self.mining.is_some()
            || self.using.is_some()
            || self.riding.is_some()
            || p.fall_flying
            || r.interacted_at
                .get()
                .is_some_and(|at| at.elapsed() < STEER_REST)
        {
            return None;
        }
        // Moving forward at yaw - a is moving at angle a (to the left) of yaw: Entity.moveRelative.
        Some(self.yaw - left.atan2(forward).to_degrees())
    }
    /// `LocalPlayer.isShiftKeyDown` (and so `isSecondaryUseActive`): the input as of the player's
    /// last tick (`input.keyPresses`, what the last ServerboundPlayerInputPacket said), not the key
    /// now: a sneak pressed in the frame of a click acts from the next tick, as on the server.
    fn remote_shift_down(&self) -> bool {
        self.remote
            .as_ref()
            .is_some_and(|r| r.last_input & SHIFT_INPUT != 0)
    }
    fn next_sequence(&mut self) -> i32 {
        let r = self.remote.as_mut().expect("remote");
        r.sequence += 1;
        r.sequence
    }
    fn item_name(&self, id: i32) -> Arc<str> {
        let r = self.remote.as_ref().expect("remote");
        Arc::from(
            r.builtin
                .get("minecraft:item")
                .and_then(|i| i.name(id))
                .unwrap_or("minecraft:air"),
        )
    }
    /// The server's network id of a mob effect (`brewing::EFFECTS` id).
    pub(super) fn remote_effect_network_id(&self, effect: &str) -> Option<i32> {
        let r = self.remote.as_ref()?;
        r.builtin
            .get("minecraft:mob_effect")?
            .id(&format!("minecraft:{effect}"))
    }
    /// A mob effect's network id as the effect (`brewing::EFFECTS` id).
    pub(super) fn remote_effect_id(&self, effect: i32) -> Option<&'static str> {
        self.remote_effect_def(effect).map(|def| def.id)
    }
    fn remote_effect_def(
        &self,
        effect: i32,
    ) -> Option<&'static horizon_client::brewing::EffectDef> {
        let r = self.remote.as_ref()?;
        let name = r.builtin.get("minecraft:mob_effect")?.name(effect)?;
        horizon_client::brewing::effect(name)
    }
    /// ClientPacketListener.handleUpdateMobEffect for the player itself
    /// (`LivingEntity.forceAddEffect`): the effect replaces any of its kind.
    /// Other entities' effects are not shown.
    fn remote_effect(&mut self, p: &play::UpdateMobEffect) {
        if self.remote.as_ref().is_none_or(|r| r.entity_id != p.id) {
            return;
        }
        let Some(def) = self.remote_effect_def(p.effect) else {
            return;
        };
        let id = def.id;
        let mut e = horizon_client::effects::EffectInstance::from_def(def, p.duration, p.amplifier);
        e.ambient = p.flags & 1 != 0;
        e.visible = p.flags & 2 != 0;
        e.show_icon = p.flags & 4 != 0;
        let list = &mut self.potions.effects.list;
        match list.iter_mut().find(|x| x.id == id) {
            // forceAddEffect's copyBlendState: an updated effect keeps its fade.
            Some(existing) => {
                e.client_blend = existing.client_blend;
                *existing = e;
            }
            None => {
                // ClientboundUpdateMobEffectPacket.shouldBlend (flag 8).
                let blend = p.flags & 8 != 0;
                if id == "darkness" {
                    self.sculk.darkness = super::sculk::darkness_blend_added(&e, blend);
                } else if id == "nausea" {
                    let (o_nausea, nausea) = screen_effects::nausea_blend_added(&e, blend);
                    self.screen_fx.o_nausea = o_nausea;
                    self.screen_fx.nausea = nausea;
                } else if !blend {
                    let has_effect = !e.ends_within(0);
                    e.client_blend = (has_effect, has_effect);
                }
                list.push(e);
            }
        }
    }
    fn item_id(&self, name: &str) -> i32 {
        let r = self.remote.as_ref().expect("remote");
        r.builtin
            .get("minecraft:item")
            .and_then(|i| i.id(name))
            .unwrap_or(0)
    }
    /// A stack from the server, decoded as the saved and command item NBT
    /// are, so tooltips and models see every field.
    pub(super) fn to_stack(&self, s: &Option<Stack>) -> Option<ItemStack> {
        let r = self.remote.as_ref()?;
        let stack = s.as_ref()?;
        if !horizon_protocol::item::custom::permitted(stack, r.registries.custom_components)
            .unwrap_or(false)
        {
            log::warn!("Dropping a stack with unnegotiated custom components");
            return None;
        }
        Some(horizon_client::item_sync::from_network(
            stack,
            &r.builtin,
            &r.registries,
        ))
    }
    /// A stack to the server (`SetCreativeModeSlot`): the same component
    /// patch a Horizon server sends its clients, with the registries' ids, so
    /// the server's echo doesn't hand back a bare item.
    fn to_network(&self, s: &Option<ItemStack>) -> Option<Stack> {
        let r = self.remote.as_ref()?;
        // A stack as the server sent it, only its count changed: its own
        // components go back, as a vanilla client keeps them.
        if let Some((stack, patch)) = s.as_ref().and_then(|s| Some((s, s.wire_patch()?))) {
            let mut out = Stack::new(self.item_id(&stack.item), stack.count as i32);
            let safe = r.registries.custom_components
                || horizon_protocol::item::custom::entries().next().is_none()
                || horizon_protocol::item::WirePatch::decode(&patch)
                    .and_then(|(added, removed)| {
                        let decoded = Stack {
                            added,
                            removed,
                            ..Stack::new(out.item, out.count)
                        };
                        horizon_protocol::item::custom::permitted(&decoded, false)
                    })
                    .unwrap_or(false);
            if safe {
                out.wire = horizon_protocol::item::WirePatch(Some(patch));
                return Some(out);
            }
            log::warn!("Cannot send cached custom components to this peer");
        }
        horizon_client::lan::guest::stack_to_network(
            |name| self.item_id(name),
            // Built-in first, as a hub's `SyncData::id` does: potions are one.
            |registry, name| {
                r.builtin
                    .get(registry)
                    .and_then(|ids| ids.id(name))
                    .or_else(|| r.registries.id(registry, name))
                    .or_else(|| {
                        (r.registries.custom_components
                            && registry == "minecraft:data_component_type")
                            .then(|| horizon_protocol::item::custom::id(name))
                            .flatten()
                    })
            },
            s,
        )
    }
    /// `HashedStack.create` with the connection's registries (the client's
    /// `decoratedHashOpsGenerator`).
    fn hashed_stack(&self, s: &Option<ItemStack>) -> Option<Hashed> {
        let r = self.remote.as_ref()?;
        let names = SyncedNames {
            builtin: &r.builtin,
            dynamic: &r.registries,
        };
        self.to_network(s).map(|s| Hashed::create(&s, &names))
    }
    /// A line in the connection's chat, with its styles, click and hover
    /// events (ChatComponent.addMessage). The lines the Chat option hides
    /// are dropped there (its `visibleMessageFilter`).
    pub(super) fn add_chat_line(
        &mut self,
        content: horizon_client::text::Text,
        source: horizon_client::chat::ChatSource,
    ) -> bool {
        self.signed_chat_line(content, source, None, ChatTrust::Secure)
    }
    /// `chat_line` of a player message, keeping its signature for a
    /// deletion (`GuiMessage.signature`).
    fn signed_chat_line(
        &mut self,
        content: horizon_client::text::Text,
        source: horizon_client::chat::ChatSource,
        signature: Option<Signature>,
        trust: ChatTrust,
    ) -> bool {
        if !self.ui.chat.shows(source) {
            return false;
        }
        let plain = content.plain();
        let focused = self.screen == Screen::Chat;
        if source == horizon_client::chat::ChatSource::Player {
            self.ui.chat.add_signed(
                &self.gui.atlas,
                content,
                signature,
                trust.tag(),
                self.visual_ticks(),
                focused,
            );
        } else {
            // addServerSystemMessage and addClientSystemMessage both carry
            // GuiMessageTag.systemSinglePlayer: the grey bar at the line's left.
            let indicator = Some(horizon_client::chat::SYSTEM_INDICATOR);
            self.ui.chat.add(
                &self.gui.atlas,
                content,
                indicator,
                source,
                self.visual_ticks(),
                focused,
            );
        }
        if let Some(r) = self.remote.as_mut() {
            r.chat.push((plain, Instant::now()));
            r.chat_received += 1;
            if r.chat.len() > 100 {
                r.chat.remove(0);
            }
        }
        true
    }

    /// `ChatListener.handleMessage` for player and disguised chat: held in the
    /// queue while Chat Delay is set, shown at once otherwise. `narration` is
    /// the chat type's narration of it (`decorateNarration`).
    fn handle_player_chat(&mut self, chat: QueuedChat) {
        if self.chat_delay_ms() > 0 {
            if let Some(r) = self.remote.as_mut() {
                r.chat_queue.push_back(chat);
            }
        } else {
            self.accept_chat(chat);
        }
    }

    /// `ChatListener.handleChatMessageError`: a message that fails validation shows as the red,
    /// italic `chat.validation_error` in its chat type (queued under Chat Delay as any
    /// message). Its signature counts as processed but not shown, for the acknowledgement the
    /// client sends the server. Vanilla also
    /// skips a blocked sender and one restricted to friends (not kept here either) and
    /// narrates the line.
    fn handle_chat_message_error(
        &mut self,
        chat_type: &play::ChatTypeBound,
        signature: Option<&Signature>,
    ) {
        if let Some(signature) = signature {
            self.mark_message_processed(signature, false);
        }
        let error = serde_json::json!({
            "translate": "chat.validation_error",
            "color": "red",
            "italic": true,
        });
        let line = self.decorate_chat(chat_type, &error, Decoration::Chat);
        // saySystemChatQueued(boundChatType.decorateNarration(CHAT_VALIDATION_ERROR)).
        let narration = self
            .decorate_chat(chat_type, &error, Decoration::Narration)
            .plain();
        self.handle_player_chat(QueuedChat {
            line,
            signature: None,
            trust: ChatTrust::Error,
            narration,
            sender: 0,
            chat_type: chat_type.chat_type,
        });
    }

    /// `ChatTrustLevel.evaluate` of a player message, with
    /// `ChatListener.evaluateTrustLevel`: the player's own message on their
    /// own server is secure. `decorated` is the line shown, `unsigned` the
    /// server's replacement for the signed content.
    fn chat_trust(
        &self,
        sender: u128,
        signed: bool,
        timestamp: i64,
        content: &str,
        unsigned: Option<&Tag>,
        decorated: &horizon_client::text::Text,
    ) -> ChatTrust {
        use horizon_client::scoreboard::{comp_text, tag_to_json};
        if self.integrated.is_some() && self.remote.as_ref().is_some_and(|r| r.profile == sender) {
            return ChatTrust::Secure;
        }
        // PlayerChatMessage.hasExpiredClient: MESSAGE_EXPIRES_AFTER_CLIENT is
        // the server's five minutes and two more.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as i64);
        let expired = now
            > timestamp.saturating_add(
                horizon_protocol::secure_chat::EXPIRES_AFTER_SERVER + 2 * 60 * 1000,
            );
        if !signed || expired {
            return ChatTrust::NotSecure;
        }
        // ChatTrustLevel.isModified: the signed content is not in the line, or
        // the replacement uses a font other than the default.
        let font_changed = unsigned.is_some_and(|tag| {
            comp_text(&tag_to_json(tag)).spans.iter().any(|span| {
                span.style
                    .font
                    .as_deref()
                    .is_some_and(|font| font != "minecraft:default")
            })
        });
        if !decorated.plain().contains(content) || font_changed {
            ChatTrust::Modified(content.to_owned())
        } else {
            ChatTrust::Secure
        }
    }

    /// `ClientPacketListener.handleDeleteChat`: the packed signature names a
    /// message; one still held back by Chat Delay never shows
    /// (`ChatListener.removeFromDelayedMessageQueue`), a shown one is
    /// replaced (`ChatComponent.deleteMessage`). A signature the cache
    /// doesn't hold is an invalid packet.
    fn handle_delete_chat(&mut self, packed: &(i32, Option<Vec<u8>>)) -> Result<()> {
        let Some(r) = self.remote.as_mut() else {
            return Ok(());
        };
        let Some(signature) = r.signatures.unpack(packed) else {
            let reason =
                horizon_client::text::Text::translate("multiplayer.disconnect.invalid_packet", &[]);
            return self.leave_server(Some(reason.plain()));
        };
        r.secure_chat.ignore_pending(&signature);
        let queued = r.chat_queue.len();
        r.chat_queue
            .retain(|chat| chat.signature.as_ref() != Some(&signature));
        if r.chat_queue.len() == queued {
            let ticks = self.visual_ticks();
            self.ui
                .chat
                .delete_message(&self.gui.atlas, &signature, ticks);
        }
        Ok(())
    }

    /// A message's `accept`: shown, it is narrated (`narrateChatMessage`) and
    /// starts the Chat Delay wait again.
    fn accept_chat(&mut self, chat: QueuedChat) -> bool {
        let signature = chat.signature;
        let shown = self.player_chat_line(chat);
        if let Some(signature) = &signature {
            self.mark_message_processed(signature, shown);
        }
        if shown && let Some(r) = self.remote.as_mut() {
            r.chat_previous = Some(Instant::now());
        }
        shown
    }

    /// `showMessageToPlayer`: the line joins the chat if Only Show Secure
    /// Chat, mods and the Chat option let it through, and the narrator says
    /// it (`sayChatQueued`).
    fn player_chat_line(&mut self, chat: QueuedChat) -> bool {
        let QueuedChat {
            mut line,
            signature,
            trust,
            narration,
            sender,
            chat_type,
        } = chat;
        // ChatListener.showMessageToPlayer: Only Show Secure Chat drops what is not secure.
        if matches!(trust, ChatTrust::NotSecure) && self.settings.file.bool("onlyShowSecureChat") {
            return false;
        }
        // handleChatMessageError says its line as system chat, and mods are not told of it.
        let error = matches!(trust, ChatTrust::Error);
        if !error && !self.chat_received(false, false, sender, Some(chat_type), &mut line) {
            return false;
        }
        let source = horizon_client::chat::ChatSource::Player;
        let shown = self.signed_chat_line(line, source, signature, trust);
        if shown && error {
            self.say_system_chat_queued(&horizon_client::text::Text::untrusted(&narration));
        } else if shown {
            self.say_chat_queued(&narration);
        }
        shown
    }

    /// NeoForge's `ClientChatReceivedEvent` (`ClientHooks.onClientChat`,
    /// `onClientPlayerChat` and `onClientSystemChat`): mods may replace
    /// `line` or cancel it. False when cancelled.
    fn chat_received(
        &mut self,
        system: bool,
        overlay: bool,
        sender: u128,
        chat_type: Option<i32>,
        line: &mut horizon_client::text::Text,
    ) -> bool {
        if !mod_api::__private::listened(mod_api::HookId::ClientChatReceived) {
            return true;
        }
        let registries = self.remote.as_ref().map(|r| Arc::clone(&r.registries));
        let name = chat_type
            .zip(registries.as_deref())
            .and_then(|(id, registries)| registries.name("minecraft:chat_type", id));
        let mut cancel = false;
        self.native_client_chat_received(system, overlay, sender, name, line, &mut cancel);
        !cancel
    }

    /// `ChatListener.messageDelay`: Chat Delay in milliseconds, as Java
    /// truncates it.
    fn chat_delay_ms(&self) -> i64 {
        (self.settings.chat_delay * 1000.0) as i64
    }

    /// `ChatListener.tick`: while paused the wait does not run down; with no
    /// Chat Delay the queue flushes; otherwise the next message the chat takes
    /// shows once the delay has passed since the last shown one.
    pub(super) fn tick_chat_queue(&mut self) {
        let delay = self.chat_delay_ms();
        // Minecraft.isPaused: a single-player pausing screen, not every pause menu.
        if self.pause {
            if delay > 0
                && let Some(r) = self.remote.as_mut()
            {
                r.chat_previous = r.chat_previous.map(|t| t + Duration::from_millis(50));
            }
            return;
        }
        if delay == 0 {
            if self
                .remote
                .as_ref()
                .is_some_and(|r| !r.chat_queue.is_empty())
            {
                self.flush_chat_queue();
            }
            return;
        }
        let ready = self.remote.as_ref().is_some_and(|r| {
            r.chat_previous
                .is_none_or(|t| t.elapsed() >= Duration::from_millis(delay as u64))
        });
        if !ready {
            return;
        }
        // A message the Chat option hides is dropped and the next one is tried,
        // until one shows.
        while let Some(chat) = self.remote.as_mut().and_then(|r| r.chat_queue.pop_front()) {
            if self.accept_chat(chat) {
                break;
            }
        }
    }

    /// `ChatListener.flushQueue`: every held-back message shows now, and the
    /// delay starts over.
    fn flush_chat_queue(&mut self) {
        let Some(queue) = self
            .remote
            .as_mut()
            .map(|r| std::mem::take(&mut r.chat_queue))
        else {
            return;
        };
        for chat in queue {
            self.accept_chat(chat);
        }
        if let Some(r) = self.remote.as_mut() {
            r.chat_previous = None;
        }
    }

    /// The minecarts the server sent, for the control socket's `status`
    /// (their velocity is not tracked here).
    pub(super) fn remote_carts(&self) -> Option<Vec<serde_json::Value>> {
        let r = self.remote.as_ref()?;
        let mut carts: Vec<&Tracked> = r
            .entities
            .values()
            .filter(|t| t.kind.ends_with("minecart"))
            .collect();
        carts.sort_by(|a, b| {
            a.position
                .partial_cmp(&b.position)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(
            carts
                .into_iter()
                .map(|t| serde_json::json!({"kind": t.kind, "position": t.position, "velocity": [0.0, 0.0, 0.0]}))
                .collect(),
        )
    }

    /// This player's account: a splitscreen player's own, else the game's (P1's).
    pub(super) fn auth(&self) -> Option<std::sync::Arc<horizon_client::auth::AuthSession>> {
        match self.options.split_guest {
            true => self.split_auth.clone(),
            false => horizon_client::auth::current(),
        }
    }
    /// Connected to a server in online mode (its Login): each player needs its own account.
    pub(super) fn online_server(&self) -> bool {
        self.remote.as_ref().is_some_and(|r| r.online_mode)
    }

    /// `ClientSuggestionProvider.customCompletionSuggestions` of the server
    /// connection (none without one).
    pub(super) fn remote_custom_completions(&self) -> std::collections::BTreeSet<String> {
        self.remote
            .as_ref()
            .map(|r| r.custom_completions.clone())
            .unwrap_or_default()
    }
    /// The names in the server's player list, sorted.
    pub(super) fn remote_online_names(&self) -> Option<Vec<String>> {
        let r = self.remote.as_ref()?;
        let mut names: Vec<String> = r.players.values().cloned().collect();
        names.sort();
        Some(names)
    }
    /// The open menu's window id (0: the inventory).
    pub(super) fn remote_menu_id(&self) -> i32 {
        self.remote
            .as_ref()
            .and_then(|r| r.menu.as_ref())
            .map_or(0, |m| m.0)
    }
    /// The server's last `n` chat lines, newest first.
    pub(super) fn remote_chat(&self, n: usize) -> Option<Vec<String>> {
        let r = self.remote.as_ref()?;
        Some(r.chat.iter().rev().take(n).map(|l| l.0.clone()).collect())
    }

    /// Received events left for the next frame (`poll_remote`): until they
    /// are handled, the terrain the server sent is not all there.
    pub(super) fn remote_backlog(&self) -> usize {
        self.remote.as_ref().map_or(0, |r| r.backlog.len())
    }
    /// A headless command oracle has a real connection but no movement tick.
    /// Acknowledge loaded terrain without predicting or sending movement.
    pub(super) fn poll_stationary_client(&mut self) -> Result<()> {
        self.poll_remote()?;
        let loaded = self.player.as_ref().is_some_and(|p| {
            self.streaming.is_lit(
                (p.position.x.floor() as i32) >> 4,
                (p.position.z.floor() as i32) >> 4,
            )
        });
        if loaded && self.remote.as_ref().is_some_and(|r| !r.loaded_sent) {
            self.send(S::PlayerLoaded(Empty));
            let r = self.remote.as_mut().unwrap();
            r.loaded_sent = true;
            r.level_load.finish();
        }
        Ok(())
    }

    /// Process what the network thread received (once per frame, like the
    /// client's packet processor), in order. Chunk packets get
    /// `CHUNK_PACKETS_PER_FRAME`: once a frame has spent it, the events from
    /// the next chunk packet on wait for the next frame, unless they have
    /// already waited `BACKLOG_LIMIT` (then all of them go, or on a phone
    /// `OVERDUE_CHUNK_PACKETS_PER_FRAME` of them). A chunk packet still
    /// decoding holds back what follows it.
    pub(super) fn poll_remote(&mut self) -> Result<()> {
        let polled = self.poll_remote_events();
        // The block events and block changes of a piston clock start and end
        // thousands of moves at once: the collision view and the renderer
        // take them all in one sync, before the tick or frame reads them.
        if std::mem::take(&mut self.moving_pistons_stale) {
            self.sync_moving_pistons();
        }
        polled
    }
    fn poll_remote_events(&mut self) -> Result<()> {
        let Some(r) = self.remote.as_mut() else {
            return Ok(());
        };
        let mut events = std::mem::take(&mut r.backlog);
        // Chunk packets start decoding now; the frame that reaches one in
        // order inserts its column.
        for event in r.session.poll() {
            events.push_back(match event {
                Event::Packet(p) if matches!(*p, C::LevelChunkWithLight(_)) => {
                    match r.decoder.submit(r.chunk_format.clone(), p) {
                        None => Received::Chunk(r.chunk_format.clone()),
                        Some(p) => Received::Event(Event::Packet(p)),
                    }
                }
                event => Received::Event(event),
            });
        }
        let overdue = r
            .backlog_since
            .is_some_and(|since| since.elapsed() >= BACKLOG_LIMIT);
        let budget = match overdue {
            false => CHUNK_PACKETS_PER_FRAME,
            true if cfg!(target_os = "android") => OVERDUE_CHUNK_PACKETS_PER_FRAME,
            true => Duration::MAX,
        };
        let mut chunks = Duration::ZERO;
        // A run of ForgetLevelChunk (thousands when the server's view moves far: a
        // teleport at 32 chunks) is forgotten at once, before whatever follows it.
        let mut forgotten = horizon_client::fast_hash::FxHashSet::default();
        while let Some(item) = events.pop_front() {
            if let Received::Event(Event::Packet(packet)) = &item
                && let C::ForgetLevelChunk(p) = &**packet
            {
                self.observe_client_packet(packet);
                forgotten.insert((p.pos[0], p.pos[1]));
                continue;
            }
            self.forget_remote_columns(&mut forgotten, true);
            let chunk = matches!(item, Received::Chunk(_));
            if chunk && chunks >= budget {
                events.push_front(item);
                break;
            }
            let started = chunk.then(Instant::now);
            let event = match item {
                Received::Event(event) => event,
                Received::Chunk(format) => {
                    // A column still decoding waits for the next frame, with
                    // what came after it, overdue or not.
                    let Some(next) = self.remote.as_ref().map(|r| r.decoder.next()) else {
                        break;
                    };
                    let (packet, column) = match next {
                        Next::Ready(decoded) => decoded,
                        Next::Decoding => {
                            events.push_front(Received::Chunk(format));
                            break;
                        }
                        Next::Lost => {
                            log::warn!("Horizon: the chunk decoder stopped; a column is missing");
                            continue;
                        }
                    };
                    self.observe_client_packet(&packet);
                    if let C::LevelChunkWithLight(p) = *packet {
                        self.remote_chunk(&p, Some((&format, column)));
                    }
                    chunks += started.map_or(Duration::ZERO, |t| t.elapsed());
                    continue;
                }
            };
            match event {
                Event::Status(s) => {
                    if let Some(r) = self.remote.as_mut() {
                        r.status = s;
                    }
                }
                Event::Joined {
                    registries,
                    name,
                    uuid,
                } => {
                    self.set_registries(registries);
                    self.extras_joined();
                    self.close_dialogs();
                    if let Some(r) = self.remote.as_mut() {
                        r.joined = true;
                        r.play_began = Some(Instant::now());
                        // handleLogin: startWaitingForNewLevel with Reason.OTHER.
                        r.profile = uuid;
                        r.profile_name = name.clone();
                        r.status = horizon_client::lang::get_or(
                            "multiplayer.downloadingTerrain",
                            "Loading terrain...",
                        );
                    }
                    self.begin_book_negotiation();
                }
                Event::Features(features) => {
                    if let Some(remote) = &mut self.remote {
                        remote.features = features.into();
                    }
                }
                Event::Brand(brand) => {
                    if let Some(remote) = &mut self.remote {
                        remote.brand = brand;
                    }
                }
                Event::QuietCommands => {
                    if let Some(remote) = &mut self.remote {
                        remote.quiet_commands = true;
                    }
                }
                Event::ModNet => {
                    if let Some(remote) = &mut self.remote {
                        remote.mod_net = true;
                    }
                }
                Event::FarTerrain(offer) => {
                    if let Some(remote) = &mut self.remote {
                        remote.far_terrain = Some(offer);
                    }
                }
                Event::Reconfiguring => {
                    log::info!("Horizon: the server is reconfiguring the connection");
                    self.reset_book_view();
                    // handleConfigurationStart: the held-back chat shows, then
                    // clearClientLevel opens ServerReconfigScreen over no level.
                    self.flush_chat_queue();
                    self.send_chat_acknowledgement();
                    self.close_dialogs();
                    // The level goes, and a new listener starts from nothing when the
                    // configuration ends; the server configuring us now (a proxy's next one)
                    // offers its own far terrain, or none, and announces its
                    // own features again.
                    self.clear_remote_world();
                    if let Some(remote) = &mut self.remote {
                        remote.far_terrain = None;
                        remote.quiet_commands = false;
                        remote.mod_net = false;
                    }
                    if let Some(remote) = &self.remote {
                        remote.session.reset_far_terrain();
                    }
                    self.remove_local_player_body();
                    self.set_screen(Screen::Playing);
                    self.capture(false);
                    // clearClientLevel drops the client level now, so its
                    // unload comes before the configuration screen, not
                    // after the next login's load.
                    self.forget_all_remote_columns();
                    if let Some(old) = self.remote.as_mut().and_then(|r| r.heard_level.take()) {
                        self.native_client_level_unload(&old);
                    }
                    self.native_lifecycle.begin_close();
                    self.native_lifecycle_world_closed();
                    let ticks = self.ticks;
                    if let Some(r) = self.remote.as_mut() {
                        r.reset_listener();
                        r.reconfiguring = Some(ticks);
                        r.chat_tree_received = false;
                    }
                }
                Event::Reconfigured(registries) => {
                    // handleConfigurationStart â†’ clearClientLevel, and the next
                    // ClientPacketListener starts with no players or scoreboard,
                    // no waypoints (its own ClientWaypointManager: a proxy's next
                    // server never untracks the last one's), and its new
                    // ClientSuggestionProvider with no custom chat completions.
                    // clearClientLevel clears the narrator.
                    self.clear_narrator();
                    self.hud_disconnected();
                    self.score.board = Default::default();
                    self.clear_waypoints();
                    if let Some(r) = self.remote.as_mut() {
                        r.attachments.clear();
                        r.players.clear();
                        r.tab_players.clear();
                        r.listed.clear();
                        r.signatures = Default::default();
                        r.custom_completions.clear();
                    }
                    self.ui.registries.custom_completions.clear();
                    self.set_registries(registries);
                    if let Some(r) = self.remote.as_mut() {
                        r.joined = true;
                        r.play_began = Some(Instant::now());
                        r.status = horizon_client::lang::get_or(
                            "multiplayer.downloadingTerrain",
                            "Loading terrain...",
                        );
                    }
                }
                // handleTransfer: "Cannot transfer to server from
                // singleplayer" (no ServerData), so the world just closes.
                Event::Transfer { host, port } => {
                    if let Some(r) = self.remote.as_mut()
                        && r.session.address != "local"
                    {
                        r.transfer = Some(format!("{host}:{port}"));
                    }
                }
                Event::PackPush(p) => self.handle_pack_push(p),
                Event::PackPop(id) => self.handle_pack_pop(id),
                Event::ServerLinks(links) => {
                    if let Some(r) = self.remote.as_mut() {
                        r.server_links = super::server_links::trusted(links);
                    }
                }
                Event::ShowDialog(dialog) => self.show_dialog(&dialog),
                Event::ClearDialog => self.clear_dialog(),
                Event::CodeOfConduct(text) => self.handle_code_of_conduct(text),
                Event::DecodeError(e) => {
                    if let Some(r) = self.remote.as_mut() {
                        r.decode_errors += 1;
                        if r.decode_errors <= 20 {
                            log::warn!("Horizon: skipped undecodable packet {e}");
                        }
                    }
                }
                Event::Disconnected {
                    reason,
                    key,
                    from_server,
                } => {
                    self.reset_book_view();
                    log::info!("Horizon: disconnected: {reason}");
                    // ClientboundTransferPacket left a target: connect there
                    // instead of showing the disconnect, the stored cookies
                    // carried over (ClientCommonPacketListenerImpl.
                    // handleTransfer â†’ ConnectScreen.startConnecting).
                    let transfer = self.remote.as_mut().and_then(|r| {
                        r.transfer.take().map(|address| {
                            (address, r.session.name.clone(), r.session.cookies.clone())
                        })
                    });
                    if let Some((address, name, cookies)) = transfer {
                        self.leave_server(None)?;
                        self.connect_server_with_cookies(&address, &name, Some(cookies))?;
                        return Ok(());
                    }
                    let retry = self.reconnect_after(&reason, key.as_deref(), from_server);
                    self.leave_server(Some(reason))?;
                    self.offer_reconnect(retry);
                    return Ok(());
                }
                Event::Packet(packet) => self.handle_packet(*packet)?,
            }
            chunks += started.map_or(Duration::ZERO, |t| t.elapsed());
            if self.remote.is_none() {
                break;
            }
        }
        if self.remote.is_some() {
            self.forget_remote_columns(&mut forgotten, true);
        }
        if let Some(r) = self.remote.as_mut() {
            r.backlog_since = if events.is_empty() {
                None
            } else {
                Some(r.backlog_since.unwrap_or_else(Instant::now))
            };
            r.backlog = events;
        }
        self.send_suggestion_request();
        Ok(())
    }
    /// ClientSuggestionProvider.customSuggestion: the chat's completion
    /// request for an `ask_server` node goes to the server.
    fn send_suggestion_request(&mut self) {
        if self.remote.is_none() {
            return;
        }
        let Some((id, command)) = self
            .ui
            .chat_screen
            .as_mut()
            .and_then(|s| s.take_request())
            .or_else(|| self.command_block_suggestion_request())
        else {
            return;
        };
        let mut w = horizon_protocol::buf::Writer::new();
        w.varint(id);
        w.string(&command);
        self.send(S::Unknown(
            horizon_protocol::ids::play::serverbound::COMMAND_SUGGESTION,
            w.data,
        ));
    }
    /// Entity.DATA_CUSTOM_NAME (2) and DATA_CUSTOM_NAME_VISIBLE (3) of any entity. Displays
    /// (text, item and block) are `text_displays.rs`'s (`remote_objects.rs` feeds them).
    fn remote_names_and_displays(&mut self, id: i32, values: &[(u8, i32, Value)]) {
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        for (index, _, value) in values {
            match (index, value) {
                (2, Value::OptText(t)) => {
                    r.names.entry(id).or_insert((None, false)).0 = t.as_ref().map(remote_text);
                }
                (3, Value::Bool(v)) => r.names.entry(id).or_insert((None, false)).1 = *v,
                _ => {}
            }
        }
    }
    /// ClientboundCommandSuggestionsPacket: id, range start and length, then
    /// each suggestion's text and optional tooltip.
    fn handle_command_suggestions(&mut self, data: &[u8]) {
        let lang = self
            .remote
            .as_ref()
            .map(|r| r.lang.clone())
            .unwrap_or_default();
        let mut r = horizon_protocol::buf::Reader::new(data);
        let decoded = (|| {
            let id = r.varint()?;
            let start = r.varint()?;
            let length = r.varint()?;
            let n = r.count(usize::MAX)?;
            let mut list = Vec::with_capacity(n.min(1000));
            for _ in 0..n {
                let text = r.string(32767)?;
                let tooltip = if r.bool()? {
                    Some(plain(&r.nbt()?, &lang))
                } else {
                    None
                };
                list.push((text, tooltip));
            }
            horizon_protocol::Result::Ok((id, start, length, list))
        })();
        if let Ok((id, start, length, list)) = decoded {
            let range = (start.max(0) as usize, length.max(0) as usize);
            self.chat_server_suggestions(id, range, list.clone());
            self.command_block_server_suggestions(id, range, list);
        }
    }
    fn set_registries(&mut self, registries: Arc<Dynamic>) {
        let entries: Vec<(&str, serde_json::Value)> = registries
            .registries
            .get("minecraft:enchantment")
            .into_iter()
            .flatten()
            .filter_map(|(name, data)| {
                Some((
                    name.as_str(),
                    horizon_client::scoreboard::tag_to_json(data.as_ref()?),
                ))
            })
            .collect();
        let enchantments = match self
            .stations()
            .enchantments
            .with_entries(entries.iter().map(|(n, v)| (*n, v)))
        {
            Ok(table) => table.map(Arc::new),
            Err(error) => {
                log::warn!("Horizon: refused enchantment registry: {error}");
                return;
            }
        };
        // The server's biomes the game lacks (a world data pack's) join the
        // game's table, from the data it sent: colours, meshing and the
        // chunk format all go through that table. The window's own server
        // thread set the world's table already.
        if self.integrated.is_none() {
            let table = registries
                .registries
                .get(BIOME_REGISTRY)
                .and_then(|list| client_biomes(self.res.biomes(), list))
                .map(Arc::new);
            if table.is_some() || self.world_biomes.is_some() {
                self.world_biomes_base = table.clone();
                self.world_biomes = table;
                let context = self.worker_context();
                self.streaming.reload_resources(&mut self.world, context);
            }
        }
        let biomes: Vec<u16> = registries
            .registries
            .get(BIOME_REGISTRY)
            .map(|list| {
                list.iter()
                    .map(|(name, _)| world_biomes!(self).id(name))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(r) = self.remote.as_mut() {
            r.enchantments = enchantments;
            let bits = ceil_log2(biomes.len());
            let column = r.chunk_format.layout();
            r.chunk_format =
                Arc::new(ChunkFormat::new(&r.dimension, biomes, bits).in_column(column));
            r.registries = registries;
            r.resolve_clocks();
        }
    }
    /// Switch to a (possibly new) dimension: a fresh client level.
    pub(in crate::game) fn enter_remote_dimension(
        &mut self,
        spawn: &play::SpawnInfo,
        keep_world: bool,
        reason: horizon_client::level_loading::Reason,
    ) {
        // Every level has the sea level its server says (`CommonPlayerSpawnInfo.seaLevel`): the
        // table may still hold a local world's (a superflat Overworld's).
        if let Ok(key) = horizon_core::LevelKey::parse(&spawn.dimension) {
            horizon_world::level::register_sea_level(key, spawn.sea_level);
            if !matches!(
                key,
                horizon_core::LevelKey::OVERWORLD
                    | horizon_core::LevelKey::NETHER
                    | horizon_core::LevelKey::END
            ) {
                Dim::add_extra(&spawn.dimension);
            }
        }
        // The replacement player starts with its inventory menu, even if the
        // server switched levels without closing the previous container.
        self.remote_close_local();
        self.set_screen(Screen::Playing);
        let Some(r) = self.remote.as_ref() else {
            return;
        };
        // handleRespawn keeps the ClientLevel only for the same level key:
        // two levels sharing a dimension type are still two levels.
        let same = keep_world && r.heard_level.as_deref() == Some(spawn.dimension.as_str());
        let transition_screen = horizon_client_render::dimension_transition::select(
            mod_api::__private::dimension_key(&spawn.dimension),
            r.heard_level
                .as_deref()
                .map(mod_api::__private::dimension_key),
        );
        if !same {
            self.forget_all_remote_columns();
        }
        let r = self.remote.as_mut().unwrap();
        // A Horizon server's bulk edit sends whole columns with the light
        // from before its relight; any other server's packet light is final.
        let relight_resent = r.quiet_commands;
        let dimension = r.registries.dimension(spawn.dimension_type);
        // The server's dimension type gives the level's shape, and its
        // chunks the shape's column.
        let shape = horizon_client::chunk::LevelShape::new(
            dimension.min_y,
            dimension.height,
            dimension.logical_height,
            dimension.has_skylight,
            dimension.has_ceiling,
        )
        .unwrap_or_else(|_| super::Dim::from_id(&dimension.name).shape());
        r.chunk_format = Arc::new(
            r.chunk_format
                .with_dimension(&dimension)
                .in_column(shape.column()),
        );
        // The server's dimension type gives the level's facts (what it does to fluids, mobs,
        // beds and time), not only its shape: a level of a mod the server runs is no
        // longer taken for the Overworld.
        r.level_facts = None;
        r.level_environment = None;
        if let Some((name, json)) = r.registries.dimension_json(spawn.dimension_type) {
            if !matches!(
                spawn.dimension.as_str(),
                "minecraft:overworld" | "minecraft:the_nether" | "minecraft:the_end"
            ) {
                r.level_environment = horizon_client::environment::Environment::with_type(
                    self.res.environment(),
                    &json,
                );
            }
            let cached = r
                .facts_cache
                .get(&spawn.dimension)
                .filter(|(old_name, old, _)| *old_name == name && *old == json)
                .map(|(_, _, facts)| facts.clone());
            r.level_facts = cached.or_else(|| {
                use horizon_world::dimension_type::DimensionFacts;
                let base = DimensionFacts::base_for_type(&name);
                let facts = (|| {
                    for field in ["coordinate_scale", "ambient_light"] {
                        if json.get(field).is_some()
                            && !json[field].as_f64().is_some_and(f64::is_finite)
                        {
                            return Err(format!("invalid {field}"));
                        }
                    }
                    if json.get("monster_spawn_block_light_limit").is_some()
                        && json["monster_spawn_block_light_limit"].as_i64().is_none()
                    {
                        return Err("invalid monster_spawn_block_light_limit".into());
                    }
                    DimensionFacts::patched(base, &name, &json)
                })()
                .map_err(|why| log::warn!("the server's dimension type {name} is refused ({why})"))
                .unwrap_or_else(|_| base.clone());
                if !sane_facts(&facts) {
                    log::warn!(
                        "the server's dimension type {name} is refused (a value out of range)"
                    );
                    return Some(Arc::new(base.clone()));
                }
                let facts = Arc::new(facts);
                r.facts_cache.insert(
                    spawn.dimension.clone(),
                    (name.clone(), json.clone(), facts.clone()),
                );
                Some(facts)
            });
        }
        r.dimension = dimension;
        r.resolve_clocks();
        self.world.day_time = r.clock_time(r.default_clock);
        r.level = spawn.dimension.clone();
        r.level_key = mod_api::__private::dimension_key(&spawn.dimension);
        r.hashed_seed = spawn.seed;
        r.game_mode = spawn.game_type;
        r.flat = spawn.flat;
        r.loaded_sent = false;
        r.abilities = None;
        // A replacement LevelLoadingScreen starts its bar from nothing (the
        // first one carries on from the server's start).
        if self.player.is_some() {
            self.loading_progress = 0.0;
        }
        // startWaitingForNewLevel: `LevelLoadTracker.startClientLoad` and the
        // `LevelLoadingScreen`.
        r.level_load.start_client_load(client_millis(), reason);
        r.level_load.set_transition_screen(transition_screen);
        // A replacement LocalPlayer has a fresh ambient mood and sendPosition baseline.
        self.ambience.moodiness = 0.0;
        r.position_sender = Default::default();
        r.predictions.clear();
        self.set_mode_id(spawn.game_type);
        self.dim = Dim::from_id(&spawn.dimension);
        self.audio.set_world(&spawn.dimension);
        // ClientPacketListener.handleLogin / handleRespawn: the server's
        // Player.getLastDeathLocation, for the recovery compass.
        self.level.last_death = spawn
            .last_death
            .as_ref()
            .map(|d| (d.dimension.clone(), d.pos));
        if !same {
            // A new ClientLevel: Minecraft.setLevel stops every sound.
            self.stop_sounds_for_new_level();
            let mut streaming = Streaming::new(
                Arc::new(NoTerrain),
                self.worker_context(),
                self.res.threads(),
            );
            streaming.remote = true;
            streaming.radius = self.streaming.radius;
            self.streaming = streaming;
            let (time, day, weather) = (
                self.world.game_time,
                self.world.day_time,
                self.world.weather,
            );
            self.reset_client_world(spawn.seed);
            // Entity IDs belong to this ClientLevel, not the connection. A
            // same-level respawn keeps it; a replacement invalidates handles.
            self.native_lifecycle_open_started();
            self.native_lifecycle_world_ready();
            self.world.shape = shape;
            self.world.game_time = time;
            self.world.day_time = day;
            self.world.weather = weather;
            self.mobs.clear();
            self.clear_remote_objects();
            self.clear_remote_projectiles();
            self.dropped.clear();
            self.clear_client_particles();
            if let Some(r) = self.remote.as_mut() {
                r.entities.clear();
                r.passenger_vehicles.clear();
                r.lerps.clear();
                r.cart_lerps.clear();
                r.stepped.clear();
                r.head_lerps.clear();
                r.pending_items.clear();
                // The entity ids are the old world's; the new dimension's
                // player entities re-map to their profiles on AddEntity.
                r.level_players.clear();
                r.level_player_names.clear();
                r.level_player_textures.clear();
                // A new ClientLevel: a new chunk storage, centred on 0, 0.
                r.chunk_center = (0, 0);
            }
            self.clear_lod();
            self.pending_uploads.clear();
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.clear_sections(self.world.layout());
            }
            // The new ClientLevel is told before setLevel drops the old one.
            let old = self
                .remote
                .as_mut()
                .and_then(|r| r.heard_level.replace(spawn.dimension.clone()));
            self.native_client_level_load(&spawn.dimension);
            if let Some(old) = old {
                self.native_client_level_unload(&old);
            }
        }
        self.streaming.relight_resent = relight_resent;
        let has_sky = self
            .remote
            .as_ref()
            .is_none_or(|r| r.dimension.has_skylight);
        self.streaming.sky_light = has_sky;
        self.streaming.nether_shading = self.dim == Dim::Nether;
        // The seed in SpawnInfo is already the hashed one the client's BiomeManager holds.
        self.streaming.biome_zoom = horizon_core::biome::BiomeZoom::new(spawn.seed);
        let player = self.player.as_ref();
        self.particles
            .detach_tracking_target(TrackingTarget::LocalPlayer, || {
                player.map(player_tracking_body)
            });
        self.remove_local_player_body();
        // handleLogin / handleRespawn: LocalPlayer.setPortalCooldown, so a
        // player who arrives standing in a portal gets no overlay from it.
        self.portals.player.cooldown = spawn.portal_cooldown;
        if let Some(profile) = self.remote.as_ref().map(|remote| remote.profile) {
            self.initialize_entity(Holder::Player(profile), "minecraft:player");
        }
        self.apply_settings();
    }

    /// `ClientChunkCache.Storage.inRange`: within the storage around the
    /// view centre, `calculateStorageRange` (the server's radius, at least
    /// two, plus three) chunks each way.
    pub(super) fn remote_chunk_in_range(&self, x: i32, z: i32) -> bool {
        if self.replay.playing() {
            return true;
        }
        self.remote
            .as_ref()
            .is_none_or(|r| in_storage_range(r.view_distance, r.chunk_center, x, z))
    }
    fn chunk_inside_remote_range(center: (i32, i32), distance: i32, x: i32, z: i32) -> bool {
        in_storage_range(distance, center, x, z)
    }
    /// The columns a moved or resized storage no longer holds. Returns the
    /// positions it freed (`forget_remote_columns`).
    fn drop_remote_chunks_out_of_range(
        &mut self,
        center: (i32, i32),
        distance: i32,
    ) -> Vec<(i32, i32)> {
        let mut gone: horizon_client::fast_hash::FxHashSet<(i32, i32)> = self
            .world
            .chunks
            .keys()
            .copied()
            .filter(|&(x, z)| !Self::chunk_inside_remote_range(center, distance, x, z))
            .collect();
        self.forget_remote_columns(&mut gone, true)
    }
    /// Every column goes with the client level. Returns the positions it
    /// freed (`forget_remote_columns`).
    fn forget_all_remote_columns(&mut self) -> Vec<(i32, i32)> {
        let mut gone = self.world.chunks.keys().copied().collect();
        self.forget_remote_columns(&mut gone, false)
    }
    /// Forgets the `gone` columns (and empties the set): their chunks (freed on a thread
    /// of their own when there are many), their sections in the renderer and their block
    /// entities, each store once for all of them. When the level stays open,
    /// far terrain keeps their last state (`server_terrain`). Returns the
    /// positions it freed, in `gone`'s order: only those still resident,
    /// so a later forget of the same positions frees nothing.
    fn forget_remote_columns(
        &mut self,
        gone: &mut horizon_client::fast_hash::FxHashSet<(i32, i32)>,
        keep_terrain: bool,
    ) -> Vec<(i32, i32)> {
        if gone.is_empty() {
            return Vec::new();
        }
        // The client chunk seam: one unload per column still resident, with
        // its data still there, before `take_remote_columns` frees it.
        let freed_at: Vec<(i32, i32)> = gone
            .iter()
            .filter(|&&(x, z)| self.world.chunks.contains_key(&(x, z)))
            .copied()
            .collect();
        for &(x, z) in &freed_at {
            self.client_chunk_unload(x, z);
        }
        let freed =
            horizon_client::streaming::Streaming::take_remote_columns(&mut self.world, gone);
        self.streaming.forget_columns(&mut self.world, gone);
        if keep_terrain && let Some(lod) = &mut self.lod {
            for chunk in &freed {
                lod.observe(chunk.clone(), self.res.facts());
            }
        }
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.remove_columns(gone.iter().copied());
        }
        horizon_client::chunk::free_columns(freed, gone.len());
        self.world
            .block_entities
            .unload(|k| gone.contains(&(k.0 >> 4, k.2 >> 4)));
        if let Some(r) = self.remote.as_mut() {
            r.attachments.retain(|holder, _| match *holder {
                Holder::Chunk([x, z]) => !gone.contains(&(x, z)),
                Holder::BlockEntity([x, _, z]) => !gone.contains(&(x >> 4, z >> 4)),
                _ => true,
            });
        }
        gone.clear();
        freed_at
    }
    /// `handleBlockEntityData`: the block entity at `pos`, when it is of the
    /// packet's type, loads the tag (`loadWithComponents`); an empty tag
    /// changes nothing.
    pub(super) fn remote_block_entity_data(&mut self, pos: [i32; 3], kind: i32, tag: &Tag) {
        // Starling: a content mod's machine (kinetics.rs). Its tag is kept as the chunk keeps the
        // block entities it does not model, for the window to draw what the machine holds.
        if kind == horizon_client::lan::chunks::MACHINE_ENTITY_TYPE {
            if let Tag::Compound(fields) = tag
                && !fields.is_empty()
            {
                let mut full = fields.clone();
                for (k, v) in ["x", "y", "z"].into_iter().zip(pos) {
                    full.insert(k.into(), Tag::Int(v));
                }
                super::spill::put_saved_entity(&mut self.world, pos, Tag::Compound(full));
            }
            return;
        }
        let key = (pos[0], pos[1], pos[2]);
        let Some(name) = self
            .remote
            .as_ref()
            .and_then(|r| r.builtin.get("minecraft:block_entity_type")?.name(kind))
        else {
            return;
        };
        // A decorated pot is no block entity here: its tag is kept with the chunk.
        if name == "minecraft:decorated_pot" {
            if let Tag::Compound(fields) = tag
                && !fields.is_empty()
            {
                let mut full = fields.clone();
                full.insert("id".into(), Tag::String(name.into()));
                for (k, v) in ["x", "y", "z"].into_iter().zip(pos) {
                    full.insert(k.into(), Tag::Int(v));
                }
                super::spill::put_saved_entity(&mut self.world, pos, Tag::Compound(full));
                // The section is drawn again with the pot's sherds.
                let state = self.world.block(pos[0], pos[1], pos[2]);
                let _ = self.world.set(pos[0], pos[1], pos[2], state);
            }
            return;
        }
        let Tag::Compound(fields) = tag else {
            return;
        };
        // Only the declared type on one of its valid blocks accepts an update tag.
        // Nothing is drawn from it yet, so its section stays as it is.
        let block = self.world.block(pos[0], pos[1], pos[2]);
        if let Some(BlockEntity::Mod(m)) = self.world.block_entities.get_mut(&key) {
            if m.id() == name
                && horizon_client::block_entity_types::valid(
                    &self.res.facts().get(block).name,
                    name,
                ) == Some(true)
            {
                m.merge_update_tag(fields);
            }
            return;
        }
        let Some(entity) = self.world.block_entities.get(&key) else {
            return;
        };
        if entity.id() != name || fields.is_empty() {
            return;
        }
        let mut full = fields.clone();
        full.insert("id".into(), Tag::String(name.into()));
        for (k, v) in ["x", "y", "z"].into_iter().zip(pos) {
            full.insert(k.into(), Tag::Int(v));
        }
        if let Some((_, mut loaded)) =
            horizon_client::save::decode_block_entity(&Tag::Compound(full))
        {
            keep_client_state(self.world.block_entities.get(&key), &mut loaded);
            let command_block = matches!(loaded, BlockEntity::CommandBlock(_));
            self.world.block_entities.insert(key, loaded);
            self.update_spawner_display(key);
            // The section is drawn again with what the block entity shows.
            let state = self.world.block(pos[0], pos[1], pos[2]);
            let _ = self.world.set(pos[0], pos[1], pos[2], state);
            if command_block {
                self.command_block_data_arrived(key);
            }
        }
    }
    /// `handleAnimate`: a player wakes up, or an entity gets the critical
    /// hit (`CRIT`) or magic critical hit (`ENCHANTED_HIT`) emitter.
    fn remote_animate(&mut self, id: i32, action: u8) {
        use horizon_client::particles::{ParticleOptions, ParticleType};
        let Some((target, body)) = self.remote_tracking_body(id) else {
            return;
        };
        let particle = match action {
            // Player.stopSleepInBed.
            0 => {
                if target == TrackingTarget::LocalPlayer {
                    self.sleeping = None;
                    self.sleep_timer = 100;
                }
                return;
            }
            1 => ParticleType::Crit,
            2 => ParticleType::EnchantedHit,
            _ => return,
        };
        self.particles
            .tracking_emitter(ParticleOptions::simple(particle), target, body);
    }
    /// `handleServerData`: the server list entry of the server joined keeps
    /// the icon the server sends (`ServerList.saveSingleServer`). Single
    /// player has no entry.
    fn remote_server_data(&mut self, icon: Option<&[u8]>) {
        let (Some(r), Some(png)) = (self.remote.as_ref(), icon) else {
            return;
        };
        if self.integrated.is_some() {
            return;
        }
        if let Ok(dir) = std::env::current_dir() {
            horizon_client::multiplayer::save_server_icon(
                &dir.join("saves"),
                &r.session.address,
                png,
            );
        }
    }
    /// Keep the renderable block entities (chests, furnaces) in step with blocks.
    /// `previous`: the block's state before, when it changed. A column sent again has
    /// none, and keeps its block entities' synced attachments: the server does not send
    /// them again (`remote_chunk` drops those of block entities the column lost).
    pub(super) fn sync_block_entity(
        &mut self,
        pos: (i32, i32, i32),
        previous: Option<BlockStateId>,
        state: BlockStateId,
    ) {
        let name = self.res.facts().get(state).name.clone();
        // LevelChunk.setBlockState: another block's block entity goes, even
        // one of the same type (a mod's blocks may share one).
        let other_block = previous.is_some_and(|p| self.res.facts().get(p).name != name);
        self.ensure_stateful_entity_table();
        // State-aware factories decide creation only. A valid same-block
        // instance survives even when the new state would create none.
        if !other_block
            && self.world.block_entities.stateful_entity(state.0)
            && matches!(
                self.world.block_entities.get(&pos),
                Some(BlockEntity::Mod(_))
            )
        {
            return;
        }
        // A mod block's is the mod's own, as on the server.
        match self.new_block_entity_for_state(state) {
            Some(entity) => {
                let same = self.world.block_entities.get(&pos).is_some_and(|e| {
                    e.id() == entity.id() && !(other_block && matches!(e, BlockEntity::Mod(_)))
                });
                if !same {
                    if previous.is_some() {
                        self.forget_block_entity_attachments(pos);
                    }
                    self.world.block_entities.insert(pos, entity);
                }
            }
            None => {
                if previous.is_some() {
                    self.forget_block_entity_attachments(pos);
                }
                self.world.block_entities.remove(&pos);
            }
        }
    }
    /// The server's synced attachments of a block entity that goes with it.
    fn forget_block_entity_attachments(&mut self, (x, y, z): (i32, i32, i32)) {
        if let Some(r) = self.remote.as_mut() {
            r.attachments.remove(&Holder::BlockEntity([x, y, z]));
        }
    }
    /// ClientLevel.setServerVerifiedBlockState: a block the client predicted keeps its
    /// predicted state until the server acknowledges the action; the server's state is
    /// remembered for then.
    fn server_block(&mut self, pos: [i32; 3], state: i32) {
        if let Some(p) = self
            .remote
            .as_mut()
            .and_then(|r| r.predictions.get_mut(&pos))
        {
            p.state = state;
            return;
        }
        self.set_block(pos, state);
    }
    /// BlockStatePredictionHandler.endPredictionsUpTo (ClientboundBlockChangedAckPacket): the
    /// actions up to `sequence` are settled, so the blocks they predicted take the state the
    /// server gave them (a break the server refused comes back).
    fn end_predictions(&mut self, sequence: i32) {
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let last_teleport = r.last_teleport_sequence;
        let settled: Vec<_> = r
            .predictions
            .iter()
            .filter(|(_, p)| p.sequence <= sequence)
            .map(|(&pos, &p)| (pos, p))
            .collect();
        for (pos, prediction) in settled {
            if let Some(r) = self.remote.as_mut() {
                r.predictions.remove(&pos);
            }
            if self.world.block(pos[0], pos[1], pos[2]).0 as i32 == prediction.state {
                continue;
            }
            self.set_block(pos, prediction.state);
            // ClientLevel.syncBlockState: the block that comes back may now hold
            // the player, and then it goes to the position the prediction
            // remembered.
            let meets = self.block_meets_player(pos, prediction.state);
            if let Some(at) = snap_target(prediction.player_pos, last_teleport, sequence, meets) {
                self.snap_player_to(at);
            }
        }
    }
    /// The player's position, for `retainKnownServerState`'s `player.position()`.
    fn player_position(&self) -> [f64; 3] {
        let at = self
            .player
            .as_ref()
            .map(|p| p.position)
            .unwrap_or(Vec3::ZERO);
        [at.x, at.y, at.z]
    }
    /// `Entity.isColliding(pos, state)` for the local player: the state's own
    /// collision shape, at the player's pose, against the player's box.
    fn block_meets_player(&self, pos: [i32; 3], state: i32) -> bool {
        let Some(player) = self.player.as_ref() else {
            return false;
        };
        let entity = player.bounds();
        let context = horizon_world::collision_context::CollisionContext::for_entity(
            horizon_world::collision_context::CollisionEntity {
                identity: horizon_world::entity_obstruction::EntityIdentity(0),
                kind: horizon_world::collision_context::EntityContextKind::Player,
                y: player.position.y,
                descending: player.shift_key_down,
                main_hand: None,
            },
            false,
        );
        let mut scene = horizon_client::collisions::SceneCollisions {
            world: &self.world,
            binding: self.collision,
            subject: None,
        };
        let at = horizon_core::position::BlockPos::new(pos[0], pos[1], pos[2]);
        match self.collision.collision_shape_with_context(
            &mut scene,
            BlockStateId(state.max(0) as u32),
            at,
            &context,
        ) {
            Ok(shape) => shape_meets(&shape, pos, entity),
            Err(_) => false,
        }
    }
    /// `Entity.absSnapTo`: the remembered point, clamped to the level's edge.
    /// The velocity is left alone, and the player has no separate previous
    /// position to reset (its box is drawn from `position`).
    fn snap_player_to(&mut self, at: [f64; 3]) {
        let Some(player) = self.player.as_mut() else {
            return;
        };
        let x = at[0].clamp(-3.0e7, 3.0e7);
        let z = at[2].clamp(-3.0e7, 3.0e7);
        player.position = Vec3::new(x, at[1], z);
    }
    fn set_block(&mut self, pos: [i32; 3], state: i32) {
        // A state we don't have (a server with mods we lack) reads as air, as in its chunks.
        let state = state.max(0) as u32;
        let known = horizon_client::chunk::known_block_state(state);
        let state = BlockStateId(if known { state } else { 0 });
        let previous = self.world.block(pos[0], pos[1], pos[2]);
        if self.world.loaded(pos[0], pos[2])
            && self.world.set(pos[0], pos[1], pos[2], state).is_ok()
        {
            self.sync_block_entity((pos[0], pos[1], pos[2]), Some(previous), state);
            // LevelChunk.setBlockState removes the block entity of a block
            // that changes: the moving piston the client's moveBlocks made
            // stops when the server places what it moved (#3002).
            let facts = self.res.facts();
            let moving = |s: BlockStateId| &*facts.get(s).name == "minecraft:moving_piston";
            if moving(previous) && !moving(state) {
                let at = horizon_core::position::BlockPos::new(pos[0], pos[1], pos[2]);
                self.block_ticks.moving_pistons.retain(|(p, _)| *p != at);
                self.moving_pistons_stale = true;
            }
        }
        // A block we just placed: play its place sound (the server skips the placer).
        if let Some((target, _)) = self.remote.as_ref().and_then(|r| r.placing)
            && target == pos
            && !self.res.facts().is_air(state)
        {
            let s = self.res.facts().get(state).sounds.clone();
            let at = [
                f64::from(pos[0]) + 0.5,
                f64::from(pos[1]) + 0.5,
                f64::from(pos[2]) + 0.5,
            ];
            self.play_client_local_sound(
                LevelSound::new(&s.place, at, (s.volume + 1.0) / 2.0, s.pitch * 0.8)
                    .in_source(mod_api::SoundSource::Blocks),
            );
            if let Some(r) = self.remote.as_mut() {
                r.placing = None;
            }
        }
    }
    /// `Entity.isVehicle` on a client: the entities a tracked entity rides.
    pub(super) fn remote_vehicle_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.remote
            .iter()
            .flat_map(|r| r.entities.values().filter_map(|t| t.vehicle))
    }
    /// The `SetPassengers` bookkeeping of the tracked entities.
    pub(super) fn remote_seat_passengers(&mut self, vehicle: u32, passengers: &[i32]) {
        if let Some(r) = self.remote.as_mut() {
            seat_passengers(&mut r.entities, vehicle, passengers);
        }
    }
    /// `recreateFromPacket`: yRot and xRot from the packet, the body and head
    /// both at its head yaw (LivingEntity).
    pub(super) fn client_attachment_entity_exists(&self, holder: Holder) -> bool {
        self.remote.as_ref().is_some_and(|remote| match holder {
            Holder::Entity(id) => remote.entities.contains_key(&(id as i32)),
            Holder::Player(uuid) => {
                remote.profile == uuid || remote.level_players.values().any(|p| *p == uuid)
            }
            _ => false,
        })
    }

    pub(super) fn spawn_entity(
        &mut self,
        id: i32,
        kind: &str,
        position: [f64; 3],
        rotation: [f32; 3],
    ) {
        self.detach_remote_tracking_entity(id);
        let [yaw, pitch, head] = rotation;
        let at = Vec3::new(position[0], position[1], position[2]);
        let mob_type = if kind == "minecraft:player" {
            self.remote.as_ref().and_then(|r| r.player_type.clone())
        } else {
            self.res.mob_types().get(kind).cloned()
        };
        if kind == "minecraft:ender_dragon" {
            self.spawn_remote_dragon(id, position, yaw);
        }
        if kind == "minecraft:experience_orb" {
            // Its value (ExperienceOrb.DATA_VALUE) follows in SetEntityData.
            self.orbs.retain(|o| o.id != id as u32);
            self.orbs.push(ExperienceOrb {
                attachment_lifetime: Default::default(),
                rotation: [0.0, 0.0],
                join: Default::default(),
                no_gravity: false,
                id: id as u32,
                position: at,
                previous: at,
                velocity: Vec3::ZERO,
                value: 1,
                age: 0,
                on_ground: false,
                count: 1,
                ticks: 0,
                health: 5,
                following: None,
            });
        }
        if let Some(t) = mob_type {
            // The window's copy of the server's mob: no mod part goals â€”
            // their `register_goals` ran on the authoritative side (#1077).
            let mut mob = Mob::new_remote(id as u32, t, at, head);
            super::remote_mob_data::client_defaults(&mut mob);
            (mob.yaw, mob.previous_yaw) = (yaw, yaw);
            (mob.head_pitch, mob.previous_head_pitch) = (pitch, pitch);
            mob.persistent = true;
            self.mobs.retain(|m| m.id != id as u32);
            self.mobs.push(mob);
        }
        if let Some(r) = self.remote.as_mut() {
            // A new entity has a new interpolation handler (a mob's or an
            // armor stand's), even under an id the server used before.
            r.stepped.remove(&id);
            if kind == "minecraft:item" {
                r.pending_items.insert(id, (position, Vec3::ZERO));
            }
            r.entities.insert(
                id,
                Tracked {
                    kind: kind.to_owned(),
                    position,
                    codec_base: position,
                    silent: false,
                    vehicle: None,
                },
            );
        }
    }
    /// A move packet for entity `id`: to the end of `path` (None: a
    /// rotation alone), turned to `rotation` if it has one. `packet` says
    /// whether a living entity more than 64 blocks away is put there at once
    /// (its `snaps`).
    fn move_entity(
        &mut self,
        id: i32,
        path: Option<Path>,
        rotation: Option<(f32, f32)>,
        on_ground: bool,
        packet: MovePacket,
    ) {
        // The dragon is drawn from its own state (game/mobs/dragon.rs), which
        // tick_remote_dragon eases to the server's moves through `lerps`. Its
        // copy in the mob list draws nothing: given the move, it kept the
        // dragon where it spawned (#2083).
        let dragon = self.dragon.as_ref().is_some_and(|d| d.id as i32 == id);
        let driven = self.driven_mount().is_some_and(|m| m.id as i32 == id);
        let mob = self
            .mob_index
            .get_mut(&mut self.mobs, id as u32)
            .filter(|_| !dragon);
        let tracked = self
            .remote
            .as_ref()
            .and_then(|r| r.entities.get(&id))
            .map(|t| t.position);
        let position = path
            .as_ref()
            .map(Path::end)
            .or(tracked)
            .or_else(|| {
                mob.as_ref()
                    .map(|m| [m.position.x, m.position.y, m.position.z])
            })
            .unwrap_or([0.0; 3]);
        if let Some(r) = self.remote.as_mut() {
            if let Some(t) = r.entities.get_mut(&id) {
                t.position = position;
            }
            if driven {
                // handleMoveEntity, handleEntityPositionSync: the mount this
                // client moves (isLocalInstanceAuthoritative) keeps its own
                // position, and no steps pile up for after the ride.
                r.stepped.remove(&id);
                return;
            }
            if let Some(m) = mob {
                // Shulker.createInterpolationHandler is NO_OP, so a move packet
                // is setPos at once. The 6-tick slide starts from the block
                // the shulker occupied before this packet.
                if m.species() == horizon_client::mobs::Species::Shulker {
                    r.stepped.remove(&id);
                    if path.is_some() {
                        horizon_client::mobs::shulker::client_set_pos(
                            m,
                            position[0],
                            position[1],
                            position[2],
                        );
                    }
                    if let Some((yaw, pitch)) = rotation {
                        m.yaw += wrap_degrees(yaw - m.yaw);
                        m.head_pitch = (pitch % 360.0).clamp(-90.0, 90.0);
                    }
                    m.on_ground = on_ground;
                    return;
                }
                // A living entity: LivingEntity.createInterpolationHandler's
                // SteppedInterpolationHandler, over its type's update interval.
                let entity = PosRot {
                    position: [m.position.x, m.position.y, m.position.z],
                    yaw: m.yaw,
                    pitch: m.head_pitch,
                };
                let handler = r.stepped.entry(id).or_insert_with(|| {
                    let interval = horizon_client::lan::tracking::settings(&m.kind.id).1;
                    SteppedInterpolation::new(interval as i32)
                });
                if let Some(at) = stepped_move(handler, packet, entity, position, path, rotation) {
                    // snapTo: setOldPosAndRot too.
                    place_interpolated(m, at);
                    m.previous = m.position;
                    m.previous_yaw = m.yaw;
                    m.previous_head_pitch = m.head_pitch;
                }
                m.on_ground = on_ground;
                return;
            }
            // An armor stand is a living entity too: the same handler, over
            // its type's update interval, stepped in `tick_remote_entities`.
            if let Some(a) = self.armor_stands.iter_mut().find(|a| a.id == id as u32) {
                let entity = PosRot {
                    position: [a.position.x, a.position.y, a.position.z],
                    yaw: a.yaw,
                    pitch: 0.0,
                };
                let handler = r.stepped.entry(id).or_insert_with(|| {
                    let interval =
                        horizon_client::lan::tracking::settings("minecraft:armor_stand").1;
                    SteppedInterpolation::new(interval as i32)
                });
                if let Some(at) = stepped_move(handler, packet, entity, position, path, rotation) {
                    a.snap(
                        Vec3::new(at.position[0], at.position[1], at.position[2]),
                        at.yaw,
                    );
                }
                a.on_ground = on_ground;
                return;
            }
            // Interaction extends Entity and keeps its NO_OP interpolation
            // handler: the server's position replaces its own at once.
            if let Some(i) = self.interactions.get_mut(&id) {
                if path.is_some() {
                    i.position = Vec3::new(position[0], position[1], position[2]);
                }
                return;
            }
            // Projectiles keep Entity's NO_OP InterpolationHandler too: the
            // server's position replaces theirs at once and their own tick
            // flies them on (`tick_remote_projectiles`). Eased towards it
            // instead, they only moved when an update came, every 10 to 20
            // ticks.
            if let Some(i) = self
                .creatures
                .projectiles
                .iter()
                .position(|p| p.id == id as u32)
            {
                self.creatures.projectiles.update(i, |p| {
                    if path.is_some() {
                        p.position = Vec3::new(position[0], position[1], position[2]);
                    }
                    if let Some((yaw, pitch)) = rotation {
                        p.yaw = yaw;
                        p.pitch = pitch;
                    }
                });
                return;
            }
            if path.is_some()
                && (self.potions.move_remote_thrown(
                    id as u32,
                    Vec3::new(position[0], position[1], position[2]),
                ) || self.end.move_remote_eye(id as u32, position))
            {
                return;
            }
            // A fishing hook keeps a LinearInterpolationHandler: eased towards the
            // update over three ticks, on top of its own movement (`tick_remote_hooks`).
            if path.is_some() {
                if let Some(h) = self.hook.as_mut().filter(|h| h.id() == id as u32) {
                    h.move_to(position, on_ground);
                    return;
                }
                if let Some(h) = self.remote_hooks.iter_mut().find(|h| h.id() == id as u32) {
                    h.move_to(position, on_ground);
                    return;
                }
            }
            // A thrown trident is an arrow: placed at once, flown on by its tick.
            if let Some(t) = self.tridents.iter_mut().find(|t| t.id == id as u32) {
                if path.is_some() {
                    t.position = Vec3::new(position[0], position[1], position[2]);
                }
                if let Some((yaw, pitch)) = rotation {
                    t.yaw = yaw;
                    t.pitch = pitch;
                }
                return;
            }
            // So does a firework rocket: its own tick flies it on.
            if path.is_some()
                && let Some(f) = self.rockets.iter_mut().find(|f| f.id == id as u32)
            {
                f.position = Vec3::new(position[0], position[1], position[2]);
                return;
            }
            if let Some(p) = r.pending_items.get_mut(&id) {
                p.0 = position;
            }
            // ItemEntity, FallingBlockEntity and PrimedTnt keep Entity's
            // NO_OP InterpolationHandler: moveOrInterpolateTo puts them there
            // at once, and their own tick moves them on between the server's
            // updates (`tick_remote_entities`, `tick_remote_projectiles`).
            let at = Vec3::new(position[0], position[1], position[2]);
            if let Some(d) = self.dropped.iter_mut().find(|d| d.id == id as u32) {
                d.position = at;
                d.on_ground = on_ground;
                return;
            }
            if let Some(i) = self.falling.iter().position(|f| f.id == id as u32) {
                self.falling.update(i, |f| f.position = at);
                return;
            }
            if let Some(t) = self.tnts.iter_mut().find(|t| t.id == id as u32) {
                t.position = at;
                t.on_ground = on_ground;
                return;
            }
            // ExperienceOrb.createInterpolationHandler is a
            // LinearInterpolationHandler: the correction is spread over 3
            // ticks (`tick_remote_entities`), while `ExperienceOrb.tick` keeps
            // moving it.
            if let Some(o) = self.orbs.iter_mut().find(|o| o.id == id as u32) {
                o.on_ground = on_ground;
                r.lerps.insert(
                    id,
                    Lerp {
                        target: position,
                        rotation: None,
                        steps: 3,
                    },
                );
                return;
            }
            if r.cart_lerps.contains_key(&id) {
                // A minecart on NewMinecartBehavior keeps Entity's NO_OP
                // InterpolationHandler: a teleport puts it there at once.
                if let Some(i) = self.carts.iter().position(|c| c.id == id as u32) {
                    self.carts.update(i, |c| {
                        if path.is_some() {
                            c.position = at;
                        }
                        if let Some((yaw, pitch)) = rotation {
                            c.yaw = yaw;
                            c.pitch = pitch;
                        }
                    });
                }
            } else {
                // AbstractInterpolationHandler.interpolateTo: without a
                // rotation, the one it was heading for.
                let rotation = rotation.or_else(|| r.lerps.get(&id).and_then(|l| l.rotation));
                r.lerps.insert(
                    id,
                    Lerp {
                        target: position,
                        rotation,
                        steps: 3,
                    },
                );
            }
        }
        self.turn_remote_object(id, rotation.map(|(yaw, _)| yaw), on_ground);
        self.move_remote_text_display(id, position, rotation);
        self.turn_remote_projectile(id, rotation);
    }
    pub(super) fn detach_remote_tracking_entity(&mut self, id: i32) {
        let mobs = &self.mobs;
        let index = &mut self.mob_index;
        self.particles
            .detach_tracking_target(TrackingTarget::Entity(id), || {
                index.get(mobs, id as u32).map(mob_tracking_body)
            });
    }

    /// Map a menu slot to Horizon's inventory screen slot ids.
    // Starling: open to the game, for the old helper native mods call (`menu_slot_old`).
    pub(in crate::game) fn inventory_screen_slot(menu: &str, slot: i32) -> Option<usize> {
        let s = usize::try_from(slot).ok()?;
        // A container of `n` slots, then the inventory and the hotbar.
        let slots = |n: usize| -> Option<usize> {
            match s {
                _ if s < n => Some(300 + s),
                _ if s < n + 27 => Some(9 + s - n),
                _ if s < n + 36 => Some(s - n - 27),
                _ => None,
            }
        };
        let rows = |n: usize| slots(n * 9);
        match menu {
            "inventory" => match s {
                0 => Some(200),
                1..=4 => Some(100 + s - 1),
                5..=8 => Some(500 + s - 5),
                9..=35 => Some(s),
                36..=44 => Some(s - 36),
                // InventoryMenu.SHIELD_SLOT: the off hand.
                45 => Some(504),
                _ => None,
            },
            "minecraft:crafting" => match s {
                0 => Some(200),
                1..=9 => Some(100 + s - 1),
                10..=36 => Some(9 + s - 10),
                37..=45 => Some(s - 37),
                _ => None,
            },
            "minecraft:merchant" => match s {
                0 | 1 => Some(super::stations::INPUT + s),
                2 => Some(super::stations::RESULT),
                3..=29 => Some(9 + s - 3),
                30..=38 => Some(s - 30),
                _ => None,
            },
            "minecraft:furnace" | "minecraft:blast_furnace" | "minecraft:smoker" => match s {
                0..=2 => Some(300 + s),
                3..=29 => Some(9 + s - 3),
                30..=38 => Some(s - 30),
                _ => None,
            },
            "minecraft:generic_9x1" => rows(1),
            "minecraft:generic_9x2" => rows(2),
            "minecraft:generic_9x3" | "minecraft:shulker_box" => rows(3),
            "minecraft:generic_9x4" => rows(4),
            "minecraft:generic_9x5" => rows(5),
            "minecraft:generic_9x6" => rows(6),
            // HopperMenu; DispenserMenu (dispensers and droppers).
            "minecraft:hopper" => slots(5),
            "minecraft:generic_3x3" => slots(9),
            "minecraft:brewing_stand" => slots(5),
            // CrafterMenu: the grid, the inventory, then the result (the
            // client's own).
            "minecraft:crafter_3x3" => slots(9),
            // HorseInventoryMenu: the saddle, the body armour, the chest,
            // then the inventory.
            m if m.starts_with(MOUNT_MENU) => {
                let n = m[MOUNT_MENU.len()..].parse::<usize>().ok()? * 3;
                match s {
                    0 => Some(super::mount_menu::SADDLE),
                    1 => Some(super::mount_menu::ARMOR),
                    _ if s < 2 + n => Some(super::mount_menu::CHEST + s - 2),
                    _ if s < 2 + n + 27 => Some(9 + s - 2 - n),
                    _ if s < 2 + n + 36 => Some(s - 2 - n - 27),
                    _ => None,
                }
            }
            // A content mod's menu type: its container, then the inventory.
            // A workstation: its inputs, its result, then the inventory.
            m => {
                if let Some(t) = horizon_client::containers::mod_menu_type(m) {
                    return slots(t.slots);
                }
                let kind = Self::remote_station_kind(m)?;
                let inputs = kind.inputs();
                let own = inputs + usize::from(kind.has_result());
                match s {
                    _ if s < inputs => Some(super::stations::INPUT + s),
                    _ if s < own => Some(super::stations::RESULT),
                    _ if s < own + 27 => Some(9 + s - own),
                    _ if s < own + 36 => Some(s - own - 27),
                    _ => None,
                }
            }
        }
    }
    /// Find a game slot in the menu's complete, contiguous slot mapping.
    pub(super) fn menu_slot_index(menu: &str, internal: usize) -> Option<usize> {
        (0..)
            .map_while(|slot| Self::inventory_screen_slot(menu, slot))
            .position(|slot| slot == internal)
    }
    /// The workstation screen of a menu type a server opens.
    pub(super) fn remote_station_kind(menu: &str) -> Option<super::stations::Kind> {
        use super::stations::Kind;
        Some(match menu {
            "minecraft:loom" => Kind::Loom,
            "minecraft:stonecutter" => Kind::Stonecutter,
            "minecraft:grindstone" => Kind::Grindstone,
            "minecraft:anvil" => Kind::Anvil,
            "minecraft:smithing" => Kind::Smithing,
            "minecraft:enchantment" => Kind::Enchanting,
            "minecraft:beacon" => Kind::Beacon,
            "minecraft:cartography_table" => Kind::Cartography,
            _ => return None,
        })
    }
    fn open_menu_name(&self, container: i32) -> String {
        match self.remote.as_ref().and_then(|r| r.menu.as_ref()) {
            Some((id, name)) if *id == container && container != 0 => name.clone(),
            _ => "inventory".into(),
        }
    }
    fn set_menu_slot(&mut self, container: i32, slot: i32, item: Option<ItemStack>) {
        let menu = self.open_menu_name(container);
        if menu == "minecraft:lectern" {
            if slot == 0 {
                self.remote_lectern_book(item);
            }
            return;
        }
        if container != 0 && menu == "inventory" {
            return;
        }
        let Some(internal) = Self::inventory_screen_slot(&menu, slot) else {
            return;
        };
        if internal == 200 {
            // ClientboundContainerSetSlotPacket for the open menu's slot 0: the
            // server's result of its own recipes (a world's data packs' too).
            if container == self.remote_menu().0
                && let Some(r) = self.remote.as_mut()
            {
                r.craft_result = item;
            }
            return;
        }
        let slot = self.slot_mut(internal);
        let changed = !super::hand::matches_for_swap(slot.as_ref(), item.as_ref());
        *slot = item;
        if internal < 36 {
            self.remote_hand_slot_set(internal, changed);
        } else if internal == 504 {
            self.remote_hand_slot_set(super::shield::OFFHAND_SLOT, changed);
        }
        if internal >= super::stations::INPUT && self.station.is_some() {
            self.remote_station_slot_changed(internal);
        }
    }

    /// `ClientPacketListener.handleUpdateAttributes` for a mob, each value from
    /// `AttributeInstance.calculateValue`: its maximum health (the hearts drawn
    /// over its mount), and its `minecraft:scale`, which sizes its box
    /// (`LivingEntity.getDimensions`) and model (`LivingEntityRenderState.scale`)
    /// as the local player's does: NPC plugins make their fake players bigger
    /// with it, datapacks scale any mob.
    fn remote_mob_attributes(&mut self, id: i32, values: &[play::AttributeSnapshot]) {
        let catalog = &self.motion.attributes;
        let value = |name: &str| {
            let attribute = catalog.id(name)?;
            let s = values
                .iter()
                .find(|s| s.attribute == attribute.registry_id())?;
            Some(
                catalog
                    .attribute(attribute)
                    .definition
                    .sanitize(snapshot_value(s)),
            )
        };
        let (max_health, scale) = (value("minecraft:max_health"), value("minecraft:scale"));
        let (speed, jump) = (
            value("minecraft:movement_speed"),
            value("minecraft:jump_strength"),
        );
        let Some(m) = self.mob_index.get_mut(&mut self.mobs, id as u32) else {
            return;
        };
        // A horse its rider's client moves goes at its own speed and jumps
        // its own height (`getSpeed`, `getCustomJump`).
        if horizon_client::mobs::equine::moved_by_rider(m.species()) {
            if let Some(speed) = speed {
                horizon_client::mobs::finalize::set_base(
                    m,
                    horizon_client::mobs::finalize::Attr::MovementSpeed,
                    speed,
                );
            }
            if let Some(jump) = jump {
                m.extra.vector[0] = jump;
            }
        }
        if let Some(max_health) = max_health {
            horizon_client::mobs::finalize::set_base(
                m,
                horizon_client::mobs::finalize::Attr::MaxHealth,
                max_health,
            );
        }
        if let Some(scale) = scale {
            m.extra.model_scale = scale as f32;
        }
    }

    pub(in crate::game) fn handle_packet(&mut self, packet: C) -> Result<()> {
        self.observe_client_packet(&packet);
        match packet {
            // ClientCommonPacketListenerImpl.handlePing runs on the game
            // thread, after all preceding packets (especially teleports).
            C::Ping(p) => self.send(S::Pong(p)),
            // A native mod's server half to its client half.
            C::CustomPayload(p) if self.is_book_or_mod_channel(&p.channel) => {
                self.remote_book_or_mod_payload(&p.channel, &p.data)?;
            }
            // The synced attachments of a holder the server shows the window.
            C::CustomPayload(p) if p.channel == horizon_protocol::attachment_sync::CHANNEL => {
                self.remote_attachment_sync(&p.data);
            }
            // A mod menu type's block and its mod's bytes, just before the
            // menu's `OpenScreen`.
            C::CustomPayload(p) if p.channel == horizon_protocol::menu_extra::CHANNEL => {
                match horizon_protocol::menu_extra::decode(&p.data) {
                    Ok(extra) => {
                        if let Some(r) = self.remote.as_mut() {
                            r.menu_extra = Some(extra);
                        }
                    }
                    Err(e) => log::debug!("Horizon: a bad menu extra payload: {e}"),
                }
            }
            C::CustomPayload(p) if p.channel == horizon_client::lan::recipes::TRANSFER_CHANNEL => {
                self.remote_recipe_transfer(&p.data);
            }
            // An anvil cost the data-slot short cannot hold, after that short.
            C::CustomPayload(p) if p.channel == horizon_protocol::anvil_cost::CHANNEL => {
                self.remote_anvil_wide_cost(&p.data);
            }
            // The server's recipes its native mods asked to send.
            C::CustomPayload(p) if p.channel == horizon_client::lan::recipes::CONTENT_CHANNEL => {
                self.remote_recipe_content(&p.data);
            }
            // PingDebugMonitor.onPongReceived.
            C::PongResponse(p) => {
                let delay = self.debug_millis() - p.time;
                self.debug_charts.ping.log(delay);
            }
            C::Login(p) => {
                // A client's parts, from the tags it plays with.
                if let Err(e) = self.resolve_parts() {
                    log::warn!("Horizon: {e}");
                }
                // Configuration's tags, over the block ids the parts just bound.
                self.install_remote_tags();
                self.refresh_render_visibility();
                self.test_highlights.clear();
                self.forget_synced_recipes();
                self.level.hardcore = p.hardcore;
                log::info!(
                    "Horizon: joined as entity {}, game mode {}",
                    p.player_id,
                    p.spawn.game_type
                );
                // handleLogin: startWaitingForNewLevel with Reason.OTHER, which
                // replaces ServerReconfigScreen.
                if let Some(r) = self.remote.as_mut() {
                    r.reconfiguring = None;
                    r.attachments.clear();
                    // A new ClientPacketListener owns a fresh clock manager.
                    r.clocks.clear();
                    r.clock_game_time = 0;
                    self.world.game_time = 0;
                }
                // handleLogin makes a new MultiPlayerGameMode (carried index 0) and a
                // new LocalPlayer (slot 0, no effects, riding nothing: the server
                // pairs a vehicle again with ClientboundSetPassengersPacket).
                self.selected = 0;
                self.potions.effects.clear();
                self.riding = None;
                if let Some(player) = self.player.as_mut() {
                    player.entity_id = p.player_id;
                }
                if let Some(r) = self.remote.as_mut() {
                    r.player_attributes.clear();
                    r.carried.set(Some(0));
                    // handleLogin makes a new ClientLevel: its predictions count from 0.
                    r.sequence = 0;
                    r.own_invisible = false;
                    r.own_on_fire = false;
                    r.online_mode = p.online_mode;
                    r.levels.clone_from(&p.levels);
                    r.enforces_secure_chat = p.enforces_secure_chat;
                    r.entity_id = p.player_id;
                    r.local_entity_silent = false;
                    // handleLogin: the new LocalPlayer is the camera.
                    r.camera = None;
                    r.logged_in = true;
                    r.view_distance = p.chunk_radius;
                    r.simulation_distance = p.simulation_distance;
                    r.show_death_screen = p.show_death_screen;
                    r.reduced_debug_info = p.reduced_debug_info;
                    r.do_limited_crafting = p.limited_crafting;
                }
                self.bind_local_player_residency();
                self.refresh_debug_renderers();
                // LocalPlayer.setReducedDebugInfo always rebuilds, also on a second login.
                self.debug_entries.invalidate();
                self.streaming.radius = self
                    .settings
                    .render_distance
                    .min(p.chunk_radius.saturating_add(1))
                    .max(2);

                // handleLogin: startWaitingForNewLevel(.., Reason.OTHER).
                self.enter_remote_dimension(
                    &p.spawn,
                    false,
                    horizon_client::level_loading::Reason::Other,
                );
                // No ClientInformation here: the configuration phase sent it,
                // and ClientPacketListener.broadcastClientInformation sends it
                // again only when an option changes it.
                self.remote_chat_login();
                if let Some(remote) = &self.remote {
                    self.native_client_player_logging_in(p.player_id, remote.profile);
                }
            }
            C::Respawn(p) => {
                // The new LocalPlayer copies reduced debug info through its setter.
                self.debug_entries.invalidate();
                let old_player = self.remote.as_ref().map(|r| (r.entity_id, r.profile));
                if let Some((_, uuid)) = old_player {
                    self.forget_entity_initialization(Holder::Player(uuid));
                }
                if let Some(r) = self.remote.as_ref() {
                    self.audio.remove_sound_entity(r.entity_id as u32);
                }
                // A new player, without effects (the server sends them
                // again) and with no item in use.
                self.potions.effects.clear();
                // The server sends the level, chunks and entities again.
                if let Some(r) = self.remote.as_mut() {
                    r.attachments
                        .retain(|holder, _| matches!(holder, Holder::Player(_)));
                }
                self.using = None;
                self.end_mod_use();
                // After a death (no KEEP_ENTITY_DATA) the new LocalPlayer
                // rides nothing: the server pairs a vehicle again with its
                // passengers. A dimension change keeps the vehicle it travels in.
                if p.keep & 2 == 0 {
                    self.riding = None;
                    // KEEP_ENTITY_DATA copies the old player's synched data
                    // (its shared flags) to the new one; without it they reset.
                    if let Some(r) = self.remote.as_mut() {
                        r.own_invisible = false;
                        r.own_on_fire = false;
                        r.local_entity_silent = false;
                    }
                }
                // handleRespawn compares the levels' keys: a mod's level is
                // its own dimension even where `Dim` has no name for it.
                let dimension_changed = self
                    .remote
                    .as_ref()
                    .is_some_and(|r| r.level != p.spawn.dimension);
                let died = self.screen == Screen::Dead
                    || self
                        .player
                        .as_ref()
                        .is_some_and(|m| m.health.health() <= 0.0);
                let reason = horizon_client::level_loading::reason_for_respawn(
                    died,
                    &p.spawn.dimension,
                    self.dim.id(),
                );
                if dimension_changed {
                    horizon_client::spans::startup::arm("dimension.change");
                }
                // handleRespawn makes a new ClientLevel for another dimension
                // key (a proxy's server switch too): its predictions count from 0.
                if let Some(r) = self.remote.as_mut()
                    && r.level != p.spawn.dimension
                {
                    r.sequence = 0;
                }
                self.enter_remote_dimension(&p.spawn, true, reason);
                // handleRespawn adds the new LocalPlayer after what the kept
                // level holds; a new level holds nothing yet.
                let before: std::collections::HashSet<u32> =
                    self.mobs.iter().map(|m| m.id).collect();
                if let Some(r) = self.remote.as_mut() {
                    r.ticked_before_player = before;
                }
                if dimension_changed {
                    // LevelExtractor.setLevel: a new level drops the highlights.
                    self.test_highlights.clear();
                    let (music, _) = self.situational_music();
                    self.music.stop_playing(music.as_ref(), &mut self.audio);
                }
                self.set_screen(Screen::Playing);
                if let Some(r) = self.remote.as_mut() {
                    r.death_message = None;
                    r.health = (20.0, 20, 5.0);
                    // handleRespawn: the new LocalPlayer is the camera.
                    r.camera = None;
                }
                self.capture(true);
                if let Some((old_id, uuid)) = old_player {
                    self.native_client_player_clone(
                        old_id,
                        self.own_entity_id(),
                        uuid,
                        died,
                        dimension_changed,
                    );
                }
            }
            C::PlayerPosition(p) => self.remote_teleport(&p)?,
            C::PlayerRotation(p) => {
                self.yaw = if p.relative_yaw {
                    self.yaw + p.yaw
                } else {
                    p.yaw
                };
                self.pitch = (if p.relative_pitch {
                    self.pitch + p.pitch
                } else {
                    p.pitch
                })
                .clamp(-90.0, 90.0);
                self.snap_player_model_pitch();
            }
            C::PlayerLookAt(p) => self.remote_look_at(&p),
            // ClientCommonPacketListenerImpl.handlePostEffects.
            C::PostEffects(p) => self.post_effects = p.effects,
            C::SetCamera(p) => self.remote_set_camera(p.camera_id),
            C::LevelChunkWithLight(p) => self.remote_chunk(&p, None),
            C::LightUpdate(p) => self.remote_light_update(&p),
            C::ChunksBiomes(p) => self.remote_chunks_biomes(&p),
            // `poll_remote` forgets a run of them at once.
            C::ForgetLevelChunk(p) => {
                self.forget_remote_columns(&mut [(p.pos[0], p.pos[1])].into_iter().collect(), true);
            }
            C::ChunkBatchFinished(_) => {
                // ChunkBatchSizeCalculator's ceiling; meshing runs on workers.
                self.send(S::ChunkBatchReceived(play::ChunkBatchReceived {
                    chunks_per_tick: 64.0,
                }));
            }
            C::BlockUpdate(p) => self.server_block(p.pos, p.state),
            C::AddTransientBlock(p) => self.add_transient_block(p.pos, p.state),
            // ClientPacketListener.handleBlockDestruction: another player's crack.
            C::BlockDestruction(p) => {
                self.destroy_block_progress(p.id, p.pos, i32::from(p.progress));
            }
            C::SectionBlocksUpdate(p) => {
                let [sx, sy, sz] = p.section;
                for change in p.changes {
                    let local = (change & 0xFFF) as i32;
                    let pos = [
                        sx * 16 + (local >> 8 & 15),
                        sy * 16 + (local & 15),
                        sz * 16 + (local >> 4 & 15),
                    ];
                    self.server_block(pos, (change >> 12) as i32);
                }
            }
            C::BlockChangedAck(p) => self.end_predictions(p.sequence),
            // Level.blockEvent on the client level: the block's triggerEvent
            // runs on the client's copy of whatever is at pos (a piston's
            // moveBlocks builds the moving block entities it animates, a note
            // block plays); what it doesn't handle reaches the client copy of
            // the block entity (chest lids, shulker boxes).
            C::BlockEvent(p) => {
                let pos = horizon_core::position::BlockPos::new(p.pos[0], p.pos[1], p.pos[2]);
                let triggered = self.with_level(|b, level, ticks| {
                    b.client_block_event(level, ticks, pos, i32::from(p.b0), i32::from(p.b1))
                });
                self.moving_pistons_stale = true;
                self.process_level_events()?;
                if !triggered {
                    let state = self.world.block(p.pos[0], p.pos[1], p.pos[2]);
                    let block = &self.res.facts().get(state).name;
                    let named = self.remote.as_ref().and_then(|r| {
                        r.builtin
                            .get("minecraft:block")?
                            .name(p.block)
                            .map(str::to_owned)
                    });
                    if named.as_deref() == Some(&**block) {
                        self.block_entity_event(p.pos, i32::from(p.b0), i32::from(p.b1));
                    }
                }
            }
            // ClientLevel.levelEvent: LevelEventHandler, as this client's own
            // level plays them (particles and sounds: a spawner's burst, bone
            // meal, waxing, a block broken by someone else).
            // ClientPacketListener.handleLevelEvent: a global one is heard
            // from anywhere (ClientLevel.globalLevelEvent).
            C::LevelEvent(p) if p.global => self.global_level_event_sound(p.kind, p.pos),
            C::LevelEvent(p) => self.level_event_at(p.kind, p.pos, p.data),
            C::LevelParticles(p) => self.remote_level_particles(&p),
            C::Explode(p) => self.remote_explode(&p),
            C::Sound(p) => {
                if let Some(name) = self.remote_sound_name(&p.sound)
                    && !self.settings.extras.pvp.combat.mutes_sound(&name)
                {
                    let at = [
                        f64::from(p.x) / 8.0,
                        f64::from(p.y) / 8.0,
                        f64::from(p.z) / 8.0,
                    ];
                    let mut sound = LevelSound::new(&name, at, p.volume, p.pitch);
                    if let Ok(source) = usize::try_from(p.source)
                        && let Some(source) = mod_api::SoundSource::ALL.get(source)
                    {
                        sound.source = *source;
                    }
                    self.play_client_level_sound(sound, p.seed);
                }
            }
            C::SoundEntity(p) => {
                if let Some(name) = self.remote_sound_name(&p.sound)
                    && !self.settings.extras.pvp.combat.mutes_sound(&name)
                {
                    let mut sound = LevelSound::new(&name, [0.0; 3], p.volume, p.pitch);
                    sound.emitter = Some(p.id as u32);
                    if let Ok(source) = usize::try_from(p.source)
                        && let Some(source) = mod_api::SoundSource::ALL.get(source)
                    {
                        sound.source = *source;
                    }
                    self.play_client_level_sound(sound, p.seed);
                }
            }
            C::StopSound(p) => self
                .audio
                .stop_sounds(p.sound.as_deref(), p.source.map(|s| s as usize)),
            C::AddEntity(p) => {
                self.audio.remove_sound_entity(p.id as u32);
                let kind = self
                    .remote
                    .as_ref()
                    .and_then(|r| {
                        r.builtin
                            .get("minecraft:entity_type")?
                            .name(p.kind)
                            .map(str::to_owned)
                    })
                    .unwrap_or_default();
                let rotation =
                    [p.yaw, p.pitch, p.head_yaw].map(horizon_protocol::buf::unpack_degrees);
                if let Some(r) = self.remote.as_mut() {
                    // ClientLevel.addEntity: an id it holds already is removed
                    // and added again, after the player.
                    r.ticked_before_player.remove(&(p.id as u32));
                    // A new entity looks its PlayerInfo up afresh.
                    r.level_player_textures.remove(&p.id);
                    if kind == "minecraft:player" {
                        r.level_players.insert(p.id, p.uuid);
                        // createEntityFromPacket: the player takes the profile of its list entry.
                        match r.players.get(&p.uuid) {
                            Some(name) => r.level_player_names.insert(p.id, name.clone()),
                            None => r.level_player_names.remove(&p.id),
                        };
                    } else {
                        r.level_players.remove(&p.id);
                        r.level_player_names.remove(&p.id);
                    }
                }
                // Entity.getTeam: a mob is on a team by its UUID, a player by
                // its profile name (`Game::scoreboard_name`).
                self.persist.uuids.insert(
                    p.id as u32,
                    horizon_client::lan::playerdata::uuid_ints(p.uuid),
                );
                let holder = if kind == "minecraft:player" {
                    Holder::Player(p.uuid)
                } else {
                    Holder::Entity(p.id as u32)
                };
                self.initialize_entity(holder, &kind);
                self.spawn_entity(p.id, &kind, p.position, rotation);
                if let Some(r) = self.remote.as_mut() {
                    r.cart_lerps.remove(&p.id);
                    r.lerps.remove(&p.id);
                }
                if !self.spawn_remote_object(
                    p.id,
                    &kind,
                    p.position,
                    (rotation[0], rotation[1]),
                    p.data,
                ) {
                    self.spawn_remote_projectile(&kind, &p, (rotation[0], rotation[1]));
                }
                // Entity.recreateFromPacket: objects must exist before applying motion.
                self.remote_motion(p.id, p.velocity);
                if self.minecart_improvements()
                    && self.carts.iter().any(|c| c.id == p.id as u32)
                    && let Some(r) = self.remote.as_mut()
                {
                    r.cart_lerps.insert(p.id, CartLerp::new());
                }
                self.move_remote_text_display(p.id, p.position, Some((rotation[0], rotation[1])));
                self.move_remote_interaction(p.id, p.position);
            }
            C::RemoveEntities(p) => {
                for &id in &p.ids {
                    self.forget_remote_entity_construction(id);
                    if let Some(r) = self.remote.as_mut() {
                        if let Some(uuid) = r.level_players.remove(&id) {
                            r.attachments.remove(&Holder::Player(uuid));
                        }
                        r.level_player_names.remove(&id);
                        r.level_player_textures.remove(&id);
                        r.attachments.remove(&Holder::Entity(id as u32));
                    }
                }
                self.remove_entities(&p.ids);
            }
            C::MoveEntityPos(play::MoveEntityPos(m))
            | C::MoveEntityPosRot(play::MoveEntityPosRot(m))
            | C::MoveEntityRot(play::MoveEntityRot(m)) => {
                // handleMoveEntity: the delta counts from the position codec's
                // base, which then moves to the delta's end.
                let base = self
                    .remote
                    .as_ref()
                    .and_then(|r| r.entities.get(&m.id))
                    .map(|t| t.codec_base);
                let Some(base) = base else { return Ok(()) };
                // VecDeltaCodec.decode: each step's position from the last.
                let path = m.delta.as_ref().map(|d| match d {
                    play::VecDelta::Linear(_) => Path::Linear(d.end(base)),
                    play::VecDelta::Stepped(steps) => {
                        let mut at = base;
                        Path::Stepped(
                            steps
                                .iter()
                                .map(|(delta, ticks)| {
                                    at = play::VecDelta::apply(at, *delta);
                                    (at, *ticks)
                                })
                                .collect(),
                        )
                    }
                });
                let rotation = m.rotation.map(|(y, x)| {
                    (
                        horizon_protocol::buf::unpack_degrees(y),
                        horizon_protocol::buf::unpack_degrees(x),
                    )
                });
                if let (Some(path), Some(t)) = (
                    &path,
                    self.remote.as_mut().and_then(|r| r.entities.get_mut(&m.id)),
                ) {
                    t.codec_base = path.end();
                }
                self.move_entity(m.id, path, rotation, m.on_ground, MovePacket::Move);
            }
            C::EntityPositionSync(p) => {
                let path = match p.path {
                    play::PositionPath::Linear(at) => Path::Linear(at),
                    play::PositionPath::Stepped(steps) => Path::Stepped(steps),
                };
                // handleEntityPositionSync: the position codec's base is set.
                if let Some(t) = self.remote.as_mut().and_then(|r| r.entities.get_mut(&p.id)) {
                    t.codec_base = path.end();
                }
                self.move_entity(
                    p.id,
                    Some(path),
                    Some((p.yaw, p.pitch)),
                    p.on_ground,
                    MovePacket::PositionSync,
                )
            }
            C::TeleportEntity(p) => {
                let base = self
                    .remote
                    .as_ref()
                    .and_then(|r| r.entities.get(&p.id))
                    .map_or([0.0; 3], |t| t.position);
                let rel = |bit: i32, i: usize| {
                    if p.relatives & bit != 0 {
                        base[i] + p.position[i]
                    } else {
                        p.position[i]
                    }
                };
                let position = [
                    rel(play::relative::X, 0),
                    rel(play::relative::Y, 1),
                    rel(play::relative::Z, 2),
                ];
                // setValuesFromPositionPacket: interpolated, unless too far.
                self.move_entity(
                    p.id,
                    Some(Path::Linear(position)),
                    Some((p.yaw, p.pitch)),
                    p.on_ground,
                    MovePacket::Teleport,
                );
                // handleTeleportEntity: the mount this client moves is put
                // there at once.
                if self.driven_mount().is_some_and(|m| m.id as i32 == p.id) {
                    self.remote_move_vehicle(position, p.yaw);
                }
            }
            C::SetPassengers(p) => self.remote_set_passengers(p.vehicle, &p.passengers),
            // ClientPacketListener.handleEntityLinkPacket: who holds a mob's
            // lead (this player, another player, a mob or a fence knot), for
            // the rope.
            C::SetEntityLink(p) => {
                let me = self.remote.as_ref().map(|r| r.entity_id);
                use horizon_client::mobs::leash::Holder;
                let holder = match p.dest {
                    0 => None,
                    d if Some(d) == me => Some(Holder::Player),
                    d => match self.remote_knots.get(&(d as u32)) {
                        Some(&block) => Some(Holder::Knot(block)),
                        // Not a known knot yet: kept by id until the knot
                        // arrives (game/network/remote_objects.rs).
                        None if self.boats.iter().any(|b| b.id == d as u32) => {
                            Some(Holder::Boat(d as u32))
                        }
                        None => Some(Holder::Mob(d as u32)),
                    },
                };
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.source as u32) {
                    m.extra.leash = holder;
                } else if let Some(i) = self.boats.iter().position(|b| b.id == p.source as u32) {
                    self.boats.update(i, |b| {
                        // A leashed boat or raft (AbstractBoat is Leashable).
                        b.leash = holder;
                    });
                }
            }
            C::MoveVehicle(p) => self.remote_move_vehicle(p.position, p.yaw),
            C::MoveMinecartAlongTrack(p) => self.remote_move_minecart(p),
            // ClientPacketListener.handleRotateMob: lerpHeadTo(yHeadRot, 3).
            C::RotateHead(p) => {
                if let Some(r) = self.remote.as_mut() {
                    let yaw = horizon_protocol::buf::unpack_degrees(p.head_yaw);
                    r.head_lerps.insert(p.id, (yaw, 3));
                }
            }
            C::SetEntityData(p) => {
                // Any entity's custom name (armor stand holograms, objects).
                self.remote_names_and_displays(p.id, &p.values);
                if let Some(r) = self.remote.as_mut() {
                    for (index, _, value) in &p.values {
                        if let (4, Value::Bool(silent)) = (index, value) {
                            if p.id == r.entity_id {
                                r.local_entity_silent = *silent;
                            }
                            if let Some(t) = r.entities.get_mut(&p.id) {
                                t.silent = *silent;
                            }
                        }
                    }
                }
                // Entity.DATA_SHARED_FLAGS_ID, DATA_CUSTOM_NAME and
                // DATA_CUSTOM_NAME_VISIBLE: server NPCs and holograms rely on
                // these instead of saved NBT.
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    for (index, _, value) in &p.values {
                        match (index, value) {
                            (0, Value::Byte(flags)) => {
                                m.invisible = flags & 0x20 != 0;
                                m.glowing = flags & 0x40 != 0;
                            }
                            (2, Value::OptText(name)) => {
                                let text = name.as_ref().map(horizon_client::entity_nbt::name_text);
                                m.custom_name =
                                    text.as_ref().map(horizon_client::text::Text::plain);
                                m.custom_name_text = text;
                            }
                            (2, Value::Text(name)) => {
                                let text = horizon_client::entity_nbt::name_text(name);
                                m.custom_name = Some(text.plain());
                                m.custom_name_text = Some(text);
                            }
                            (3, Value::Bool(visible)) => m.name_visible = *visible,
                            _ => {}
                        }
                    }
                }
                if let Some((_, _, Value::Byte(flags))) = p.values.iter().find(|v| v.0 == 0)
                    && let Some(r) = self.remote.as_mut()
                {
                    if flags & 0x40 != 0 {
                        r.glowing.insert(p.id);
                    } else {
                        r.glowing.remove(&p.id);
                    }
                }
                self.remote_mob_data(p.id, &p.values);
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                    && m.species() == horizon_client::mobs::Species::Creeper
                {
                    for (index, _, value) in &p.values {
                        match (index, value) {
                            // Creeper.DATA_SWELL_DIR, DATA_IS_POWERED, DATA_IS_IGNITED.
                            (16, Value::Int(dir)) => m.extra.counter = *dir,
                            (17, Value::Bool(powered)) => m.extra.charged = *powered,
                            (18, Value::Bool(ignited)) => m.extra.flag = *ignited,
                            _ => {}
                        }
                    }
                }
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    for (index, _, value) in &p.values {
                        match (index, value) {
                            // Entity.DATA_TICKS_FROZEN: isFullyFrozen shakes it.
                            (7, Value::Int(ticks)) => m.ticks_frozen = *ticks,
                            // Strider.DATA_SUFFOCATING (after AgeableMob's two
                            // and DATA_BOOST_TIME): a cold strider shakes.
                            (19, Value::Bool(cold))
                                if m.species() == horizon_client::mobs::Species::Strider =>
                            {
                                m.extra.flag = *cold;
                            }
                            _ => {}
                        }
                    }
                }
                // LivingEntity.DATA_LIVING_ENTITY_FLAGS: for another player
                // too, remote_mob_data's apply reads index 8 above.
                let mannequin = self
                    .mobs
                    .iter()
                    .any(|m| m.id == p.id as u32 && &*m.kind.id == "minecraft:mannequin");
                let other = mannequin
                    || self.remote.as_ref().is_some_and(|r| {
                        r.level_players.contains_key(&p.id) && r.entity_id != p.id
                    });
                // LivingEntity.onSyncedDataUpdated(SLEEPING_POS_ID): another
                // player lies down in their bed (drawn lying in it), or gets up.
                if other {
                    for (index, _, value) in &p.values {
                        if let (14, Value::OptPos(bed)) = (index, value) {
                            let code = bed.map_or(0, |b| self.bed_facing(b));
                            let at = bed.and_then(|head| self.sleeping_position(head));
                            if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                                m.extra.ints[1] = code;
                                if bed.is_some() {
                                    // setPosToBed leaves the server's position alone
                                    // when the client's bed has no sleep shape.
                                    if let Some(at) = at {
                                        m.position = at;
                                        m.previous = at;
                                    }
                                    m.head_pitch = 0.0;
                                    m.previous_head_pitch = 0.0;
                                }
                            }
                        }
                        // Avatar.DATA_PLAYER_MAIN_HAND (HumanoidArm: 0 left, 1
                        // right): the arm a swing of the main hand moves.
                        if let (15, Value::Id(arm)) = (index, value)
                            && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                        {
                            m.extra.main_hand_left = *arm == 0;
                        }
                        // Avatar.DATA_PLAYER_MODE_CUSTOMISATION: the skin layers
                        // the player shows (PlayerModelPart bits).
                        if let (16, Value::Byte(bits)) = (index, value)
                            && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                        {
                            m.extra.hidden_model_parts = !(*bits as u8) & 0x7F;
                        }
                    }
                }
                // LivingEntity.onSyncedDataUpdated(SLEEPING_POS_ID): this
                // player lies down in the bed, or gets up.
                if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) {
                    for (index, _, value) in &p.values {
                        match (index, value) {
                            (14, Value::OptPos(bed)) => {
                                if bed.is_some() && self.sleeping.is_none() {
                                    self.sleep_timer = 0;
                                }
                                self.sleeping = *bed;
                                if let Some(head) = *bed {
                                    // LivingEntity.onSyncedDataUpdated(SLEEPING_POS_ID): setPosToBed.
                                    let at = self.sleeping_position(head);
                                    if let Some(pl) = self.player.as_mut() {
                                        if let Some(at) = at {
                                            pl.position = at;
                                        }
                                        pl.velocity = Vec3::ZERO;
                                        pl.pitch = 0.0;
                                    }
                                    if let Some(at) = at {
                                        self.previous_position = at;
                                    }
                                    self.pitch = 0.0;
                                    self.snap_player_model_pitch();
                                }
                            }
                            // Entity.DATA_SHARED_FLAGS_ID's sprinting bit: the
                            // server stops the sprint after a hit's extra
                            // knockback (the flag only, as assignValues sets
                            // it: the speed modifier and wasSprinting stay).
                            // Its invisible bit (Entity.isInvisible): the
                            // Invisibility effect's, or a plugin's that hides
                            // the real body under a pose (#3788).
                            (0, Value::Byte(flags)) => {
                                if let Some(pl) = self.player.as_mut() {
                                    pl.sprinting = flags & 0x08 != 0;
                                    // Entity.DATA_SHARED_FLAGS_ID's fall-flying
                                    // bit (0x80, `Entity.setSharedFlag(7, ...)`):
                                    // `LivingEntity.updateFallFlying` clears the
                                    // glide on the server only and `LocalPlayer`
                                    // never ends one itself, so this packet is
                                    // the glide's own end. Applied when it is
                                    // handled â€” the frame's poll, before its
                                    // ticks â€” as ClientPacketListener does, so a
                                    // clear that arrives mid-tick lets that tick
                                    // keep its glide travel (#7616).
                                    pl.fall_flying = flags & -128 != 0;
                                }
                                let on_fire = flags & 0x01 != 0;
                                if let Some(r) = self.remote.as_mut() {
                                    r.own_invisible = flags & 0x20 != 0;
                                    let was_on_fire =
                                        std::mem::replace(&mut r.own_on_fire, on_fire);
                                    // The server put the fire out: the
                                    // window's countdown ends with it.
                                    if was_on_fire
                                        && !on_fire
                                        && let Some(pl) = self.player.as_mut()
                                    {
                                        pl.fire_ticks = pl.fire_ticks.min(0);
                                    }
                                }
                            }
                            // Entity.DATA_AIR_SUPPLY_ID and
                            // Player.DATA_PLAYER_ABSORPTION_ID: the bubbles
                            // and golden hearts the server's player has.
                            (1, Value::Int(air)) => {
                                if let Some(pl) = self.player.as_mut() {
                                    pl.air = *air;
                                }
                            }
                            // Entity.DATA_TICKS_FROZEN: the server's count of the
                            // powder snow freeze, which the client's own copy reads
                            // for the frost overlay, the frozen hearts and the
                            // shiver of its third-person model (#11217).
                            (7, Value::Int(ticks)) => {
                                if let Some(pl) = self.player.as_mut() {
                                    pl.ticks_frozen = *ticks;
                                }
                            }
                            // LivingEntity.DATA_ARROW_COUNT_ID and DATA_STINGER_COUNT_ID.
                            (12, Value::Int(count)) => {
                                if let Some(r) = self.remote.as_mut() {
                                    r.own_arrows = *count;
                                }
                            }
                            (13, Value::Int(count)) => {
                                if let Some(r) = self.remote.as_mut() {
                                    r.own_stingers = *count;
                                }
                            }
                            (19 | 20, Value::Id(v)) => {
                                if let Some(r) = self.remote.as_mut() {
                                    r.own_shoulders[usize::from(*index == 20)] =
                                        (*v > 0).then(|| (*v - 1).clamp(0, 4));
                                }
                            }
                            (17, Value::Float(amount)) => {
                                if let Some(pl) = self.player.as_mut() {
                                    let _ = pl.internal_set_absorption(*amount);
                                }
                            }
                            // LivingEntity.DATA_LIVING_ENTITY_FLAGS: the server
                            // starts or stops this player's use, as an axe that
                            // disables the raised shield does (Player.disableShield's
                            // stopUsingItem). LocalPlayer.onSyncedDataUpdated.
                            (8, Value::Byte(flags)) => {
                                let selected = self.selected;
                                let using = self.using;
                                self.using = synced_local_use(using, *flags, selected, |slot| {
                                    self.stack_in(slot).is_some()
                                });
                                // onSyncedDataUpdated's useItemRemaining: getUseDuration,
                                // without startUsingItem's event.
                                if using.is_none()
                                    && let Some((slot, _)) = self.using
                                    && let Some(stack) = self.stack_in(slot).cloned()
                                {
                                    self.use_duration = self.use_duration(&stack);
                                    self.use_ticks_set = false;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Entity.DATA_SHARED_FLAGS_ID's crouching and sprinting bits:
                // another player's or a mannequin's crouch (their model and
                // Entity.isDiscrete's name tag), and the dust a runner raises.
                if (mannequin
                    || self
                        .remote
                        .as_ref()
                        .is_some_and(|r| r.level_players.contains_key(&p.id)))
                    && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                {
                    for (index, _, value) in &p.values {
                        if let (0, Value::Byte(flags)) = (index, value) {
                            m.extra.crouching = flags & 2 != 0;
                            m.extra.sprinting = flags & 8 != 0;
                        }
                    }
                }
                for (index, _, value) in &p.values {
                    if *index == ITEM_ENTITY_ITEM
                        && let Value::Item(stack) = value
                    {
                        self.remote_item_data(p.id, stack);
                    }
                    // ExperienceOrb.DATA_VALUE (the icon's size).
                    if *index == 8
                        && let Value::Int(v) = value
                        && let Some(o) = self.orbs.iter_mut().find(|o| o.id == p.id as u32)
                    {
                        o.value = *v;
                    }
                }
                self.remote_object_data(p.id, &p.values);
                self.remote_projectile_data(p.id, &p.values);
                self.remote_dragon_data(p.id, &p.values);
            }
            // ClientPacketListener.handleSetEquipment: what a mob or another
            // player holds and wears (EquipmentSlot ordinals).
            C::SetEquipment(p) => {
                self.remote_object_equipment(p.id, &p.slots);
                let items: Vec<(u8, Option<Arc<str>>, Option<ItemStack>)> = p
                    .slots
                    .iter()
                    .map(|(slot, stack)| {
                        (
                            *slot,
                            stack.as_ref().map(|s| self.item_name(s.item)),
                            self.to_stack(stack),
                        )
                    })
                    .collect();
                // CrossbowItem.isCharged: the main hand's `charged_projectiles`
                // (AvatarRenderer.getArmPose holds a loaded crossbow up).
                let charged = p.slots.iter().find(|(slot, _)| *slot == 0).map(|(_, s)| {
                    s.as_ref().is_some_and(|s| {
                        matches!(
                            s.component(horizon_protocol::item::component::CHARGED_PROJECTILES),
                            Some(horizon_protocol::item::Component::Stacks(list))
                                if !list.is_empty()
                        )
                    })
                });
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    if let Some(charged) = charged
                        && &*m.kind.id == "minecraft:player"
                    {
                        m.extra.charged = charged;
                    }
                    for (slot, item, stack) in items {
                        let stack = stack.or_else(|| item.map(|item| ItemStack::new(&item, 1)));
                        m.set_equipment_stack(slot as usize, stack);
                    }
                }
            }
            C::TakeItemEntity(p) => {
                // ClientPacketListener.handleTakeItemEntity: the pickup sound at
                // the orb, item or arrow, whoever collects it (the server sends
                // none); an entity this client doesn't track sounds at this player.
                let orb_at = self
                    .orbs
                    .iter()
                    .find(|o| o.id == p.item as u32)
                    .map(|o| o.position);
                let orb = orb_at.is_some();
                let item = self.dropped.iter().find(|d| d.id == p.item as u32);
                let shot = self
                    .creatures
                    .projectiles
                    .iter()
                    .find(|s| s.id == p.item as u32);
                let shot = shot.map(|s| s.position).or_else(|| {
                    self.tridents
                        .iter()
                        .find(|t| t.id == p.item as u32)
                        .map(|t| t.position)
                });
                let at = match (item, shot) {
                    (Some(d), _) => Some([d.position.x, d.position.y, d.position.z]),
                    (None, Some(s)) => Some([s.x, s.y, s.z]),
                    (None, None)
                        if self
                            .remote
                            .as_ref()
                            .is_some_and(|r| r.entity_id == p.player) =>
                    {
                        self.eye_position()
                    }
                    (None, None) => None,
                };
                if let Some(o) = orb_at {
                    let pitch = (self.world.random.next_float() - self.world.random.next_float())
                        * 0.35
                        + 0.9;
                    self.audio.play(
                        "minecraft:entity.experience_orb.pickup",
                        Some([o.x, o.y, o.z]),
                        0.1,
                        pitch,
                    );
                } else if let Some(at) = at {
                    let pitch = ((self.world.random.next_float() - self.world.random.next_float())
                        * 0.7
                        + 1.0)
                        * 2.0;
                    self.audio
                        .play("minecraft:entity.item.pickup", Some(at), 0.2, pitch);
                }
                // ItemPickupParticle toward the collector (this player, another one or a mob).
                let collector = if self
                    .remote
                    .as_ref()
                    .is_some_and(|r| r.entity_id == p.player)
                {
                    super::pickups::Collector::Player
                } else {
                    super::pickups::Collector::Mob(p.player as u32)
                };
                if let Some(item) = self.dropped.iter().find(|d| d.id == p.item as u32).cloned() {
                    self.add_pickup(super::pickups::Taken::Item(Box::new(item)), collector);
                } else if let Some(orb) = self.orbs.iter().find(|o| o.id == p.item as u32).cloned()
                {
                    self.add_pickup(super::pickups::Taken::Orb(Box::new(orb)), collector);
                } else if let Some(shot) = self
                    .creatures
                    .projectiles
                    .iter()
                    .find(|s| s.id == p.item as u32)
                    .cloned()
                {
                    // An arrow flies to its collector too, then goes.
                    self.add_pickup(super::pickups::Taken::Arrow(Box::new(shot)), collector);
                } else if let Some(i) = self.tridents.iter().position(|t| t.id == p.item as u32) {
                    // So does a trident: a server's client keeps them apart.
                    let trident = self.tridents.remove(i);
                    self.add_pickup(super::pickups::Taken::Trident(Box::new(trident)), collector);
                }
                // An orb stays: the server may still hold merged ones in it.
                if !orb {
                    self.remove_entity(p.item);
                }
            }
            // ClientPacketListener.handleSetEntityMotion: knockback on this player.
            C::SetEntityMotion(p) if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) => {
                if let Some(pl) = self.player.as_mut() {
                    pl.velocity = Vec3::new(p.velocity[0], p.velocity[1], p.velocity[2]);
                }
            }
            // The same packet for another entity: Entity.lerpMotion.
            C::SetEntityMotion(p) => self.remote_motion(p.id, p.velocity),
            // ClientPacketListener.handleProjectilePowerPacket: the power a
            // hurting projectile accelerates by, sent bundled with its motion.
            C::ProjectilePower(p) => {
                super::remote_projectiles::remote_projectile_power(
                    &mut self.creatures.projectiles,
                    p.id,
                    p.power,
                );
            }
            // ClientPacketListener.handleDamageEvent: LivingEntity.handleDamageEvent
            // jolts the walk animation and restarts the hurt flash, and the
            // hurt player's own client plays its hurt sound (`Player.getHurtSound`:
            // the damage type's effects) at `getSoundVolume` and
            // `Player.getVoicePitch`, which is 1.0 for every player. The
            // server's sound packet reaches every other player around, never
            // this one (#4149), and that copy is played at 1.0.
            // This player's own hurt tilt and flash come from the server too
            // (sent to the player itself): its own moves never hurt it.
            C::DamageEvent(p) => {
                let me = self.remote.as_ref().map(|r| r.entity_id);
                let own = me == Some(p.id);
                self.pvp_hurt(p.id, own, p.direct.map(|d| Some(d) == me));
                if own {
                    // For `events`: this client has no combat tracker of its own here.
                    let cause = self.remote.as_ref().and_then(|r| {
                        r.registries
                            .name("minecraft:damage_type", p.source_type)
                            .map(str::to_owned)
                    });
                    let by = p
                        .cause
                        .and_then(|id| u32::try_from(id).ok())
                        .and_then(|id| {
                            self.mob_index
                                .get(&self.mobs, id)
                                .map(|m| m.kind.id.to_string())
                        });
                    self.events
                        .note_hit(cause.unwrap_or_else(|| "unknown".into()), by);
                    if let Some(pl) = self.player.as_mut() {
                        pl.health.animate_hurt();
                        let sound = self.remote_hurt_sound(p.source_type);
                        let pitch = player_voice_pitch();
                        self.play_client_player_sound(sound, 1.0, pitch);
                    }
                } else if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    m.limb_speed = 1.5;
                    m.hurt_time = 10;
                } else {
                    self.remote_dragon_hurt(p.id);
                }
            }
            C::HurtAnimation(p) => {
                if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) {
                    // Player.animateHurt: the tilt turns around the yaw.
                    if let Some(pl) = self.player.as_mut() {
                        pl.health.animate_hurt_dir(p.yaw);
                    }
                } else if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    m.hurt_time = 10;
                } else {
                    self.remote_dragon_hurt(p.id);
                }
            }
            C::EntityEvent(p) => {
                if p.event == super::fireworks::FIREWORK_EXPLODE_EVENT {
                    self.remote_firework_explosion(p.id);
                }
                self.remote_fangs_event(p.id as u32, p.event);
                self.remote_boat_event(p.id, p.event);
                // MinecartTNT.handleEntityEvent 70: prime the fuse the window draws.
                self.remote_tnt_cart_event(p.id, p.event);
                // Arrow.handleEntityEvent 0: a tipped arrow's burst of particles.
                if p.event == 0 {
                    self.remote_arrow_puff(p.id as u32);
                }
                // LivingEntity.handleEntityEvent 2: a spear's hit felt.
                if p.event == 2 {
                    self.remote_kinetic_hit(p.id);
                }
                // LocalPlayer.setPermissions also refreshes an open World Options screen.
                if (24..=28).contains(&p.event)
                    && self.remote.as_ref().is_some_and(|r| r.entity_id == p.id)
                {
                    self.ui.registries.permission = (p.event - 24) as u8;
                    self.remote_world_options_permission(p.event >= 26);
                }
                // Player.handleEntityEvent 22 / 23: reduced debug info on / off.
                if (22..=23).contains(&p.event)
                    && let Some(r) = self.remote.as_mut().filter(|r| r.entity_id == p.id)
                {
                    r.reduced_debug_info = p.event == 22;
                    self.debug_entries.invalidate();
                }
                // Ravager.handleEntityEvent 69: the roar's puff and push.
                if p.event == 69 {
                    self.remote_ravager_roar(p.id);
                }
                // ClientPacketListener.handleEntityEvent 63: a sniffer's digging sound.
                if p.event == 63 {
                    self.start_sniffer_digging_sound(p.id as u32);
                }
                // ClientPacketListener.handleEntityEvent 21: a guardian's hum.
                if p.event == 21 {
                    self.start_guardian_sound(p.id as u32);
                }
                // ZombieVillager.handleEntityEvent 16: the cure starts.
                if p.event == 16 {
                    self.zombie_villager_cure_sound(p.id as u32);
                }
                // Witch.handleEntityEvent 15: the magic puffs over its hat.
                if p.event == 15 {
                    self.witch_magic_particles(p.id as u32);
                }
                // MinecartTNT.handleEntityEvent 70: primeFuse on the client.
                if p.event == 70 {
                    self.remote_cart_primed(p.id as u32);
                }
                // Fox.handleEntityEvent 45: the crumbs of the food in its mouth.
                if p.event == 45 {
                    self.fox_eat_particles(p.id as u32);
                }
                // Rabbit.handleEntityEvent(1): the hop clocks, and
                // Entity.spawnSprintParticle (one block-dust crumb).
                if p.event == 1 {
                    let hop = {
                        let m = self.mob_index.get_mut(&mut self.mobs, p.id as u32);
                        if let Some(m) = m
                            && m.species() == horizon_client::mobs::Species::Rabbit
                        {
                            let sample =
                                (m.position, m.velocity, f64::from(m.width()), m.supporting);
                            horizon_client::mobs::rabbit::hop_event(m);
                            Some(sample)
                        } else {
                            None
                        }
                    };
                    if let Some((position, movement, width, support)) = hop {
                        let id = p.id as u32;
                        self.sprint_particle(position, movement, support, width, |game| {
                            // this.random: the rabbit's own, x then z.
                            game.mob_index
                                .get_mut(&mut game.mobs, id)
                                .map(|m| (m.random.next_double(), m.random.next_double()))
                        });
                    }
                }
                if p.event == 19
                    && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                {
                    horizon_client::mobs::aquatic::squid_cycle_event(m);
                }
                // LivingEntity.handleEntityEvent 55: another player swapped
                // its main and off hand (`swapHandItems`).
                if p.event == 55
                    && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                {
                    m.extra.swap_hands();
                }
                // LivingEntity.handleEntityEvent 47 to 52, 65 and 68: an
                // equipped item broke (onEquippedItemBroken).
                if let Some(slot) = horizon_client::mobs::item_break::equipment_break_slot(p.event)
                {
                    self.remote_equipment_break(p.id, slot);
                }
                // EntityEvent.DEATH: start the death animation.
                if p.event == 3
                    && let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32)
                {
                    m.health = 0.0;
                }
                // LivingEntity.handleEntityEvent 3 on this player: its death
                // sound, which Player.playSound plays for this client alone
                // (the server's goes to everyone else).
                if p.event == 3
                    && self.remote.as_ref().is_some_and(|r| r.entity_id == p.id)
                    && self.player.is_some()
                {
                    let sound = "minecraft:entity.player.death";
                    let pitch = player_voice_pitch();
                    self.play_client_player_sound(sound, 1.0, pitch);
                }
                // The events that start a mob's animation (the warden's).
                if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    horizon_client::mobs::sync::entity_event(m, p.event as u8);
                }
                // LivingEntity.handleEntityEvent 46: portal particles from the
                // old position to the new one (chorus fruit, an enderman).
                if p.event == 46 {
                    self.entity_teleport_particles(p.id);
                }
                // LivingEntity.handleEntityEvent 60: the poof as the dead body goes.
                if p.event == 60
                    && let Some(m) = self.mob_index.get(&self.mobs, p.id as u32)
                {
                    let (at, w, h) = (m.position, f64::from(m.width()), f64::from(m.height()));
                    self.death_poof([at.x, at.y, at.z], w, h);
                }
                if p.event == super::totem::TOTEM_EVENT {
                    self.remote_totem(p.id);
                }
                // Mob.handleEntityEvent 20: the spawn poof.
                if p.event == 20 && self.mobs.iter().any(|m| m.id == p.id as u32) {
                    self.poof_particles(p.id as u32);
                }
                // The particle bursts of Villager 12 to 14 and 42, TamableAnimal
                // and AbstractHorse 6 and 7, Animal and Allay 18, Ocelot 40 and
                // 41, Dolphin 38: the table in `entity_event_burst` is the one
                // list of them; any other event draws nothing here.
                self.entity_event_particles(p.id as u32, p.event);
                // Snowball and ThrownEgg.handleEntityEvent 3: the crumbs where
                // the projectile is (the server removes it right after).
                if p.event == 3 {
                    let crumbs = self
                        .creatures
                        .projectiles
                        .iter()
                        .find(|pr| pr.id == p.id as u32)
                        .and_then(|pr| {
                            let item = match pr.kind {
                                horizon_client::projectiles::ProjectileKind::Snowball => {
                                    "minecraft:snowball"
                                }
                                horizon_client::projectiles::ProjectileKind::Egg => "minecraft:egg",
                                horizon_client::projectiles::ProjectileKind::Thrown(t) => t.item,
                                _ => return None,
                            };
                            Some(([pr.position.x, pr.position.y, pr.position.z], item))
                        });
                    if let Some((at, item)) = crumbs {
                        self.item_burst(at, item);
                    }
                }
                if p.event == super::combat::DROWN_EVENT {
                    self.remote_drown(p.id);
                }
                // Entity events 53 and 54: honey crumbs while sliding, and the
                // ten from a landing (`HoneyBlock.showSlideParticles` /
                // `showJumpParticles`).
                if horizon_client::landing::honey_particle_count(p.event).is_some() {
                    let at = if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) {
                        self.player
                            .as_ref()
                            .map(|q| [q.position.x, q.position.y, q.position.z])
                    } else {
                        self.mobs
                            .iter()
                            .find(|m| m.id == p.id as u32)
                            .map(|m| [m.position.x, m.position.y, m.position.z])
                    };
                    if let Some(at) = at {
                        self.honey_entity_particles(at, p.event);
                    }
                }
                if p.event == super::consume::USE_ITEM_COMPLETE_EVENT
                    && self.remote.as_ref().is_some_and(|r| r.entity_id == p.id)
                {
                    self.remote_complete_using();
                }
            }
            C::Animate(p) => self.remote_animate(p.id, p.action),
            // ClientPacketListener.handleSwingAnimation: the entity swings
            // (LivingEntity.swing, which a swing in its first half ignores).
            C::SwingAnimation(p) => {
                let hand = if p.hand == 1 { Hand::Off } else { Hand::Main };
                let animation = horizon_client::first_person::SwingAnimation {
                    kind: p.kind.clamp(0, 255) as u8,
                    duration: p.duration,
                };
                if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) {
                    self.swing_hand(hand, animation);
                } else if let Some(m) = self.mob_index.get_mut(&mut self.mobs, p.id as u32) {
                    m.swing_hand_modified(hand == Hand::Off, animation);
                }
            }
            // The player's own instances: its maximum health (the hearts the
            // HUD draws and the health it may hold), reach, speed and so on.
            // Then the received health again, which the old maximum clamped.
            C::UpdateAttributes(p) if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id) => {
                // Kept for a LocalPlayer made after the packet came, and across respawns.
                if let Some(r) = self.remote.as_mut() {
                    for snapshot in &p.values {
                        r.player_attributes
                            .retain(|s| s.attribute != snapshot.attribute);
                        r.player_attributes.push(snapshot.clone());
                    }
                }
                if let Some(pl) = self.player.as_mut() {
                    horizon_client::lan::attributes::apply(&mut pl.attributes, &p.values);
                }
                self.apply_remote_health();
            }
            C::UpdateAttributes(p) => self.remote_mob_attributes(p.id, &p.values),
            C::UpdateMobEffect(p) => self.remote_effect(&p),
            C::RemoveMobEffect(p) => {
                if self.remote.as_ref().is_some_and(|r| r.entity_id == p.id)
                    && let Some(id) = self.remote_effect_id(p.effect)
                {
                    self.potions.effects.list.retain(|e| e.id != id);
                }
            }
            C::ChangeDifficulty(p) => {
                self.ui.difficulty = p.difficulty.clamp(0, 3) as u8;
                self.level.difficulty_locked = p.locked;
                self.remote_world_options_difficulty(self.ui.difficulty, p.locked);
            }
            C::GameRuleValues(p) => self.remote_world_options_rules(p.entries),
            // ClientPacketListener.handleSetHealth: LocalPlayer.hurtTo.
            C::SetHealth(p) => {
                let flash = self.remote.as_mut().is_some_and(|r| {
                    r.health = (p.health, p.food, p.saturation);
                    std::mem::replace(&mut r.flash_on_set_health, true)
                });
                if flash && let Some(pl) = self.player.as_mut() {
                    pl.health.flash_health_change(p.health);
                }
                self.apply_remote_health();
                // The death screen opens (or the respawn is asked for) with the
                // death packet when the rule hides it.
                let shown = self.remote.as_ref().is_none_or(|r| r.show_death_screen);
                if p.health <= 0.0 && shown && self.screen != Screen::Dead && self.player.is_some()
                {
                    self.close_dialogs();
                    self.set_screen(Screen::Dead);
                    self.ui.death_time = 0;
                    self.ui.death_confirm = None;
                    self.capture(false);
                }
            }
            C::PlayerCombatKill(p) => {
                // The component resolves in the client's current language, as
                // the chat line does, not in the language the session joined
                // with (`Remote.lang` is the vanilla server-language table).
                let message = remote_text(&p.message);
                let me = self.remote.as_ref().is_some_and(|r| r.entity_id == p.id);
                if let Some(r) = self.remote.as_mut() {
                    r.death_message = Some(message);
                }
                // ClientPacketListener.handlePlayerCombatKill: the death
                // screen, whatever health was last sent, or `respawn()` at
                // once when it is hidden.
                if me && self.screen != Screen::Dead && self.player.is_some() {
                    if self.remote.as_ref().is_none_or(|r| r.show_death_screen) {
                        self.close_dialogs();
                        self.set_screen(Screen::Dead);
                        self.ui.death_time = 0;
                        self.ui.death_confirm = None;
                        self.capture(false);
                    } else {
                        self.respawn_player()?;
                    }
                }
            }
            C::TrackedWaypoint(p) => self.handle_waypoint(&p),
            C::SetExperience(p) => {
                self.experience = Experience {
                    level: p.level,
                    progress: p.progress,
                    total: p.total,
                }
            }
            // ClientPacketListener.handleSetTime / ClientClockManager.handleUpdates:
            // advance all clocks to the packet's game time before applying its
            // (possibly partial) updates. The sky still uses only the default clock.
            C::SetTime(p) => {
                self.world.game_time = p.game_time;
                self.remote_clock_tick();
                if let Some(r) = self.remote.as_mut() {
                    for clock in p.clocks {
                        if r.registries
                            .name("minecraft:world_clock", clock.clock)
                            .is_some()
                        {
                            r.clocks.insert(clock.clock, clock);
                        }
                    }
                    self.world.day_time = r.clock_time(r.default_clock);
                }
            }
            C::GameEvent(p) => match p.event {
                // NO_RESPAWN_BLOCK_AVAILABLE: ClientPacketListener.handleGameEvent
                // sends the player this system message.
                0 => self.system_message(horizon_client::text::Text::translate(
                    "block.minecraft.spawn.not_valid",
                    &[],
                )),
                // LEVEL_CHUNKS_LOAD_START: the server has begun sending the level.
                13 => {
                    if let Some(r) = self.remote.as_mut() {
                        r.level_load.loading_packets_received();
                    }
                }
                // GUARDIAN_ELDER_EFFECT: the apparition, and the curse sound
                // unless the elder guardian is silent.
                10 => {
                    self.particles.elder_guardian();
                    if p.param == 1.0
                        && let Some(at) = self.player.as_ref().map(|m| m.position)
                    {
                        self.play_client_local_sound(
                            LevelSound::new(
                                "minecraft:entity.elder_guardian.curse",
                                [at.x, at.y, at.z],
                                1.0,
                                1.0,
                            )
                            .in_source(mod_api::SoundSource::Hostile),
                        );
                    }
                }
                // WIN_GAME: the poem and credits, then PERFORM_RESPAWN.
                4 if self.screen != Screen::Win => {
                    self.open_win_screen(true, super::win::Finish::RemoteRespawn);
                }
                1 => {
                    self.world.weather.raining = true;
                    self.world.weather.rain_level = (0.0, 0.0);
                }
                2 => {
                    self.world.weather.raining = false;
                    self.world.weather.rain_level = (1.0, 1.0);
                }
                3 => {
                    let mode = p.param as i32;
                    self.set_mode_id(mode);
                    if let Some(r) = self.remote.as_mut() {
                        r.game_mode = mode;
                    }
                }
                // IMMEDIATE_RESPAWN: LocalPlayer.setShowDeathScreen.
                11 => {
                    if let Some(r) = self.remote.as_mut() {
                        r.show_death_screen = p.param == 0.0;
                    }
                }
                // PLAY_ARROW_HIT_SOUND: ClientPacketListener.handleGameEvent
                // plays ARROW_HIT_PLAYER at player eye height.
                6 => {
                    if let Some(at) = self.eye_position() {
                        self.play_client_local_sound(
                            LevelSound::new("minecraft:entity.arrow.hit_player", at, 0.18, 0.45)
                                .in_source(mod_api::SoundSource::Players),
                        );
                    }
                }
                // RAIN_LEVEL_CHANGE: ClientPacketListener.handleGameEvent.
                7 => {
                    self.world.weather.rain_level = (p.param, p.param);
                    self.world.weather.raining = p.param > 0.0;
                }
                // THUNDER_LEVEL_CHANGE: ClientPacketListener.handleGameEvent.
                8 => {
                    self.world.weather.thunder_level = (p.param, p.param);
                    self.world.weather.thundering = p.param > 0.0;
                }
                // PUFFER_FISH_STING: ClientPacketListener.handleGameEvent
                // plays PUFFER_FISH_STING at player position.
                9 => {
                    if let Some(at) = self.player.as_ref().map(|m| m.position) {
                        self.play_client_local_sound(
                            LevelSound::new(
                                "minecraft:entity.puffer_fish.sting",
                                [at.x, at.y, at.z],
                                1.0,
                                1.0,
                            )
                            .in_source(mod_api::SoundSource::Neutral),
                        );
                    }
                }
                // LIMITED_CRAFTING: LocalPlayer.setDoLimitedCrafting.
                12 => {
                    if let Some(r) = self.remote.as_mut() {
                        r.do_limited_crafting = p.param == 1.0;
                    }
                }
                _ => {}
            },
            C::PlayerAbilities(p) => {
                self.creative = p.flags & 8 != 0;
                if let Some(r) = self.remote.as_mut() {
                    r.abilities = Some(p);
                }
                self.apply_remote_abilities();
            }
            // ClientPacketListener.handleSetHeldSlot: a hotbar slot only, and the
            // carried index stays, so the next tick answers with SetCarriedItem.
            C::SetHeldSlot(p) => {
                if (0..9).contains(&p.slot) {
                    self.selected = p.slot as usize;
                }
            }
            C::ContainerSetContent(p) => {
                self.remote_menu_state(p.container, p.state);
                for (slot, item) in p.items.iter().enumerate() {
                    let stack = self.to_stack(item);
                    self.set_menu_slot(p.container, slot as i32, stack);
                }
                self.carried = self.to_stack(&p.carried);
                self.remote_synced();
            }
            C::ContainerSetSlot(p) => {
                self.remote_menu_state(p.container, p.state);
                let stack = self.to_stack(&p.item);
                self.tutorial_get_item(stack.as_ref());
                // ClientPacketListener.handleContainerSetSlot: a hotbar slot that
                // gains items squeezes for 5 ticks.
                if p.container == 0
                    && (36..=45).contains(&p.slot)
                    && let Some(new) = &stack
                {
                    let (pop, old) = match p.slot {
                        45 => (9, self.offhand.as_ref()),
                        s => (s as usize - 36, self.inventory[s as usize - 36].as_ref()),
                    };
                    if old.is_none_or(|old| old.count < new.count) {
                        self.hotbar_pop[pop] = 5;
                    }
                }
                let before = self.creative_stacks();
                self.set_menu_slot(p.container, i32::from(p.slot), stack);
                self.remote_synced_since(&before);
            }
            C::SetCursorItem(p) => {
                let stack = self.to_stack(&p.item);
                self.tutorial_get_item(stack.as_ref());
                self.carried = stack;
            }
            C::SetPlayerInventory(p) => {
                let stack = self.to_stack(&p.item);
                self.tutorial_get_item(stack.as_ref());
                let before = self.creative_stacks();
                let held = match p.slot {
                    0..=35 => self.inventory[p.slot as usize].as_ref(),
                    40 => self.offhand.as_ref(),
                    _ => None,
                };
                let changed = !super::hand::matches_for_swap(held, stack.as_ref());
                if let Ok(slot) = usize::try_from(p.slot) {
                    self.remote_hand_slot_set(slot, changed);
                }
                match p.slot {
                    0..=35 => self.inventory[p.slot as usize] = stack,
                    36..=39 => self.armor[3 - (p.slot - 36) as usize] = stack,
                    // Inventory.SLOT_OFFHAND.
                    40 => self.offhand = stack,
                    _ => {}
                }
                self.remote_synced_since(&before);
            }
            C::ContainerSetData(p) if self.open_menu_name(p.container) == "minecraft:lectern" => {
                self.remote_lectern_page(i32::from(p.value));
            }
            C::ContainerSetData(p) if self.remote_station_open(p.container) => {
                self.remote_station_data(usize::try_from(p.id).unwrap_or(usize::MAX), p.value);
            }
            C::ContainerSetData(p) if self.native_screen_data(p.container, p.id, p.value) => {}
            C::ContainerSetData(p) => {
                let v = i32::from(p.value);
                match self
                    .container
                    .and_then(|key| self.world.block_entities.get_mut(&key))
                {
                    Some(BlockEntity::Furnace(f)) => match p.id {
                        0 => f.lit_remaining = v,
                        1 => f.lit_total = v,
                        2 => f.cooking_timer = v,
                        3 => f.cooking_total = v,
                        _ => {}
                    },
                    // CrafterBlockEntity.containerData.
                    Some(BlockEntity::Crafter(c)) => match usize::try_from(p.id) {
                        Ok(k @ 0..9) => c.disabled[k] = v == 1,
                        Ok(9) => c.triggered = v == 1,
                        _ => {}
                    },
                    // BrewingStandBlockEntity.dataAccess.
                    Some(BlockEntity::Brewing(b)) => match p.id {
                        0 => b.brew_time = v,
                        1 => b.fuel = v,
                        2 => b.total_brew_time = v,
                        3 => b.total_fuel = v,
                        _ => {}
                    },
                    _ => {}
                }
            }
            C::OpenScreen(p) => self.remote_open_screen(p),
            C::MountScreenOpen(p) => self.remote_open_mount(p),
            C::UpdateRecipes(p) => self.remote_recipe_properties(p),

            C::RecipeBookAdd(p) => self.remote_recipe_book_add(p),
            C::RecipeBookRemove(p) => self.remote_recipe_book_remove(p),
            C::RecipeBookSettings(p) => self.remote_recipe_book_settings(p),
            C::PlaceGhostRecipe(p) => self.remote_ghost_recipe(p),
            C::MerchantOffers(p) => self.remote_merchant_offers(p),
            C::ContainerClose(_) => {
                // `LocalPlayer.clientSideCloseContainer` sets the screen to
                // null, including dialogs and WaitingForResponseScreen. This
                // differs from clearDialog, which restores the parent screen.
                let dialog = self.dialog_open();
                if dialog {
                    self.close_dialogs();
                    self.ui.chat_screen = None;
                    self.book = None;
                    self.sign_edit = None;
                }
                if !self.remote_lectern_closed() && (self.screen == Screen::Inventory || dialog) {
                    self.remote_close_local();
                    self.set_screen(Screen::Playing);
                    self.capture(true);
                }
            }
            C::TagQuery(p) => self.answer_tag_query(p.transaction, p.tag),
            C::Unknown(id, data)
                if id == horizon_protocol::ids::play::clientbound::COMMAND_SUGGESTIONS =>
            {
                self.handle_command_suggestions(&data);
            }
            C::CustomChatCompletions(p) => {
                // ClientPacketListener.handleCustomChatCompletions.
                if let Some(r) = self.remote.as_mut() {
                    horizon_client::command::modify_custom_completions(
                        &mut r.custom_completions,
                        horizon_client::command::CustomCompletionAction::from_id(p.action),
                        &p.entries,
                    );
                    self.ui.registries.custom_completions = r.custom_completions.clone();
                }
            }
            C::UpdateTags(p) => self.remote_update_tags(&p),
            C::Commands(p) => {
                let tree = horizon_client::command::network::from_packet(
                    &p,
                    horizon_client::command::vanilla::dispatcher(),
                );
                // The chat's tree follows the server's now; a mod's commands
                // go into it again (`chat_tree_for`).
                let chat_tree = self.chat_tree_for(Some(&tree));
                if let Some(r) = self.remote.as_mut() {
                    r.chat_tree = chat_tree;
                    r.chat_tree_received = true;
                }
            }
            C::Cooldown(p) => {
                // ClientPacketListener.handleItemCooldown: the server starts
                // (or confirms) a group's cooldown; 0 ends it.
                let group: Arc<str> = Arc::from(p.group.as_str());
                if p.duration == 0 {
                    self.cooldowns.remove(&group);
                } else {
                    self.cooldowns.add(group, p.duration);
                }
            }
            // ClientPacketListener.handleGameTestHighlightPos.
            C::GameTestHighlightPos(p) => self.highlight_test_pos(p.absolute, p.relative),
            // ClientPacketListener.handleTestInstanceBlockStatus.
            C::TestInstanceBlockStatus(p) => {
                self.block_editor_status(remote_text(&p.status), p.size);
            }
            C::SystemChat(p) => {
                // The component as sent, translated in the client's own
                // language; mods may change or cancel it. ChatListener.handleOverlay
                // for the action bar.
                let mut text = remote_text(&p.content);
                if !self.chat_received(true, p.overlay, 0, None, &mut text) {
                    return Ok(());
                }
                if p.overlay {
                    self.say_system_queued(&text);
                    self.titles.set_overlay(text, false);
                } else if self.add_chat_line(text.clone(), horizon_client::chat::ChatSource::Server)
                {
                    // handleSystemMessage: said once the chat takes it.
                    self.say_system_chat_queued(&text);
                }
            }
            C::PlayerChat(p) => {
                use horizon_client::scoreboard::tag_to_json;
                // SignedMessageBody.Packed.unpack: the last seen it names must
                // be in the cache; then the message joins it
                // (MessageSignatureCache.push).
                let Some(r) = self.remote.as_mut() else {
                    return Ok(());
                };
                // handlePlayerChat: the global index counts every chat packet; one out of
                // order disconnects.
                let expected = r.next_chat_index;
                r.next_chat_index = r.next_chat_index.wrapping_add(1);
                if p.global_index != expected {
                    log::error!(
                        "Missing or out-of-order chat message from server, expected index {expected} but got {}",
                        p.global_index
                    );
                    let reason = horizon_client::text::Text::translate(
                        "multiplayer.disconnect.bad_chat_index",
                        &[],
                    );
                    return self.leave_server(Some(reason.plain()));
                }
                let last_seen: Option<Vec<Signature>> =
                    p.last_seen.iter().map(|s| r.signatures.unpack(s)).collect();
                let Some(last_seen) = last_seen else {
                    log::error!(
                        "Message from player with ID {:032x} referenced unrecognized signature id",
                        p.sender
                    );
                    let reason = horizon_client::text::Text::translate(
                        "multiplayer.disconnect.invalid_packet",
                        &[],
                    );
                    return self.leave_server(Some(reason.plain()));
                };
                let signature = p
                    .signature
                    .as_deref()
                    .and_then(horizon_protocol::secure_chat::signature);
                r.signatures.push(&last_seen, signature.as_ref());
                // The sender's PlayerInfo: none is an error line; else their
                // `SignedMessageValidator` decides (`updateAndValidate`), and a message it
                // takes without a signature of its own shows as unsigned.
                if !r.tab_players.contains_key(&p.sender) {
                    log::error!(
                        "Received player chat packet for unknown player with ID: {:032x}",
                        p.sender
                    );
                    self.handle_chat_message_error(&p.chat_type, signature.as_ref());
                    return Ok(());
                }
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_millis() as i64);
                let incoming = secure_chat::Incoming {
                    sender: p.sender,
                    index: p.index,
                    signature: signature.as_ref(),
                    body: secure_chat::Body {
                        content: &p.content,
                        timestamp: p.timestamp,
                        salt: p.salt,
                        last_seen: &last_seen,
                    },
                    unsigned_content: p.unsigned_content.as_ref(),
                    filter: &p.filter,
                };
                let Some(validated) = r.validate_chat(&incoming, now) else {
                    self.handle_chat_message_error(&p.chat_type, signature.as_ref());
                    return Ok(());
                };
                let signature = signature.filter(|_| validated == secure_chat::Validated::Signed);
                // ChatListener.handlePlayerChatMessage: Only Show Secure Chat
                // shows the signed content, never the server's replacement.
                // The narrator says the message's own decorated content
                // (`narrateChatMessage(boundChatType, message.decoratedContent())`).
                let decorated = p
                    .unsigned_content
                    .as_ref()
                    .map_or_else(|| serde_json::Value::from(p.content.as_str()), tag_to_json);
                let content = if self.settings.file.bool("onlyShowSecureChat") {
                    serde_json::Value::from(p.content.as_str())
                } else {
                    decorated.clone()
                };
                let narration = self
                    .decorate_chat(&p.chat_type, &decorated, Decoration::Narration)
                    .plain();
                let line = self.decorate_chat(&p.chat_type, &content, Decoration::Chat);
                let trust = self.chat_trust(
                    p.sender,
                    signature.is_some(),
                    p.timestamp,
                    &p.content,
                    p.unsigned_content.as_ref(),
                    &line,
                );
                // A hidden or fully filtered message counts as processed, not shown.
                let Some((line, narration)) = self.shown_player_chat(&p, line, narration) else {
                    if let Some(signature) = &signature {
                        self.mark_message_processed(signature, false);
                    }
                    return Ok(());
                };
                self.handle_player_chat(QueuedChat {
                    line,
                    signature,
                    trust,
                    narration,
                    sender: p.sender,
                    chat_type: p.chat_type.chat_type,
                });
            }
            C::DisguisedChat(p) => {
                let content = horizon_client::scoreboard::tag_to_json(&p.message);
                let narration = self
                    .decorate_chat(&p.chat_type, &content, Decoration::Narration)
                    .plain();
                let line = self.decorate_chat(&p.chat_type, &content, Decoration::Chat);
                self.handle_player_chat(QueuedChat {
                    line,
                    signature: None,
                    trust: ChatTrust::Secure,
                    narration,
                    sender: 0,
                    chat_type: p.chat_type.chat_type,
                });
            }
            C::DeleteChat(p) => self.handle_delete_chat(&p.signature)?,
            C::PlayerInfoUpdate(p) => {
                if let Some(r) = self.remote.as_mut() {
                    // handlePlayerInfoUpdate: new entries first, then every
                    // action on every known entry.
                    // INITIALIZE_CHAT: the session (or none) each entry carries; the
                    // player's validator is made again when their chat is next read.
                    if p.actions & 2 != 0 {
                        for e in &p.entries {
                            r.chat_sessions.insert(e.uuid, e.chat_session.clone());
                            r.chat_validators.remove(&e.uuid);
                        }
                    }
                    if p.actions & 1 != 0 {
                        for e in &p.entries {
                            if let Some(name) = &e.name {
                                r.tab_players.entry(e.uuid).or_insert_with(|| {
                                    horizon_client::tab_list::TabPlayer::new(e.uuid, name.clone())
                                });
                                if let Some(t) = e.properties.iter().find(|p| p.name == "textures")
                                {
                                    // Without the services' profile property keys a
                                    // signature can't be checked; a signed property
                                    // counts as genuine.
                                    use horizon_client::skins::{self, Signature};
                                    let signature = if t.signature.is_some() {
                                        Signature::Signed
                                    } else {
                                        Signature::Unsigned
                                    };
                                    let textures =
                                        skins::unpack(&t.value, signature, skins::texture_uris());
                                    r.skin_textures.entry(e.uuid).or_insert(textures);
                                }
                            }
                        }
                    }
                }
                for e in p.entries {
                    if mod_api::__private::listened(mod_api::HookId::ClientPlayerChangeGameType)
                        && self.player.is_some()
                        && let Some(mode) = e.game_mode
                        && let Some(r) = self.remote.as_ref()
                        && r.profile == e.uuid
                        && let Some(info) = r.tab_players.get(&e.uuid)
                        && info.game_mode != mode
                    {
                        let current = info.game_mode;
                        self.native_client_player_change_game_type(
                            self.own_entity_id(),
                            e.uuid,
                            current,
                            mode,
                        );
                    }
                    if let Some(r) = self.remote.as_mut() {
                        if let Some(t) = r.tab_players.get_mut(&e.uuid) {
                            if let Some(mode) = e.game_mode {
                                t.game_mode = mode;
                            }
                            if let Some(latency) = e.latency {
                                t.latency = latency;
                            }
                            if let Some(name) = &e.display_name {
                                t.display_name =
                                    name.as_ref().map(horizon_client::scoreboard::tag_to_json);
                            }
                            if let Some(order) = e.list_order {
                                t.list_order = order;
                            }
                            if let Some(hat) = e.show_hat {
                                t.show_hat = hat;
                            }
                            match e.listed {
                                Some(true) => {
                                    r.listed.insert(e.uuid);
                                }
                                Some(false) => {
                                    r.listed.remove(&e.uuid);
                                }
                                None => {}
                            }
                        }
                        if let Some(name) = e.name {
                            r.players.insert(e.uuid, name);
                        }
                    }
                }
            }
            C::PlayerInfoRemove(p) => {
                if let Some(r) = self.remote.as_mut() {
                    for u in p.uuids {
                        r.players.remove(&u);
                        r.tab_players.remove(&u);
                        r.chat_sessions.remove(&u);
                        r.chat_validators.remove(&u);
                        r.skin_textures.remove(&u);
                        r.listed.remove(&u);
                        r.attachments.remove(&Holder::Player(u));
                    }
                }
            }
            C::TabList(p) => {
                // handleTabListCustomisation: empty text clears it.
                let text = |t: &Tag| {
                    let json = horizon_client::scoreboard::tag_to_json(t);
                    let text = horizon_client::scoreboard::comp_text(&json);
                    text.spans
                        .iter()
                        .any(|span| !span.text.is_empty())
                        .then_some(text)
                };
                self.tab_list.header = text(&p.header);
                self.tab_list.footer = text(&p.footer);
            }
            C::InitializeBorder(p) => {
                let b = &mut self.border;
                b.set_center([p.center_x, p.center_z]);
                if p.lerp_time > 0 {
                    b.lerp_between(p.old_size, p.new_size, p.lerp_time);
                } else {
                    b.set_size(p.new_size);
                }
                b.set_absolute_max_size(p.absolute_max_size);
                b.warning_distance = p.warning_blocks;
                b.warning_time = p.warning_time;
            }
            C::SetBorderCenter(p) => self.border.set_center([p.x, p.z]),
            C::SetBorderLerpSize(p) => {
                self.border
                    .lerp_between(p.old_size, p.new_size, p.lerp_time)
            }
            C::SetBorderSize(p) => self.border.set_size(p.size),
            C::SetBorderWarningDelay(p) => self.border.warning_time = p.delay,
            C::SetBorderWarningDistance(p) => self.border.warning_distance = p.blocks,
            C::SetObjective(p) => self.remote_objective(p),
            C::SetDisplayObjective(p) => {
                // DisplaySlot ids: list, sidebar, below_name, then the team sidebars.
                let slot = match p.slot {
                    0 => "list".to_owned(),
                    2 => "below_name".to_owned(),
                    n @ 3..=18 => format!(
                        "sidebar.team.{}",
                        horizon_client::text::COLORS[(n - 3) as usize].1
                    ),
                    _ => "sidebar".to_owned(),
                };
                let board = &mut self.score.board;
                let objective = board.objective(&p.objective).map(|o| o.name.clone());
                board.set_display_objective(&slot, objective.as_deref());
            }
            C::SetScore(p) => {
                let board = &mut self.score.board;
                if board.objective(&p.objective).is_some() {
                    let s = board.score_mut(&p.owner, &p.objective);
                    s.value = p.score;
                    s.display = p
                        .display
                        .as_ref()
                        .map(horizon_client::scoreboard::tag_to_json);
                    s.number_format = p.number_format.as_ref().map(number_format);
                }
            }
            C::ResetScore(p) => match &p.objective {
                Some(o) => self.score.board.reset_one(&p.owner, o),
                None => self.score.board.reset_all(&p.owner),
            },
            C::SetPlayerTeam(p) => self.remote_team(p),
            C::SetSimulationDistance(p) => {
                if let Some(r) = self.remote.as_mut() {
                    r.simulation_distance = p.distance;
                }
            }
            C::SetChunkCacheRadius(p) => {
                self.streaming.radius = self
                    .settings
                    .render_distance
                    .min(p.distance.saturating_add(1))
                    .max(2);
                if let Some(r) = self.remote.as_mut() {
                    r.view_distance = p.distance;
                    let center = r.chunk_center;
                    // ClientChunkCache.updateViewRadius: a new storage of that size.
                    self.drop_remote_chunks_out_of_range(center, p.distance);
                }
            }
            C::SetChunkCacheCenter(p) => {
                if let Some(r) = self.remote.as_mut() {
                    r.chunk_center = (p.x, p.z);
                    let distance = r.view_distance;
                    self.drop_remote_chunks_out_of_range((p.x, p.z), distance);
                }
            }
            C::BlockEntityData(p) => self.remote_block_entity_data(p.pos, p.kind, &p.tag),
            C::OpenSignEditor(p) => self.remote_open_sign_editor(p),
            C::OpenBook(p) => self.remote_open_book(p.hand),
            // handleSetSpawn: ClientLevel.setRespawnData, which spawn compasses point at.
            C::SetDefaultSpawnPosition(p) => {
                let [x, y, z] = p.pos.pos;
                self.spawn = (p.pos.dimension == "minecraft:overworld")
                    .then(|| Vec3::new(f64::from(x) + 0.5, f64::from(y), f64::from(z) + 0.5));
                // 1.20.2 handleSetSpawn notified ReceivingLevelScreen here; the later
                // LEVEL_CHUNKS_LOAD_START game event did not exist yet.
                if let Some(r) = self.remote.as_mut()
                    && r.session
                        .server_version()
                        .at_most(horizon_protocol::adapter::ServerVersion::V1_20_2)
                {
                    r.level_load.loading_packets_received();
                }
            }
            C::ServerData(p) => {
                if let Some(r) = self.remote.as_mut()
                    && let Some(enforced) = r.session.legacy_secure_chat_enforcement()
                    && r.enforces_secure_chat != enforced
                {
                    r.enforces_secure_chat = enforced;
                    r.chat_validators.clear();
                }
                self.remote_server_data(p.icon.as_deref());
            }
            C::MapItemData(p) => self.remote_map_item_data(&p),
            // handleTickingState / handleTickingStep: the level's TickRateManager.
            C::TickingState(p) => {
                self.clock.set_rate(p.rate);
                self.clock.set_frozen(p.frozen);
            }
            C::TickingStep(p) => self.clock.set_frozen_ticks_to_run(p.steps),
            C::UpdateAdvancements(p) => {
                let icons = p
                    .added
                    .iter()
                    .map(|a| {
                        a.display
                            .as_ref()
                            .map_or_else(String::new, |d| self.item_name(d.icon.item).to_string())
                    })
                    .collect();
                self.remote_advancements(p, icons);
            }
            // handleAwardStats: StatsCounter.setValue for each.
            C::AwardStats(p) => {
                let Some(r) = self.remote.as_ref() else {
                    return Ok(());
                };
                let mut known = Vec::new();
                for (ty, value, count) in p.stats {
                    let Some(ty) = r
                        .builtin
                        .get("minecraft:stat_type")
                        .and_then(|t| t.name(ty))
                    else {
                        continue;
                    };
                    let values = match ty {
                        "minecraft:mined" => "minecraft:block",
                        "minecraft:killed" | "minecraft:killed_by" => "minecraft:entity_type",
                        "minecraft:custom" => "minecraft:custom_stat",
                        _ => "minecraft:item",
                    };
                    if let Some(key) = r.builtin.get(values).and_then(|v| v.name(value)) {
                        known.push((ty.to_owned(), key.to_owned(), count));
                    }
                }
                for (ty, key, count) in known {
                    self.stats.values.entry(ty).or_default().insert(key, count);
                }
            }
            // ClientPacketListener.setActionBarText, setTitleText,
            // setSubtitleText, setTitlesAnimation and handleTitlesClear.
            C::BossEvent(p) => self.apply_boss_event(&p),
            C::SetActionBarText(p) => self.titles.set_overlay(remote_text(&p.text), false),
            C::SetTitleText(p) => self.titles.set_title(remote_text(&p.text)),
            C::SetSubtitleText(p) => self.titles.set_subtitle(remote_text(&p.text)),
            C::SetTitlesAnimation(p) => self.titles.set_times(p.fade_in, p.stay, p.fade_out),
            // ClientCommonPacketListenerImpl.handleShowDialog / handleClearDialog.
            C::ShowDialog(p) => self.show_play_dialog(&p.dialog),
            C::ClearDialog(_) => self.clear_dialog(),
            // ClientPacketListener.handleLowDiskSpaceWarning.
            C::LowDiskSpaceWarning(_) => horizon_client::chunk_io::report_low_disk_space(),
            C::ClearTitles(p) => {
                self.titles.clear();
                if p.value {
                    self.titles.reset_times();
                }
            }
            _ => {}
        }
        Ok(())
    }
    /// `handleAddObjective`: add, remove or change an objective.
    fn remote_objective(&mut self, p: play::SetObjective) {
        use horizon_client::scoreboard::{RenderType, tag_to_json};
        let board = &mut self.score.board;
        match (p.method, p.display) {
            (1, _) => board.remove_objective(&p.name),
            (method, Some((name, render, format))) => {
                let render = if render == 1 {
                    RenderType::Hearts
                } else {
                    RenderType::Integer
                };
                let format = format.as_ref().map(number_format);
                if method == 0 {
                    board.add_objective(
                        &p.name,
                        "dummy",
                        tag_to_json(&name),
                        render,
                        false,
                        format,
                    );
                } else if let Some(o) = board.objective_mut(&p.name) {
                    o.display_name = tag_to_json(&name);
                    o.render_type = render;
                    o.number_format = format;
                }
            }
            _ => {}
        }
    }
    /// `handleSetPlayerTeamPacket`.
    fn remote_team(&mut self, p: play::SetPlayerTeam) {
        use horizon_client::scoreboard::{CollisionRule, Visibility, tag_to_json};
        let board = &mut self.score.board;
        if p.method == 1 {
            board.remove_team(&p.name);
            return;
        }
        if p.method == 0 {
            board.add_team(&p.name);
        }
        if let Some(params) = &p.parameters
            && let Some(team) = board.team_mut(&p.name)
        {
            team.display_name = tag_to_json(&params.display_name);
            team.prefix = tag_to_json(&params.prefix);
            team.suffix = tag_to_json(&params.suffix);
            team.color = params
                .color
                .and_then(|c| horizon_client::text::COLORS.get(c as usize))
                .map(|c| c.1.to_owned());
            team.allow_friendly_fire = params.options & 1 != 0;
            team.see_friendly_invisibles = params.options & 2 != 0;
            team.nametag_visibility = Visibility::ALL
                .get(params.name_tag_visibility as usize)
                .copied()
                .unwrap_or(Visibility::Always);
            team.collision_rule = CollisionRule::ALL
                .get(params.collision_rule as usize)
                .copied()
                .unwrap_or(CollisionRule::Always);
        }
        match p.method {
            0 | 3 => {
                for player in &p.players {
                    board.add_to_team(player, &p.name);
                }
            }
            // Scoreboard.removePlayerFromTeam: only out of this team (vanilla
            // throws for a player in another one).
            4 => {
                for player in &p.players {
                    if board.players_team(player).is_some_and(|t| t.name == p.name) {
                        board.remove_from_team(player);
                    }
                }
            }
            _ => {}
        }
    }
    /// The player list's view of the connection: the listed players (with
    /// their downloaded skins once ready), online mode, and which players
    /// are in the level with an upside-down name.
    pub(super) fn publish_tab_object_skins(&mut self) {
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let fingerprint = (
            r.listed.len(),
            r.listed.iter().fold(0, |xor, uuid| xor ^ uuid),
            r.skins_revision,
            self.gui.atlas.object_skins_revision,
        );
        if r.published_skins == Some(fingerprint) {
            return;
        }
        for uuid in &r.listed {
            let Some(p) = r.tab_players.get(uuid) else {
                continue;
            };
            let skin = r.skins.get(uuid).map_or_else(
                || horizon_client::tab_list::default_skin(*uuid),
                String::as_str,
            );
            for key in [
                format!("id:{uuid:032x}"),
                format!("name:{}", p.name.to_ascii_lowercase()),
            ] {
                if object_skin_changed(&self.gui.atlas.object_skins, &key, skin) {
                    self.gui.atlas.set_object_skin(key, skin.to_owned());
                }
            }
        }
        // Publication itself mutates the atlas; cache its final revision.
        r.published_skins = Some((
            fingerprint.0,
            fingerprint.1,
            fingerprint.2,
            self.gui.atlas.object_skins_revision,
        ));
    }

    pub(super) fn remote_tab_players(
        &mut self,
    ) -> Option<(Vec<horizon_client::tab_list::TabPlayer>, bool, Vec<u128>)> {
        self.update_remote_skins();
        self.publish_tab_object_skins();
        let mut players: Vec<_> = {
            let r = self.remote.as_ref()?;
            r.listed
                .iter()
                .filter_map(|u| r.tab_players.get(u).cloned())
                .collect()
        };
        for p in &mut players {
            p.skin = self
                .remote
                .as_ref()
                .and_then(|r| r.skins.get(&p.uuid).cloned())
                .unwrap_or_else(|| horizon_client::tab_list::default_skin(p.uuid).to_owned());
        }
        let r = self.remote.as_ref()?;
        let upside_down = players
            .iter()
            .filter(|p| {
                (p.name == "Dinnerbone" || p.name == "Grumm")
                    && (p.uuid == r.profile || r.level_players.values().any(|u| *u == p.uuid))
            })
            .map(|p| p.uuid)
            .collect();
        Some((players, r.online_mode, upside_down))
    }

    /// Resolve inline `object:player` profiles through the same skin cache as
    /// player heads on items. The next frame uses the uploaded atlas texture.
    pub(super) fn resolve_inline_profiles(
        &mut self,
        profiles: &[std::sync::Arc<serde_json::Value>],
    ) {
        use horizon_client::skins;
        for profile in profiles {
            let skin = skins::head_skin(
                profile,
                &mut self.skin_profiles.borrow_mut(),
                &mut self.skin_downloads.borrow_mut(),
                |location| self.gui.atlas.sprites.contains_key(location),
            );
            let id = skins::parse_profile(profile).id;
            let name = profile
                .as_str()
                .or_else(|| profile.get("name").and_then(serde_json::Value::as_str));
            // The player list's skin for the same profile is drawn too.
            let keys = [
                Some(format!("id:{id:032x}")),
                name.map(|n| format!("name:{}", n.to_ascii_lowercase())),
            ];
            for key in keys.iter().flatten() {
                if let Some(listed) = self.gui.atlas.object_skins.get(key)
                    && listed.starts_with("skins/")
                {
                    let listed = listed.clone();
                    self.again_skin(&listed);
                }
            }
            self.gui
                .atlas
                .set_object_skin(format!("id:{id:032x}"), skin.clone());
            if let Some(name) = name {
                self.gui
                    .atlas
                    .set_object_skin(format!("name:{}", name.to_ascii_lowercase()), skin);
            }
        }
    }
    fn poll_own_profile(&mut self) {
        let auth = self.auth();
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        match &mut r.own_profile {
            OwnProfile::Idle => {
                let Some(auth) = auth else {
                    r.own_profile = OwnProfile::Done(None);
                    return;
                };
                let (send, recv) = std::sync::mpsc::channel();
                std::thread::Builder::new()
                    .name("Own profile".into())
                    .spawn(move || {
                        let _ = send.send(horizon_client::auth::own_textures(auth.as_ref()));
                    })
                    .ok();
                r.own_profile = OwnProfile::Fetching(recv);
            }
            OwnProfile::Fetching(recv) => {
                if let Ok(own) = recv.try_recv() {
                    r.own_profile = OwnProfile::Done(own);
                }
            }
            OwnProfile::Done(_) => {}
        }
    }
    /// `PlayerInfo.getSkin` for every player entity, including NPCs that are
    /// never in the tab list: resolve each profile's textures and give the
    /// level's player models their skin.
    /// The 3D skin layers model for a player (Options â†’ Skin Customization,
    /// `threeDSkinLayers`): `kind`'s outer layer with one slab per painted
    /// texel, built once per (skin, slim) from the decoded image the skin
    /// downloader still holds (`horizon_client::mobs::skin_layers`). `None`
    /// while the option is off or the image is not in memory yet.
    fn skin_3d_model(
        &self,
        kind: &std::sync::Arc<horizon_client::mobs::MobType>,
        skin: &str,
        slim: bool,
    ) -> Option<std::sync::Arc<horizon_client::mobs::Part>> {
        use horizon_client::mobs::skin_layers;
        if !self.settings.extras.three_d_skin_layers {
            return None;
        }
        let key = (skin.to_owned(), slim);
        if let Some(model) = self.skin_3d.borrow_mut().get(&key) {
            return Some(model);
        }
        let model_key = if slim { "player_slim#main" } else { "main" };
        let base = kind
            .layer(model_key)
            .cloned()
            .unwrap_or_else(|| kind.model.clone());
        let image = self.skin_downloads.borrow().image(skin)?.to_vec();
        let model = std::sync::Arc::new(skin_layers::extrude(
            &base,
            &image,
            skin_layers::SIZE,
            skin_layers::DEPTH,
        )?);
        self.skin_3d.borrow_mut().insert(key, model.clone());
        Some(model)
    }

    pub(super) fn update_remote_skins(&mut self) {
        let local = self.remote.as_ref().map(|r| r.profile);
        self.poll_own_profile();
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let local = local.unwrap_or(r.profile);
        let mut entries = skin_entries(
            &r.level_players,
            &r.skin_textures,
            &mut r.level_player_textures,
        );
        let (online_mode, names): (bool, Vec<(u128, String)>) = (
            r.online_mode,
            r.players.iter().map(|(u, n)| (*u, n.clone())).collect(),
        );
        let own = match &r.own_profile {
            OwnProfile::Done(Some(t)) => Some(t.clone()),
            _ => None,
        };
        // Minecraft.profileFuture: the signed-in account's active skin and cape
        // from `/minecraft/profile` when the server sent none (single player) or
        // only a skin without the equipped cape.
        if let Some(own) = own {
            if let Some((_, t)) = entries.iter_mut().find(|(u, _)| *u == local) {
                if t.skin.is_none() {
                    t.skin = own.skin.clone();
                    if own.skin.is_some() {
                        t.slim = own.slim;
                    }
                }
                if t.cape.is_none() {
                    t.cape = own.cape.clone();
                }
            } else if own.skin.is_some() || own.cape.is_some() {
                entries.retain(|(u, _)| *u != local);
                entries.push((local, own));
            }
        }
        // Skin Customization's Ely.by Skins: on a server that sends no skin for a player (offline
        // mode), the one their name has on Ely.by, as offline launchers show it.
        let mut from_elyby = std::collections::HashSet::new();
        // An Ely.by account's own skin where the server sent none: Ely.by's, as Mojang's profile
        // is a Microsoft account's (`own_profile` asks Mojang nothing for it).
        if !entries.iter().any(|(u, t)| *u == local && t.skin.is_some())
            && let Some(textures) = self.own_elyby_textures()
        {
            entries.retain(|(u, _)| *u != local);
            entries.push((local, textures));
            from_elyby.insert(local);
        }
        if !online_mode {
            let local_name = names
                .iter()
                .find(|(u, _)| *u == local)
                .map(|(_, n)| n.clone());
            self.mabar_update(local_name.as_deref());
            for (uuid, name) in names {
                if entries.iter().any(|(u, t)| *u == uuid && t.skin.is_some()) {
                    continue;
                }
                // Mabar: the skin and cape a phone on this LAN announced for its player.
                if uuid != local
                    && let Some(textures) = self.mabar_textures(&name)
                {
                    entries.retain(|(u, _)| *u != uuid);
                    entries.push((uuid, textures));
                    from_elyby.insert(uuid);
                    continue;
                }
                if let Some(textures) = self.elyby_textures(&name) {
                    entries.retain(|(u, _)| *u != uuid);
                    entries.push((uuid, textures));
                    from_elyby.insert(uuid);
                }
            }
        }
        let mut resolved: Vec<(u128, String)> = Vec::new();
        // PlayerSkin.model of each resolved skin: the textures' `slim`.
        let mut slim = std::collections::HashSet::new();
        // PlayerSkin.cape of each resolved skin.
        let mut capes: HashMap<u128, String> = HashMap::new();
        for (uuid, textures) in entries {
            // PlayerInfo.createSkinLookup: only the local player's own skin
            // may be unsigned; Ely.by's never are.
            let secure = uuid != local && !from_elyby.contains(&uuid);
            let skin = {
                let mut downloads = self.skin_downloads.borrow_mut();
                horizon_client::skins::lookup(&textures, secure, |kind, url| {
                    downloads.state(kind, url)
                })
            };
            if let Some(skin) = skin {
                if textures.slim {
                    slim.insert(uuid);
                }
                if let Some(cape) = horizon_client::skins::cape(&textures) {
                    capes.insert(uuid, cape);
                }
                resolved.push((uuid, skin));
            }
        }
        // Options > Skin Customization: the player's own skin, in this game only (others get
        // what the server sends of the profile).
        if let Some((own, own_slim)) = self.own_skin_sprite() {
            resolved.retain(|(u, _)| *u != local);
            resolved.push((local, own));
            if own_slim {
                slim.insert(local);
            } else {
                slim.remove(&local);
            }
        }
        // An offline name's skin from the launcher, where neither the server
        // nor an account gives the local player one.
        if !resolved.iter().any(|(u, _)| *u == local)
            && let Some((rgba, is_slim)) = horizon_client::skins::offline_skin()
        {
            const OFFLINE: &str = "horizon/offline_skin";
            if !self.skin_slots.contains(OFFLINE) || !self.gui.atlas.sprites.contains_key(OFFLINE) {
                self.store_skin(OFFLINE.into(), rgba.clone());
            }
            if *is_slim {
                slim.insert(local);
            }
            resolved.push((local, OFFLINE.into()));
        }
        // An offline name's cape from the launcher, where the profile has none.
        if !capes.contains_key(&local)
            && let Some(rgba) = horizon_client::skins::offline_cape()
        {
            const OFFLINE: &str = "capes/offline";
            if !self.skin_slots.contains(OFFLINE) || !self.gui.atlas.sprites.contains_key(OFFLINE) {
                self.store_skin(OFFLINE.into(), rgba.to_vec());
            }
            capes.insert(local, OFFLINE.into());
        }
        // PlayerInfo.getSkin: the downloaded skin and its model, else the
        // profile's default skin (DefaultPlayerSkin.get), slim or wide.
        let skin_of = |skins: &HashMap<u128, String>, uuid: u128| match skins.get(&uuid) {
            Some(skin) => (skin.clone(), slim.contains(&uuid)),
            None => {
                let skin = horizon_client::tab_list::default_skin(uuid);
                (skin.to_owned(), skin.contains("/slim/"))
            }
        };
        let skins: HashMap<_, _> = resolved.into_iter().collect();
        let Some((level, own)) = self.remote.as_ref().map(|r| {
            (
                r.level_players
                    .iter()
                    .map(|(id, uuid)| (*id as u32, *uuid))
                    .collect::<Vec<_>>(),
                r.profile,
            )
        }) else {
            return;
        };
        // Only the skins drawn hold atlas slots: the level's players, not
        // every entry of a busy server's player list.
        // Build once: resolving each player must not scan every level entity.
        let players: HashMap<_, _> = self
            .mobs
            .iter()
            .enumerate()
            .filter(|(_, m)| &*m.kind.id == "minecraft:player")
            .map(|(i, m)| (m.id, i))
            .collect();
        for (id, uuid) in level {
            let Some(&i) = players.get(&id) else {
                continue;
            };
            let (skin, is_slim) = self.drawn_skin(uuid, skin_of(&skins, uuid));
            let kind = self.mobs[i].kind.clone();
            let skin_3d = self.skin_3d_model(&kind, &skin, is_slim);
            let cape = capes.get(&uuid).filter(|c| self.again_skin(c)).cloned();
            let mob = &mut self.mobs[i];
            mob.skin = Some(skin);
            mob.slim = is_slim;
            mob.skin_3d = skin_3d;
            mob.cape = cape;
        }
        if self.player_model.is_some() {
            let (skin, is_slim) = self.drawn_skin(own, skin_of(&skins, own));
            // Avatar.isModelPartShown(CAPE): the Skin Customization option.
            let cape = capes
                .get(&own)
                .filter(|_| self.settings.file.bool("modelPart_cape"))
                .filter(|c| self.again_skin(c))
                .cloned();
            let kind = self.player_model.as_ref().map(|m| m.kind.clone());
            let skin_3d = kind
                .as_ref()
                .and_then(|kind| self.skin_3d_model(kind, &skin, is_slim));
            if let Some(mob) = self.player_model.as_mut() {
                mob.skin = Some(skin);
                mob.slim = is_slim;
                mob.skin_3d = skin_3d;
                mob.cape = cape;
            }
        }
        if let Some(r) = self.remote.as_mut()
            && r.skins != skins
        {
            r.skins_revision = r.skins_revision.wrapping_add(1);
            r.skins = skins;
        }
        // ClientMannequin.updateSkin (PlayerSkinRenderCache.lookup): a
        // mannequin's DATA_PROFILE `textures` property resolves like a
        // player's, never required to be signed.
        let mannequins: Vec<(usize, horizon_client::skins::Textures)> = self
            .mobs
            .iter()
            .enumerate()
            .filter(|(_, m)| &*m.kind.id == "minecraft:mannequin" && m.profile.is_some())
            .map(|(i, m)| {
                let profile = m.profile.as_ref();
                let t = profile.and_then(|p| p.properties.iter().find(|p| p.name == "textures"));
                let textures = if let Some(t) = t {
                    let signature = if t.signature.is_some() {
                        horizon_client::skins::Signature::Signed
                    } else {
                        horizon_client::skins::Signature::Unsigned
                    };
                    horizon_client::skins::unpack(
                        &t.value,
                        signature,
                        horizon_client::skins::texture_uris(),
                    )
                } else {
                    // `ResolvableProfile.resolveProfile`: no `textures` and
                    // only a name or an id resolves through the services.
                    let key = profile.and_then(|p| match (&p.name, p.uuid) {
                        (Some(name), None) => {
                            Some(horizon_client::skins::NameOrId::Name(name.clone()))
                        }
                        (None, Some(id)) => Some(horizon_client::skins::NameOrId::Id(id)),
                        _ => None,
                    });
                    key.and_then(|key| match self.skin_profiles.borrow_mut().state(&key) {
                        horizon_client::skins::Resolve::Done(Some(profile)) => {
                            profile.textures().map(|(value, signature)| {
                                horizon_client::skins::unpack(
                                    &value,
                                    signature,
                                    horizon_client::skins::texture_uris(),
                                )
                            })
                        }
                        _ => None,
                    })
                    .unwrap_or_default()
                };
                (i, textures)
            })
            .collect();
        let mut resolved: Vec<(usize, Option<String>, bool)> = Vec::new();
        for (id, textures) in mannequins {
            let skin = {
                let mut downloads = self.skin_downloads.borrow_mut();
                horizon_client::skins::lookup(&textures, false, |kind, url| {
                    downloads.state(kind, url)
                })
            };
            let slim = skin.is_some() && textures.slim;
            resolved.push((id, skin, slim));
        }
        for (i, skin, slim) in resolved {
            let skin = skin.filter(|s| !s.starts_with("skins/") || self.again_skin(s));
            // Resolution and atlas uploads do not insert or remove mobs.
            let mob = &mut self.mobs[i];
            mob.slim = slim && skin.is_some();
            mob.skin = skin;
        }
    }
    /// Puts a downloaded skin in an atlas skin slot (`SkinSlots::place`),
    /// unless every slot holds a skin drawn in the last frame.
    pub(super) fn store_skin(&mut self, location: String, rgba: Vec<u8>) {
        // 3D skin layers reads the pixels back when it builds the slabs.
        self.skin_downloads
            .borrow_mut()
            .remember_image(&location, &rgba);
        let Some((slot, evicted)) = self.skin_slots.place(location.clone()) else {
            return;
        };
        if let Some(old) = evicted {
            self.gui.atlas.sprites.remove(&old);
        }
        let Some(mut rect) = self
            .gui
            .atlas
            .sprites
            .get(&format!("horizon/skin{slot}"))
            .copied()
        else {
            return;
        };
        let (w, h) = (self.gui.atlas.width as f32, self.gui.atlas.height as f32);
        self.map_patches.push(horizon_client::font::Patch {
            x: (rect.u0 * w).round() as u32,
            y: (rect.v0 * h).round() as u32,
            width: 64,
            height: 64,
            rgba,
        });
        // A cape is a 64x32 texture in the top half of the slot (skins::process_cape).
        if location.starts_with("capes/") {
            rect.v1 = (rect.v0 + rect.v1) / 2.0;
        }
        self.gui.atlas.sprites.insert(location, rect);
    }
    /// Whether a downloaded skin or cape is in the atlas to draw, marking it drawn (`again_skin`
    /// without queueing it again: `own_skin_sprite` stores its own).
    pub(super) fn show_skin(&mut self, location: &str) -> bool {
        // A slot without its sprite (the atlas was rebuilt, or the upload
        // never happened) counts as missing, so the skin is stored again.
        let shown =
            self.skin_slots.contains(location) && self.gui.atlas.sprites.contains_key(location);
        if shown {
            self.skin_downloads.borrow_mut().drawn(location);
        }
        shown
    }
    /// Whether a downloaded skin or cape is in the atlas to draw, marking it
    /// drawn; if not (new, or its slot taken), its image is queued for upload
    /// again and it is drawn from the next frame.
    pub(super) fn again_skin(&mut self, location: &str) -> bool {
        if self.show_skin(location) {
            return true;
        }
        self.skin_downloads.borrow_mut().again(location);
        false
    }
    /// A player's skin and model as drawn: the downloaded skin once it is in
    /// the atlas, until then the profile's default (`DefaultPlayerSkin.get`),
    /// as `PlayerInfo.getSkin` gives before the texture has loaded.
    pub(super) fn drawn_skin(
        &mut self,
        uuid: u128,
        (skin, slim): (String, bool),
    ) -> (String, bool) {
        if (skin.starts_with("skins/") || skin == "horizon/offline_skin") && !self.again_skin(&skin)
        {
            let skin = horizon_client::tab_list::default_skin(uuid);
            return (skin.to_owned(), skin.contains("/slim/"));
        }
        (skin, slim)
    }
    /// `drawn_skin` for a face in the player list or spectator menu.
    pub(super) fn face_skin(&mut self, uuid: u128, skin: &str) -> String {
        self.drawn_skin(uuid, (skin.to_owned(), false)).0
    }
    /// ChatType.Bound.decorate (`Decoration::Chat`) and `decorateNarration`
    /// (`Decoration::Narration`): the chat type's decoration from the
    /// server's registry, applied to `content`. Legacy `Â§` codes in it (which plugins still
    /// send) style the line, as `remote_text`'s do.
    pub(super) fn decorate_chat(
        &self,
        bound: &play::ChatTypeBound,
        content: &serde_json::Value,
        decoration: Decoration,
    ) -> horizon_client::text::Text {
        use horizon_client::scoreboard::comp_text;
        let Some(r) = self.remote.as_ref() else {
            return comp_text(content).resolve_legacy();
        };
        let chat_type = r.registries.data("minecraft:chat_type", bound.chat_type);
        comp_text(&chat_type_decorate(chat_type, decoration, bound, content)).resolve_legacy()
    }
    fn remote_item_data(&mut self, id: i32, stack: &Option<Stack>) {
        let Some(stack) = self.to_stack(stack) else {
            return;
        };
        let position = self
            .remote
            .as_mut()
            .and_then(|r| r.pending_items.remove(&id));
        if let Some(item) = self.dropped.iter_mut().find(|d| d.id == id as u32) {
            item.stack = stack;
            return;
        }
        let Some((p, velocity)) = position else {
            return;
        };
        let at = Vec3::new(p[0], p[1], p[2]);
        // ItemEntity's constructor draws bobOffs from the item's own random.
        let mut random = horizon_client::mobs::entity_random(id as u32, at);
        let bob_offset = random.next_float() * std::f32::consts::PI * 2.0;
        self.dropped.push(ItemEntity {
            random,
            ..ItemEntity::new(id as u32, stack, at, velocity, bob_offset)
        });
    }
    /// `Entity.lerpMotion` (and `recreateFromPacket`'s motion) for the
    /// entities whose own tick moves them on a client: dropped items, experience
    /// orbs, falling blocks and primed TNT.
    fn remote_motion(&mut self, id: i32, velocity: [f64; 3]) {
        self.remote_dragon_motion(id, velocity);
        let v = Vec3::new(velocity[0], velocity[1], velocity[2]);
        if let Some(m) = self.mob_index.get_mut(&mut self.mobs, id as u32) {
            // Squid.aiStep derives its body pitch and roll from network motion.
            m.velocity = v;
        } else if let Some(d) = self.dropped.iter_mut().find(|d| d.id == id as u32) {
            d.velocity = v;
        } else if let Some(i) = self.falling.iter().position(|f| f.id == id as u32) {
            self.falling.update(i, |f| f.velocity = v);
        } else if let Some(t) = self.tnts.iter_mut().find(|t| t.id == id as u32) {
            t.velocity = v;
        } else if let Some(o) = self.orbs.iter_mut().find(|o| o.id == id as u32) {
            o.velocity = v;
        } else if let Some(i) = self.carts.iter().position(|c| c.id == id as u32) {
            self.carts.update(i, |c| c.velocity = v);
        } else if let Some(i) = self
            .creatures
            .projectiles
            .iter()
            .position(|p| p.id == id as u32)
        {
            self.creatures.projectiles.update(i, |p| {
                p.velocity = v;
                // AbstractArrow.lerpMotion: the stuck timer restarts, and an
                // arrow the server set moving leaves the block it was stuck in.
                if let Some(st) = p.arrow.as_mut() {
                    st.life = 0;
                    if st.in_ground && v.length_squared() > 0.0 {
                        st.in_ground = false;
                        p.stuck = false;
                    }
                }
            });
        } else if let Some(t) = self.tridents.iter_mut().find(|t| t.id == id as u32) {
            t.velocity = v;
            // AbstractArrow.lerpMotion (ThrownTrident keeps it): `life`
            // restarts, and a trident set moving leaves its block.
            t.in_ground_time = 0;
            if t.in_ground && v.length_squared() > 0.0 {
                t.in_ground = false;
            }
        } else if let Some(r) = self.rockets.iter_mut().find(|r| r.id == id as u32) {
            r.velocity = v;
        } else if self.remote_hook_motion(id as u32, v) {
        } else if let Some(p) = self
            .remote
            .as_mut()
            .and_then(|r| r.pending_items.get_mut(&id))
        {
            p.1 = v;
        } else if !self.potions.remote_thrown_motion(id as u32, v) {
            self.end.remote_eye_motion(id as u32, velocity);
        }
    }
    fn apply_remote_health(&mut self) {
        let Some((health, food, saturation)) = self.remote.as_ref().map(|r| r.health) else {
            return;
        };
        if let Some(p) = self.player.as_mut() {
            if (p.health.health() - health).abs() > f32::EPSILON {
                let _ = p.set_health(health.max(0.0), false);
            }
            p.health.food.food = food;
            p.health.food.saturation = saturation;
        }
    }
    /// ClientPacketListener.handlePlayerAbilities, also for a LocalPlayer made after the packet
    /// came (it follows Login): without one since the last respawn, the game mode's
    /// (GameType.updatePlayerAbilities).
    fn apply_remote_abilities(&mut self) {
        let abilities = self.remote.as_ref().and_then(|r| r.abilities.clone());
        let spectator = self.spectator();
        let creative = self.creative;
        let Some(player) = self.player.as_mut() else {
            return;
        };
        match abilities {
            Some(a) => {
                player.invulnerable = a.flags & 1 != 0;
                player.flying = a.flags & 2 != 0;
                player.may_fly = a.flags & 4 != 0;
                player.flying_speed = a.flying_speed;
            }
            None => {
                player.invulnerable = creative || spectator;
                player.may_fly = creative || spectator;
                player.flying = spectator;
            }
        }
    }
    /// ClientPacketListener.handleUpdateAttributes for a LocalPlayer made after the packet came
    /// (the first teleport, a respawn): the attributes the server last sent.
    fn apply_remote_attributes(&mut self) {
        let Some(snapshots) = self
            .remote
            .as_ref()
            .map(|r| r.player_attributes.clone())
            .filter(|s| !s.is_empty())
        else {
            return;
        };
        if let Some(player) = self.player.as_mut() {
            horizon_client::lan::attributes::apply(&mut player.attributes, &snapshots);
        }
    }
    /// `ServerPlayer.dismountTo` lands a rider at the ghast's own (x, maxY, z)
    /// (`HappyGhast.getDismountLocationForPassenger`). A vanilla client
    /// simulates the ghast it rides, so its copy is where the server has it;
    /// this window's copy trails the server's by the stepped interpolation and
    /// would rise into the rider's feet, which lets the rider fall through.
    /// Put the copy under the landing spot.
    fn place_ghast_under(&mut self, id: u32, spot: Vec3) {
        let Some(m) = self
            .mobs
            .iter_mut()
            .find(|m| m.id == id && m.species() == horizon_client::mobs::Species::HappyGhast)
        else {
            return;
        };
        let at = Vec3::new(spot.x, spot.y - f64::from(m.height()), spot.z);
        // Only a teleport that is that dismount: right above the ghast.
        if (at.x - m.position.x).powi(2) + (at.z - m.position.z).powi(2) > 25.0
            || (at.y - m.position.y).abs() > 5.0
        {
            return;
        }
        m.position = at;
        m.previous = at;
        if let Some(handler) = self
            .remote
            .as_mut()
            .and_then(|r| r.stepped.get_mut(&(id as i32)))
        {
            handler.cancel();
        }
    }
    /// `ClientPacketListener.handleLookAt`: `Entity.lookAt` from this
    /// player's feet or eyes, towards the packet's entity (its feet or eyes)
    /// when this client tracks it, else towards the packet's point.
    fn remote_look_at(&mut self, p: &play::PlayerLookAt) {
        // `EntityAnchorArgument.Anchor`: 0 feet, 1 eyes.
        const EYES: i32 = 1;
        let Some(player) = self.player.as_ref() else {
            return;
        };
        let mut from = [player.position.x, player.position.y, player.position.z];
        if p.from == EYES {
            from[1] += f64::from(player.dimensions.eye_height);
        }
        let to = p
            .entity
            .as_ref()
            .and_then(|e| {
                let m = self.mobs.iter().find(|m| m.id == e.id as u32)?;
                let y = if e.anchor == EYES {
                    m.eye_y()
                } else {
                    m.position.y
                };
                Some([m.position.x, y, m.position.z])
            })
            .unwrap_or(p.target);
        (self.yaw, self.pitch) = look_at_rotation(from, to);
    }
    /// ClientPacketListener.handleMovePlayer.
    fn remote_teleport(&mut self, p: &play::PlayerPosition) -> Result<()> {
        use play::relative as rel;
        let current = self
            .player
            .as_ref()
            .map(|pl| pl.position)
            .unwrap_or(Vec3::ZERO);
        let velocity = self
            .player
            .as_ref()
            .map(|pl| pl.velocity)
            .unwrap_or(Vec3::ZERO);
        let pick = |bit: i32, base: f64, v: f64| if p.relatives & bit != 0 { base + v } else { v };
        let position = Vec3::new(
            pick(rel::X, current.x, p.position[0]),
            pick(rel::Y, current.y, p.position[1]),
            pick(rel::Z, current.z, p.position[2]),
        );
        let yaw = if p.relatives & rel::Y_ROT != 0 {
            self.yaw + p.yaw
        } else {
            p.yaw
        };
        let relative_pitch = p.relatives & rel::X_ROT != 0;
        let pitch = position_packet_pitch(self.pitch, p.pitch, relative_pitch);
        let delta = horizon_client::player_sync::teleport_velocity(
            velocity,
            (self.yaw, self.pitch),
            (yaw, pitch),
            p,
        );
        let mut setback = false;
        // BlockStatePredictionHandler.onTeleport: a prediction made before this
        // one keeps its block but not the player position it remembered.
        if let Some(r) = self.remote.as_mut() {
            r.last_teleport_sequence = r.sequence;
        }
        // handleMovePlayer: a passenger keeps its own position and look (its vehicle
        // carries it), so a vehicle a datapack teleports every tick doesn't snap the
        // rider's view; the teleport is still accepted, with where the player is.
        if self.riding.is_some()
            && let Some(player) = self.player.as_ref()
        {
            let at = player.position;
            self.send(S::AcceptTeleportation(play::AcceptTeleportation {
                id: p.teleport,
                x: at.x,
                y: at.y,
                z: at.z,
                yaw: self.yaw,
                pitch: self.pitch,
            }));
            self.stop_remote_mining();
            self.mining = None;
            return Ok(());
        }
        match self.player.as_mut() {
            Some(player) => {
                // A short teleport back is a correction: the server would not take the move.
                let back = player.position.distance(position);
                let state = format!(
                    "sprinting {}, on ground {}, input {:#04x}",
                    player.sprinting,
                    player.ground.on_ground,
                    self.remote.as_ref().map_or(0, |r| r.last_input)
                );
                if back > 0.0
                    && back < 8.0
                    && let Some(r) = self.remote.as_mut()
                {
                    r.setbacks += 1;
                    setback = true;
                    if r.setbacks <= 5 || r.setbacks % 50 == 0 {
                        // What tells the causes apart: whether the server took any move (the
                        // last one sent, and when), and which teleport this is.
                        let [sx, sy, sz] = r.position_sender.last_sent();
                        log::info!(
                            "Horizon: the server moved the player back {back:.2} blocks to \
                             {:.2} {:.2} {:.2} ({} times; teleport {}, relatives {:#x}; last \
                             sent {sx:.2} {sy:.2} {sz:.2}, {} ms ago; {state})",
                            position.x,
                            position.y,
                            position.z,
                            r.setbacks,
                            p.teleport,
                            p.relatives,
                            r.moved_at.map_or(-1, |at| at.elapsed().as_millis() as i64)
                        );
                        if r.setbacks == 1 || r.setbacks % 50 == 0 {
                            let now = Instant::now();
                            let trail: Vec<String> = r
                                .trail
                                .iter()
                                .map(|([x, y, z], ground, input, at)| {
                                    format!(
                                        "-{}ms {x:.3} {y:.3} {z:.3} {}{input:#04x}",
                                        (now - *at).as_millis(),
                                        if *ground { "ground " } else { "air " }
                                    )
                                })
                                .collect();
                            log::info!("Horizon: the moves before it: {}", trail.join(" | "));
                        }
                    }
                }
                player.position = position;
                player.velocity = delta;
            }
            None => {
                let mut player = PlayerMotion::new(self.motion, position)?;
                // The id the server gave, for the hooks the motion fires.
                player.entity_id = self.own_entity_id();
                // Player.tick derives noPhysics from the current game mode.
                player.spectator = self.spectator();
                player.velocity = delta;
                self.hook.discard();
                self.player_attachments.retire();
                self.resident_entities.local_player.renew();
                self.player = Some(player);
                self.apply_remote_abilities();
                self.apply_remote_attributes();
                self.bind_local_player_residency();
                self.input_player_replaced();
                self.sway = Default::default();
                self.eye = 1.62;
                self.previous_eye = 1.62;
                self.clock.reset();
                self.apply_remote_health();
            }
        }
        if setback {
            self.stick_setback();
        }
        if !self.replay.playing() {
            self.previous_position = position;
        }
        self.yaw = yaw;
        self.pitch = pitch;
        // setValuesFromPositionPacket: setOldPosAndRot gives xRotO the same
        // change, so a relative pitch keeps the model's ease and an absolute
        // one snaps it.
        if let Some(m) = self.player_model.as_mut() {
            m.previous_head_pitch =
                position_packet_pitch(m.previous_head_pitch, p.pitch, relative_pitch);
            m.head_pitch = pitch;
        }
        if let Some(id) = self.remote.as_mut().and_then(|r| r.ghast_dismount.take()) {
            self.place_ghast_under(id, position);
        }
        // handleMovePlayer does not change LocalPlayer.sendPosition's baseline:
        // the next tick must still report the corrected state.
        self.send(S::AcceptTeleportation(play::AcceptTeleportation {
            id: p.teleport,
            x: position.x,
            y: position.y,
            z: position.z,
            yaw,
            pitch,
        }));
        if let Some(r) = self.remote.as_ref() {
            r.told_rotation.set(Some((yaw, pitch)));
        }
        // Then MultiPlayerGameMode.stopDestroyBlock: a dig the teleport (an
        // anticheat's setback, a /tp) interrupts is aborted on the wire, and the
        // attack strength starts again.
        self.stop_remote_mining();
        self.mining = None;
        Ok(())
    }

    /// `ClientPacketListener.tick`'s `LevelLoadTracker.tickClientLoad`: the
    /// player's section is shown once its column is lit; when the level is
    /// ready the loading screen closes and the server hears `PlayerLoaded`
    /// (`notifyPlayerLoaded`).
    fn tick_client_load(&mut self) {
        let Some(player) = self.player.as_ref() else {
            return;
        };
        if !self
            .remote
            .as_ref()
            .is_some_and(|r| r.level_load.is_active())
        {
            return;
        }
        // `LevelRenderer`'s playerCompiledSectionCallback: the section holding
        // the camera is there (the column is lit, as meshing needs).
        let (cx, cz) = (
            (player.position.x.floor() as i32) >> 4,
            (player.position.z.floor() as i32) >> 4,
        );
        let section_ready = self.streaming.is_lit(cx, cz);
        let camera_y = (player.position.y + f64::from(self.eye)).floor() as i32;
        let shape = self.world.shape;
        let check = horizon_client::level_loading::PlayerCheck {
            outside_build_height: camera_y < shape.min_y || camera_y >= shape.min_y + shape.height,
            spectator: player.spectator,
            alive: player.health.health() > 0.0,
        };
        let now = client_millis();
        // LevelLoadingScreen.tick.
        self.tick_load_progress();
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        if section_ready && r.level_load.wants_player_section() {
            r.level_load.player_section_ready();
        }
        r.level_load.tick_client_load(now, check);
        if !r.level_load.is_level_ready(now) {
            return;
        }
        r.level_load.finish();
        if !r.loaded_sent {
            r.loaded_sent = true;
            log::info!(
                "Horizon: the level around the player was ready {} ms after the play phase began",
                r.play_began.map_or(0, |at| at.elapsed().as_millis())
            );
            self.send(S::PlayerLoaded(Empty));
        }
        // LevelLoadingScreen.onClose.
        self.say_system_now(&horizon_client::text::Text::translate(
            "narrator.ready_to_play",
            &[],
        ));
        self.capture_loaded_world();
    }

    /// One client tick while connected (Minecraft.tick with a ClientLevel).
    pub(super) fn tick_remote(&mut self) -> Result<()> {
        self.tick_remote_level()?;
        // The rockets boosting this player tick after it, also while it waits
        // for chunks.
        self.tick_own_rockets();
        // Starling: client mods made for the old hooks tick on any server, single player's own
        // included (native_compat.rs).
        self.native_tick();
        self.tick_book_snapshot_retry();
        // Far tiles the samplers asked for (`lod_net`).
        if let Some(r) = &self.remote {
            r.session.send_far_terrain();
        }
        self.tick_chat_session();
        // Minecraft.tick sends this even when the player cannot tick yet,
        // including while waiting for chunks. Keep it outside early returns.
        if self.remote.as_ref().is_some_and(Remote::has_level) {
            self.send(S::ClientTickEnd(Empty));
            if let Some(r) = self.remote.as_mut() {
                r.count_rate(0, 1);
                let now = Instant::now();
                if let Some(gap) = r.tick_end_at.replace(now).map(|at| now - at)
                    && gap >= GAP_LOGGED
                {
                    log::info!(
                        "Horizon: {} ms between two ticks ended to the server",
                        gap.as_millis()
                    );
                }
            }
        }
        Ok(())
    }

    fn tick_remote_player(&mut self, running: bool, hold: bool) -> Result<()> {
        // Inventory.tick's ItemStack.inventoryTick: the pop time runs down.
        if !hold {
            for pop in &mut self.hotbar_pop {
                *pop = pop.saturating_sub(1);
            }
        }
        // ClientLevel.tick: the border's size change, while the level runs.
        if running {
            self.border.tick();
        }
        // ClientLevel.tick: the End's sky flash, frozen or not.
        self.tick_end_flash();
        self.player_border();
        // The entities added after the player tick after it: their pushes
        // (from where they ended last tick, onto where the player ended)
        // reach its next move, this one.
        let pushed_after = self.remote_pushes(false);
        self.tick_remote_entities(running);
        self.tick_remote_arm_poses();
        self.tick_random_textures();
        // Breeze.tick's block dust: drawn by each client for the breezes it sees.
        if running {
            self.tick_breeze_effects();
        }
        self.tick_remote_guardian_eyes(running);
        // Level.tickBlockEntities: a frozen level ticks no block entities.
        // A step advances the moving pistons created by the block event once,
        // rather than finishing them during the frozen ticks between steps.
        if running {
            self.tick_block_entity_anims();
            self.tick_remote_spawners();
            self.client_tick_beacons();
            self.tick_remote_potent_sulfur();
            self.with_level(|b, level, ticks| {
                b.client_tick_block_entities(level, ticks);
            });
            let pushes = std::mem::take(&mut self.block_ticks.piston_pushes);
            self.push_piston_entities(pushes);
            self.sync_moving_pistons();
        }
        self.tick_hand();
        if running {
            self.tick_particles();
        } else if self.clock.still() {
            self.settle_rain_fog();
        }
        self.tick_frame_visuals(running);
        // Under `HORIZON_STILL` they wait for the world too: endermen's and blazes' particles
        // draw on the particles' random, and effects count down, on every tick a frozen
        // client runs, as many as the run took to get there.
        if !hold {
            self.tick_client_ambient();
            self.tick_client_effects();
        }
        // Gui.tick: the titles, the action bar and the held item's name.
        if !hold {
            self.tick_hud();
        } else {
            // Only its countdowns wait: lying down still opens the bed's chat, waking closes it.
            self.tick_in_bed_chat();
        }
        self.tick_subtitles();
        self.tick_client_load();
        let entity_collisions = self.player_entity_collisions();
        // Those added before it tick before it: their pushes reach this move.
        let pushed = pushed_after.add_vector(self.remote_pushes(true));
        let sprint_window = self.settings.sprint_window;
        let Some(player) = self.player.as_mut() else {
            return Ok(());
        };
        let (cx, cz) = (
            (player.position.x.floor() as i32) >> 4,
            (player.position.z.floor() as i32) >> 4,
        );
        let lit = self.streaming.is_lit(cx, cz);
        let chunks = self.world.chunks.len();
        if let Some(r) = self.remote.as_mut() {
            if lit {
                if r.waiting_ticks >= 100 {
                    log::info!(
                        "Horizon: the chunk under the player came after {} s",
                        r.waiting_ticks / 20
                    );
                }
                r.waiting_ticks = 0;
            } else {
                r.waiting_ticks += 1;
                if r.waiting_ticks % 600 == 100 {
                    log::warn!(
                        "Horizon: waiting {} s for the chunk under the player ({cx}, {cz}): {chunks} \
                         chunks here, {} packets skipped",
                        r.waiting_ticks / 20,
                        r.decode_errors
                    );
                }
            }
        }
        if !lit {
            // LocalPlayer.tick can wait for chunks, but Minecraft.tick still
            // calls sendChanges: rotations and the 20-tick position reminder
            // must continue, as well as moves into the unloaded area.
            if let Some(r) = self.remote.as_ref()
                && r.loaded_sent
            {
                let input = r.last_input;
                self.remote_send_state(input);
            }
            return Ok(());
        }
        // LocalPlayer.tick reaches Player.tick only once the client has loaded
        // and the player's chunk is there; its endpoints bracket that work
        // alone, also on the death screen, not the level pass before it.
        let player = self
            .remote
            .as_ref()
            .is_some_and(|r| r.loaded_sent)
            .then(|| {
                (mod_api::__private::listened(mod_api::HookId::PlayerTickPre)
                    || mod_api::__private::listened(mod_api::HookId::PlayerTickPost))
                .then(|| self.local_player())
                .flatten()
            })
            .flatten();
        if let Some(player) = player {
            self.native_player_tick_pre(mod_api::__private::PlayerTickPreCall { player });
        }
        let result = self.tick_local_player(entity_collisions, pushed, sprint_window);
        // `EntityEvent.Size` for the pose the tick set.
        self.native_local_player_size();
        if let Some(player) = player {
            self.native_player_tick_post(mod_api::__private::PlayerTickPostCall { player });
        }
        result
    }

    /// `LocalPlayer.tick`'s `Player.tick` work: input, riding, motion, sleep
    /// and the state the server hears.
    fn tick_local_player(
        &mut self,
        entity_collisions: Vec<horizon_core::geometry::Aabb>,
        pushed: Vec3,
        sprint_window: i32,
    ) -> Result<()> {
        // Player.tick's ItemCooldowns.tick, after Pre and after
        // handleKeybinds: this path replaces `tick_world`'s for a server's
        // client (multiplayer and the default single player), so the entries
        // advance here or the slot stays shaded until the server says the
        // entry ended. `onCooldownEnded` is the plain ItemCooldowns' no-op on
        // a client. Ticking it after the keybinds keeps a held use from
        // trying again a tick before the server's entry ends.
        self.cooldowns.tick();
        // Player.tick's sleep counter (the darkening and the wake-up fade).
        self.sleep_timer = super::tick_sleep_timer(self.sleep_timer, self.sleeping.is_some());
        let stick = self.pad_move();
        if self.player.is_none() {
            return Ok(());
        }
        // The LevelLoadingScreen takes the input until the level is ready.
        let loading = self
            .remote
            .as_ref()
            .is_some_and(|r| r.level_load.is_active());
        let playing = self.screen == Screen::Playing && !loading && !self.dialog_open();
        let held = |action| playing && self.keys.held(action);
        let mut keys = Keys {
            forward: held(Action::Forward),
            backward: held(Action::Back),
            left: held(Action::Left),
            right: held(Action::Right),
            jump: held(Action::Jump),
            shift: held(Action::Sneak),
            sprint: held(Action::Sprint),
            analog: None,
        }
        .with_stick(stick.filter(|_| playing));
        // A server's anticheat simulates Java's keys: eight directions at full speed. A stick's
        // own angle and tilt (the touch stick's, a pad's) it refuses move after move
        // (PikaNetwork's game servers): there the stick presses the keys it points at, or
        // steers (`stick_steer`), as the player picked. Single player and a LAN host keep it.
        let mode = self.update_stick_mode();
        if mode != ServerStick::Analog {
            keys.analog = None;
        }
        if stick.is_some()
            && playing
            && let Some(r) = self.remote.as_mut()
        {
            r.stick_used_at = Some(Instant::now());
        }
        let steer = if mode == ServerStick::Free {
            self.stick_steer(stick.filter(|_| playing))
        } else {
            None
        };
        if steer.is_some() {
            (keys.forward, keys.backward, keys.left, keys.right) = (true, false, false, false);
            keys.analog = None;
        }
        let [mut strafe, mut forward] = keys.direction();
        let original = [strafe, forward];
        self.native_movement_input(&mut forward, &mut strafe, &mut keys.jump, &mut keys.shift);
        if [strafe, forward] != original {
            if forward != original[1] {
                keys.forward = forward > 0.0;
                keys.backward = forward < 0.0;
            }
            if strafe != original[0] {
                keys.left = strafe > 0.0;
                keys.right = strafe < 0.0;
            }
            keys.analog = Some([strafe, forward]);
        }
        if let Some(r) = self.remote.as_mut() {
            r.steer_yaw = steer;
        }
        let input = u8::from(keys.forward)
            | u8::from(keys.backward) << 1
            | u8::from(keys.left) << 2
            | u8::from(keys.right) << 3
            | u8::from(keys.jump) << 4
            | u8::from(keys.shift) << 5
            | u8::from(keys.sprint) << 6;
        // LocalPlayer.applyInput and aiStep: watching through another
        // entity, the player does not move itself; its keys still reach the
        // server, whose sneak check lets go of the camera.
        let keys = if self.controls_camera() {
            keys
        } else {
            Keys::default()
        };
        // The held lead's end eases from the look at the start of the tick
        // (`yRotO`/`xRotO`), as the local tick does; `tick` returns before it
        // reaches its own write when this window is a server client.
        self.rope_rot_old = (self.yaw, self.pitch);
        // A passenger moves with its vehicle (Entity.rideTick), not by itself.
        let riding = self.screen != Screen::Dead && self.remote_ride(keys, sprint_window)?;
        // The server sends no passenger's position: they sit where their
        // vehicles are now.
        self.position_riders();
        // Entity.getFluidInteractionBox asks the vehicle: a boat afloat shelters
        // its passenger's eyes (AbstractBoat.modifyPassengerFluidInteractionBox).
        let boat = self
            .riding
            .and_then(|id| self.boats.iter().find(|b| b.id == id))
            .map(|b| horizon_client::motion::RiddenBoat {
                under_water: b.under_water(),
                max_y: b.bounds().max_position().y,
            });
        let Some(player) = self.player.as_mut() else {
            return Ok(());
        };
        player.fluids.boat = boat;
        if !riding && !self.replay.playing() {
            self.previous_position = player.position;
        }
        let before = (
            player.ground.on_ground,
            [player.velocity.x, player.velocity.y, player.velocity.z],
            [player.position.x, player.position.y, player.position.z],
        );
        let was_flying = player.flying;
        self.sync_player_body();
        // LivingEntity.tickDeath: the local player counts its deathTime once
        // per client tick while dead (the camera's death roll and zoom).
        if self.screen == Screen::Dead {
            self.ui.death_time = self.ui.death_time.saturating_add(1);
        }
        let sleeping = self.sleeping;
        let sleep_position = sleeping.and_then(|head| self.sleeping_position(head));
        let Some(player) = self.player.as_mut() else {
            return Ok(());
        };
        let was_gliding = player.fall_flying;
        let dead = self.screen == Screen::Dead;
        // LivingEntity.isImmobile: a sleeper lies still, even if its bed's
        // shape is unavailable here; only the server decides when to wake.
        if sleeping.is_some() {
            if let Some(at) = sleep_position {
                player.position = at;
            }
            player.velocity = Vec3::ZERO;
            player.pitch = 0.0;
        } else if !dead && !riding {
            player.client_side = true;
            player.velocity = player.velocity.add_vector(pushed);
            // LivingEntity.checkAutoSpinAttack: a Riptide spin that touches a
            // mob ends and bounces back; the server deals the hit.
            player.spin_targets.clear();
            if player.auto_spin_attack_ticks > 0 {
                player.spin_targets.extend(
                    self.mobs
                        .iter()
                        .filter(|m| !m.removed)
                        .map(|m| (m.id, m.bounds())),
                );
            }
            if let Some(yaw) = steer {
                player.yaw = yaw;
            }
            MotionWorld {
                entities: &entity_collisions,
                player,
                scene: &mut self.world,
                assets: &self.pack.assets,
                collision_binding: self.collision,
                resources: self.motion,
                health: &self.health_resources,
                loop_context: &self.loop_context,
                damage_hooks: &self.damage_hooks,
                tag_items: None,
                block_registries: None,
            }
            .controlled_simulation_tick_with_sprint_window(keys, sprint_window)?;
        } else {
            // LivingEntity.baseTick counts hurtTime down for a dead or riding
            // player too (controlled_simulation_tick does it otherwise): left
            // at 10, the hurt tilt restarted every frame and shook the camera.
            player.health.advance_clock();
            if riding && !dead {
                player.client_side = true;
                let mut passenger = MotionWorld {
                    entities: &entity_collisions,
                    player: &mut *player,
                    scene: &mut self.world,
                    assets: &self.pack.assets,
                    collision_binding: self.collision,
                    resources: self.motion,
                    health: &self.health_resources,
                    loop_context: &self.loop_context,
                    damage_hooks: &self.damage_hooks,
                    tag_items: None,
                    block_registries: None,
                };
                passenger.passenger_pose_tick()?;
                // LivingEntity.aiStep: a passenger freezes in powder snow too, and the
                // frostbite overlay shows it (#8156).
                passenger.passenger_block_effects_tick()?;
            }
        }
        // The server decides fire the window cannot predict (blaze fireballs,
        // fire aspect, lightning): Entity.isOnFire reads the synched flag, so
        // the predicted tick's countdown must not end it while it is set.
        player.fire_ticks = hold_server_fire(
            player.fire_ticks,
            self.remote.as_ref().is_some_and(|r| r.own_on_fire),
        );
        // LocalPlayer.onUpdateAbilities: flight turned on or off (a double
        // jump, a spectator, landing) is sent to the server.
        let flying = player.flying;
        // LocalPlayer.aiStep: a glide this client started is sent to the
        // server (ServerboundPlayerCommandPacket START_FALL_FLYING).
        let started_gliding = player.fall_flying && !was_gliding;
        let events = std::mem::take(&mut player.events);
        self.previous_eye = self.eye;
        self.eye += (player.dimensions.eye_height - self.eye) * 0.5;
        for event in events {
            match event {
                horizon_client::motion::MotionEvent::Emission { state, muffled, .. } => {
                    let sounds = self.res.facts().get(state).sounds.clone();
                    // Player.playStepSound: muffled under the swim sound in water.
                    let (volume, pitch) = if muffled {
                        (sounds.volume * 0.05, sounds.pitch * 0.8)
                    } else {
                        (sounds.volume * 0.15, sounds.pitch)
                    };
                    self.play_client_player_sound(&sounds.step, volume, pitch);
                }
                horizon_client::motion::MotionEvent::Swim { volume, pitch } => {
                    self.play_client_player_sound("minecraft:entity.player.swim", volume, pitch);
                }
                // Entity.doWaterSplashEffect on the client's own level: the
                // server's splash skips this player (Player.playSound), and
                // its particles are the client's own.
                horizon_client::motion::MotionEvent::Splash {
                    high_speed,
                    volume,
                    pitch,
                } => {
                    let name = if high_speed {
                        "minecraft:entity.player.splash.high_speed"
                    } else {
                        "minecraft:entity.player.splash"
                    };
                    self.play_client_player_sound(name, volume, pitch);
                }
                horizon_client::motion::MotionEvent::SplashParticle {
                    bubble,
                    position,
                    velocity,
                } => {
                    use horizon_client::particles::{ParticleOptions, ParticleType};
                    let kind = if bubble {
                        ParticleType::Bubble
                    } else {
                        ParticleType::Splash
                    };
                    self.particles.add(
                        &ParticleOptions::simple(kind),
                        [position.x, position.y, position.z],
                        [velocity.x, velocity.y, velocity.z],
                    );
                }
                horizon_client::motion::MotionEvent::Chime { volume, pitch } => {
                    self.play_client_player_sound(
                        "minecraft:block.amethyst_block.chime",
                        volume,
                        pitch,
                    );
                }
                // NetherPortalBlock.entityInside on the client's own level:
                // the portal overlay and its trigger sound (the server moves
                // the player).
                horizon_client::motion::MotionEvent::PowderSnowMelt { pos } => {
                    self.powder_snow_melt(pos);
                }
                horizon_client::motion::MotionEvent::InsidePortal { .. } => {
                    self.client_inside_nether_portal();
                }
                // HoneyBlock.maybeDoSlideEffects on the client's own level:
                // the slide sound, heard here (LocalPlayer.playSound); the
                // particles come from the server's event 53.
                horizon_client::motion::MotionEvent::HoneySlide { sound: true, .. } => {
                    self.play_client_player_sound("minecraft:block.honey_block.slide", 1.0, 1.0);
                }
                // LivingEntity.causeFallDamage and HoneyBlock.fallOn on the
                // client's own level: the fall and honey sounds are heard here
                // (LocalPlayer.playSound); the server's copy skips this player
                // (Player.playSound).
                horizon_client::motion::MotionEvent::Fall(
                    horizon_client::landing::FallEvent::Sound {
                        sound,
                        volume,
                        pitch,
                        ..
                    },
                ) => {
                    if let Some(name) = sound
                        .player_fall_sound()
                        .or_else(|| sound.honey_landing_sound())
                    {
                        self.play_client_player_sound(name, volume, pitch);
                    }
                }
                _ => {}
            }
        }
        // A stand riding the player follows this move (tickPassenger).
        self.seat_own_stands();
        // Other players on this player (possibly through a plugin seat) follow this tick too;
        // the full pass only when something rides this player.
        if self.remote.as_ref().is_some_and(|r| {
            let me = r.entity_id as u32;
            r.passenger_vehicles.values().any(|&v| v == me)
        }) {
            self.position_riders();
        }
        // The server owns health and hunger.
        self.apply_remote_health();
        self.tick_avatar(before, false);
        // LivingEntity.tick: a sleeper's xRot is 0.
        if sleeping.is_some() {
            self.pitch = 0.0;
            self.snap_player_model_pitch();
        }
        // The player's own model (third person, the inventory's preview).
        if !self.replay.playing() {
            self.tick_player_model();
        }
        self.client_hooks_tick();
        if !dead {
            self.remote_tick_using();
        }
        if flying != was_flying {
            self.send(S::PlayerAbilities(play::PlayerAbilitiesRequest {
                flags: if flying { 2 } else { 0 },
            }));
        }
        if started_gliding {
            let id = self.remote.as_ref().map_or(0, |r| r.entity_id);
            self.send(S::PlayerCommand(play::PlayerCommand {
                id,
                action: 6,
                data: 0,
            }));
        }
        // LocalPlayer.sendChanges sends `input.keyPresses`, which an auto-jump's `makeJump` set to
        // jump for this tick: the server's prediction plays the jump from it. Sent as held, the jump
        // was the client's alone, and the server saw a jump its keys didn't allow.
        let auto_jumped = self.player.as_ref().is_some_and(|p| p.was_auto_jump);
        self.remote_send_state(input | u8::from(auto_jumped) << 4);
        Ok(())
    }

    /// The last `ClientboundPlayerAbilities` the server sent, on a client.
    pub(super) fn remote_abilities(&self) -> Option<&play::PlayerAbilities> {
        self.remote.as_ref()?.abilities.as_ref()
    }
    /// The local player's share of every touching entity's
    /// `LivingEntity.pushEntities`, and of the boats' pushes: on a client
    /// `EntitySelector.pushableBy` lets entities push only the local player,
    /// so the server never sends these pushes. `doPush(player)` is
    /// `player.push(entity)`. `before_player`: the mobs that tick before the
    /// player (`ticked_before_player`) and the boats, else the mobs that tick
    /// after it.
    fn remote_pushes(&self, before_player: bool) -> Vec3 {
        let (Some(p), Some(r)) = (self.player.as_ref(), self.remote.as_ref()) else {
            return Vec3::ZERO;
        };
        // LivingEntity.isPushable: alive, no spectator, off ladders.
        let feet = [
            p.position.x.floor() as i32,
            p.position.y.floor() as i32,
            p.position.z.floor() as i32,
        ];
        if self.screen == Screen::Dead
            || p.spectator
            || self.named_block_in_tag("minecraft:climbable", &self.block_name(feet))
        {
            return Vec3::ZERO;
        }
        let bounds = p.bounds();
        let teams = self.mob_teams();
        // `Entity.getTeam` is `Scoreboard.getPlayersTeam(getScoreboardName())`:
        // a player's score holder is its profile name, every other entity's
        // its UUID text. `mob_teams` keys by UUID and names the local player
        // `PLAYER_NAME`, which misses a server's teams for players.
        let board = &self.score.board;
        let player_team = board.players_team(&r.profile_name);
        let mut push = Vec3::ZERO;
        let order = |id: u32| r.ticked_before_player.contains(&id) == before_player;
        for m in self.mobs.iter().filter(|m| order(m.id)) {
            // Bats never push, nor passengers of one vehicle, nor a spectator; the
            // box test comes before the lookups that far mobs never need (#14053).
            if !m.alive()
                || m.species() == horizon_client::mobs::Species::Bat
                || self
                    .riding
                    .is_some_and(|v| v == m.id || m.extra.mount == Some(v))
                || !m.bounds().intersects(bounds)
            {
                continue;
            }
            let listed = r
                .level_players
                .get(&(m.id as i32))
                .and_then(|uuid| r.tab_players.get(uuid));
            let mob_team =
                listed.map_or_else(|| teams.of(Who::Mob(m.id)), |t| board.players_team(&t.name));
            if listed.is_some_and(|t| t.game_mode == SPECTATOR)
                || !horizon_client::scoreboard::pushable_by(mob_team, player_team, true)
            {
                continue;
            }
            // Entity.push(Entity), `this` the player.
            let (mut xa, mut za) = (m.position.x - p.position.x, m.position.z - p.position.z);
            let mut dd = xa.abs().max(za.abs());
            if dd < f64::from(0.01_f32) {
                continue;
            }
            dd = dd.sqrt();
            xa /= dd;
            za /= dd;
            let pow = (1.0 / dd).min(1.0);
            xa *= pow;
            za *= pow;
            xa *= f64::from(0.05_f32);
            za *= f64::from(0.05_f32);
            push = Vec3::new(push.x - xa, push.y, push.z - za);
        }
        // AbstractBoat.push(player) from the boat's own tick (its box grown
        // by 0.2 sideways) and from the player's pushEntities (its box).
        let area = Vec3::new(f64::from(0.2_f32), f64::from(-0.01_f32), f64::from(0.2_f32));
        // The boats' pushes stay where they were measured: with the entities
        // that tick before the player.
        let boats = if before_player {
            &self.boats[..]
        } else {
            &[][..]
        };
        for b in boats.iter().filter(|b| self.riding != Some(b.id)) {
            let bb = b.bounds();
            let times = usize::from(
                bounds.intersects(bb.inflate(area))
                    && horizon_client::scoreboard::pushable_by(None, player_team, true),
            ) + usize::from(
                bounds.intersects(bb)
                    && horizon_client::scoreboard::pushable_by(player_team, None, true),
            );
            if times > 0
                && super::boat_passengers::boat_pushes(bb, bounds, false)
                && let Some(v) = super::boat_passengers::push_impulse(b.position, p.position)
            {
                push = push.add_vector(v.scale(times as f64));
            }
        }
        push
    }
    /// ClientLevel.tickEntities; a frozen level only ticks the players
    /// (`TickRateManager.isEntityFrozen`).
    /// `GuardianRenderer.getEntityToLookAt`: a guardian's eye follows its
    /// synched attack target, else the camera.
    fn tick_remote_guardian_eyes(&mut self, running: bool) {
        use horizon_client::mobs::{Species, Who, guardian};
        if !running {
            return;
        }
        let camera = self.player.as_ref().map(|p| {
            [
                p.position.x,
                p.position.y + f64::from(p.dimensions.eye_height),
                p.position.z,
            ]
        });
        let mut looks = std::mem::take(&mut self.guardian_look_scratch);
        looks.clear();
        looks.extend(
            self.mobs
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    matches!(m.species(), Species::Guardian | Species::ElderGuardian) && m.alive()
                })
                .filter_map(|(i, m)| {
                    let look = match m.ai.target.filter(|_| m.extra.flag) {
                        Some(Who::Mob(id)) => self
                            .mobs
                            .iter()
                            .find(|t| t.id == id)
                            .map(|t| [t.position.x, t.eye_y(), t.position.z]),
                        _ => camera,
                    };
                    look.map(|look| (i, look))
                }),
        );
        for &(i, look) in &looks {
            guardian::look_at(&mut self.mobs[i], look);
        }
        self.guardian_look_scratch = looks;
    }
    /// Humanoid held arm poses, once the tick has set equipment and use state.
    pub(in crate::game) fn tick_remote_arm_poses(&mut self) {
        for i in 0..self.mobs.len() {
            let m = &self.mobs[i];
            if !horizon_client::mobs::anim::extracts_humanoid(m) {
                continue;
            }
            // Extraction also runs while the world is frozen: derive the use
            // animation from the current equipment rather than last tick's state.
            if &*m.kind.id != "minecraft:player" {
                let slot = if m.extra.using_off_hand {
                    horizon_items::equipment::OFFHAND
                } else {
                    horizon_items::equipment::MAINHAND
                };
                let animation = (m.extra.using_item > 0)
                    .then(|| m.extra.slot_stack(slot))
                    .flatten()
                    .map_or_else(Default::default, |stack| self.item_use_animation(stack));
                self.mobs[i].extra.use_animation = animation;
            }
            let m = &self.mobs[i];
            let poses = self.held_arm_poses(m.id, m);
            let charge_duration = (&*m.kind.id != "minecraft:player").then(|| {
                let using_slot = if m.extra.using_off_hand {
                    horizon_items::equipment::OFFHAND
                } else {
                    horizon_items::equipment::MAINHAND
                };
                m.extra
                    .slot_stack(using_slot)
                    .map_or(25, |stack| Self::charge_ticks(stack).max(1)) as i32
            });
            self.mobs[i].extra.held_arm_pose = poses;
            if let Some(duration) = charge_duration {
                self.mobs[i].extra.ints[5] = duration;
            }
        }
    }
    fn tick_remote_entities(&mut self, running: bool) {
        if running {
            self.tick_text_displays();
            self.tick_remote_ominous_item_spawners();
        }
        // EntityFluidInteraction.update for the entities this window only displays: the
        // server never sends the fluid contact, and the renderers read it (cod, salmon and
        // tropical fish lie on their side without `in_water`, #1591).
        // A frozen level ticks only the players, as below.
        // Entity.isInRain for the wolves, which dry off by their own `Wolf.tick`.
        let (mut fluids, mut rain, mut standings, mut water_fx) = {
            let r = self
                .remote
                .as_mut()
                .expect("remote entity tick without remote state");
            (
                std::mem::take(&mut r.tick_fluids),
                std::mem::take(&mut r.tick_rain),
                std::mem::take(&mut r.tick_standings),
                std::mem::take(&mut r.tick_water_fx),
            )
        };
        let thundering = {
            let w = self.mob_world_base(&[]);
            let players = self.remote.as_ref().map(|r| &r.level_players);
            fluids.clear();
            fluids.extend(self.mobs.iter().map(|m| {
                (running || players.is_some_and(|p| p.contains_key(&(m.id as i32))))
                    .then(|| w.fluids(m.bounds(), m.eye_y()))
            }));
            rain.clear();
            rain.extend(self.mobs.iter().map(|m| {
                m.species() == horizon_client::mobs::Species::Wolf
                    && horizon_client::mobs::tame::in_rain(m, &w)
            }));
            // The block under each mob the packs' random-model rules read (a grazing
            // cow), one hair below the feet like `Entity.getBlockStateOn`; nothing
            // without rules.
            standings.clear();
            standings.extend(self.mobs.iter().map(|m| {
                horizon_client::mobs::cem::tracks_standing(&m.kind.id).then(|| {
                    let (x, y, z) = (
                        m.position.x.floor() as i32,
                        (m.position.y - 0.0001).floor() as i32,
                        m.position.z.floor() as i32,
                    );
                    w.block_name(x, y, z).to_owned()
                })
            }));
            w.thundering
        };
        // The window's own player already splashes from its motion tick.
        let own = self.remote.as_ref().map(|r| r.entity_id);
        for i in 0..self.mobs.len() {
            let m = &mut self.mobs[i];
            let fluids = fluids[i].take();
            let standing = standings[i].take();
            if let Some(fluids) = fluids {
                let was = m.fluids.in_water;
                let first = m.first_tick;
                m.fluids = fluids;
                // Entity.doWaterSplashEffect on the client copy. Particles
                // only: ClientLevel draws addParticle, and playSeededSound
                // drops Entity.playSound (the server broadcasts that).
                if m.fluids.in_water && !was && !first && Some(m.id as i32) != own {
                    horizon_client::mobs::living::client_water_splash(m, &mut water_fx);
                }
                m.first_tick = false;
            }
            if let Some(name) = standing {
                m.standing_on = Some(name);
            }
        }
        let game_time = self.world.game_time;
        // `AbstractPiglin.isConverting`: the dimension is not piglin safe.
        let nether = self.dim == super::Dim::Nether;
        // What a wither's side heads can look at, by network entity id: the
        // mobs and the player (position and eye height).
        let wither_targets: Vec<(i32, [f64; 3], f64)> = if self
            .mobs
            .iter()
            .any(|m| m.species() == horizon_client::mobs::Species::Wither)
        {
            let own = self
                .remote
                .as_ref()
                .zip(self.eye_position())
                .map(|(r, eye)| (r.entity_id, eye, eye[1]));
            self.mobs
                .iter()
                .map(|m| {
                    (
                        m.id as i32,
                        [m.position.x, m.position.y, m.position.z],
                        m.eye_y(),
                    )
                })
                .chain(own)
                .collect()
        } else {
            Vec::new()
        };
        // ExperienceOrb.followNearbyPlayer: the players an orb may go for,
        // their feet and the point `eyeHeight / 2` above them.
        let followed: Vec<([f64; 3], [f64; 3])> = {
            let own = self
                .player
                .as_ref()
                .filter(|p| !p.health.dead_or_dying() && !self.spectator())
                .map(|p| {
                    let feet = [p.position.x, p.position.y, p.position.z];
                    let eye = f64::from(p.dimensions.eye_height);
                    (feet, [feet[0], feet[1] + eye / 2.0, feet[2]])
                });
            let others = self.remote.as_ref().into_iter().flat_map(|r| {
                self.mobs
                    .iter()
                    .filter(move |m| m.alive() && r.level_players.contains_key(&(m.id as i32)))
                    .filter(move |m| {
                        let listed = r.level_players.get(&(m.id as i32));
                        !listed
                            .and_then(|uuid| r.tab_players.get(uuid))
                            .is_some_and(|t| t.game_mode == SPECTATOR)
                    })
                    .map(|m| {
                        let feet = [m.position.x, m.position.y, m.position.z];
                        (
                            feet,
                            [feet[0], feet[1] + (m.eye_y() - feet[1]) / 2.0, feet[2]],
                        )
                    })
            });
            own.into_iter().chain(others).collect()
        };
        // The mount this client moves ticks in `remote_ride` instead, as its
        // authoritative side (LivingEntity.tick with travelRidden).
        let driven = self.driven_mount().map(|m| m.id);
        // Cat.handleLieDown: the sleeping players (another's, drawn lying in
        // a bed, and this client's own) a lying cat can lie on top of.
        let sleepers: Vec<horizon_core::geometry::Aabb> = self
            .remote
            .as_ref()
            .map(|r| {
                self.mobs
                    .iter()
                    .filter(|m| {
                        r.level_players.contains_key(&(m.id as i32)) && m.extra.ints[1] != 0
                    })
                    .map(|m| {
                        let p = m.position;
                        horizon_core::geometry::Aabb::new(
                            horizon_core::geometry::Vec3::new(p.x - 0.1, p.y, p.z - 0.1),
                            horizon_core::geometry::Vec3::new(p.x + 0.1, p.y + 0.2, p.z + 0.1),
                        )
                    })
                    .chain(self.sleeping.map(|bed| {
                        let p = horizon_core::geometry::Vec3::new(
                            f64::from(bed[0]) + 0.5,
                            f64::from(bed[1]) + 0.6875,
                            f64::from(bed[2]) + 0.5,
                        );
                        horizon_core::geometry::Aabb::new(
                            horizon_core::geometry::Vec3::new(p.x - 0.1, p.y, p.z - 0.1),
                            horizon_core::geometry::Vec3::new(p.x + 0.1, p.y + 0.2, p.z + 0.1),
                        )
                    }))
                    .collect()
            })
            .unwrap_or_default();
        // Shulker.onPeekAmountChange: the box of this client's own player, which
        // an opening lid pushes (EntitySelector.NO_SPECTATORS).
        let shulker_pushes = self
            .player_view()
            .filter(|p| p.alive && !p.spectator)
            .map(|p| p.bounds);
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let mut lerps = std::mem::take(&mut r.lerps);
        let mut cart_lerps = std::mem::take(&mut r.cart_lerps);
        let mut using = Vec::new();
        let mut mob_fx = water_fx;
        let tick_listened = mod_api::__private::listened(mod_api::HookId::PlayerTickPre)
            || mod_api::__private::listened(mod_api::HookId::PlayerTickPost);
        for (i, rain) in rain.iter().copied().enumerate() {
            let id = self.mobs[i].id;
            let player = self
                .remote
                .as_ref()
                .unwrap()
                .level_players
                .contains_key(&(id as i32));
            if (!running && !player) || driven == Some(id) {
                continue;
            }
            let animation = if player && self.mobs[i].extra.using_item > 0 {
                let extra = &self.mobs[i].extra;
                let slot = if extra.using_off_hand {
                    horizon_items::equipment::OFFHAND
                } else {
                    horizon_items::equipment::MAINHAND
                };
                extra
                    .slot_stack(slot)
                    .map_or_else(Default::default, |stack| self.item_use_animation(stack))
            } else {
                Default::default()
            };
            let player_ref = tick_listened
                .then(|| {
                    self.remote.as_ref().and_then(|r| {
                        let uuid = *r.level_players.get(&(id as i32))?;
                        Some(mod_api::__private::game::player_ref(
                            mod_api::EntityRef::new(self.native_lifecycle.epoch()?, id),
                            uuid,
                        ))
                    })
                })
                .flatten();
            let r = self.remote.as_mut().unwrap();
            let stepped = &mut r.stepped;
            let players = &r.level_players;
            let tab_players = &r.tab_players;
            let m = &mut self.mobs[i];
            if player {
                // AvatarRenderer.extractRenderState: isSpectator, from the player list.
                m.extra.spectator = players
                    .get(&(m.id as i32))
                    .and_then(|uuid| tab_players.get(uuid))
                    .is_some_and(|t| t.game_mode == SPECTATOR);
            }
            m.previous = m.position;
            // `Entity.deltaMovementOnPreviousTick`: the elytra's flight yaw
            // lerps from it, so another player's model does not snap when a
            // new tick's velocity arrives.
            m.extra.previous_velocity = m.velocity;
            m.previous_yaw = m.yaw;
            m.previous_body_yaw = m.body_yaw;
            m.previous_head_yaw = m.head_yaw;
            m.previous_head_pitch = m.head_pitch;
            m.previous_limb_speed = m.limb_speed;
            m.age += 1;
            if m.species() == horizon_client::mobs::Species::Rabbit {
                horizon_client::mobs::rabbit::base_tick(m);
            }
            if matches!(
                m.species(),
                horizon_client::mobs::Species::Horse
                    | horizon_client::mobs::Species::Donkey
                    | horizon_client::mobs::Species::Mule
                    | horizon_client::mobs::Species::SkeletonHorse
                    | horizon_client::mobs::Species::ZombieHorse
            ) {
                horizon_client::mobs::equine::tail_tick(m);
            }
            if m.species() == horizon_client::mobs::Species::Creeper {
                horizon_client::mobs::hostile::creeper_client_tick(m);
            }
            if matches!(
                m.species(),
                horizon_client::mobs::Species::Guardian
                    | horizon_client::mobs::Species::ElderGuardian
            ) {
                horizon_client::mobs::guardian::client_tick(m);
            }
            // PolarBear.tick: the stand animation follows the synched flag.
            if m.species() == horizon_client::mobs::Species::PolarBear {
                horizon_client::mobs::polar_bear::client_tick(m);
            }
            // The particles these mobs' own ticks draw on the client side.
            {
                use horizon_client::mobs::{
                    Species as S, cube, illager, illusioner, nautilus, tame, wither,
                };
                match m.species() {
                    S::Wolf => tame::client_tick(m, rain, &mut mob_fx),
                    S::Slime | S::MagmaCube | S::SulfurCube => cube::client_tick(m, &mut mob_fx),
                    S::Nautilus | S::ZombieNautilus => nautilus::client_tick(m, &mut mob_fx),
                    S::Illusioner => {
                        illusioner::client_tick(m, &mut mob_fx);
                        illager::cast_particles(m, &mut mob_fx);
                    }
                    S::Evoker => illager::cast_particles(m, &mut mob_fx),
                    S::Wither => {
                        wither::client_smoke(m, &mut mob_fx);
                        // `getInvulnerableTicks() > 0`: the charge's cloud,
                        // after the heads' smoke as in `aiStep`.
                        wither::charging_particles(m, &mut mob_fx);
                    }
                    _ => {}
                }
            }
            if m.hurt_time > 0 {
                m.hurt_time -= 1;
            }
            if !m.alive() {
                m.death_time += 1;
            }
            // LivingEntity.updateSwingTime.
            m.anim.attack_anim_o = m.anim.attack_anim;
            horizon_client::mobs::ai::tick_swing(m);
            // Entity.tick: the SteppedInterpolationHandler's step.
            if let Some(at) = stepped
                .get_mut(&(m.id as i32))
                .and_then(|h| h.interpolate())
            {
                place_interpolated(m, at);
            }
            // `ClientLevel.tickNonPassenger`'s `commonTick` interpolated the
            // entity before `Player.tick`, which `Pre` precedes.
            if let Some(player) = player_ref {
                self.native_player_tick_pre(mod_api::__private::PlayerTickPreCall { player });
            }
            let head_lerps = &mut self.remote.as_mut().unwrap().head_lerps;
            let m = &mut self.mobs[i];
            // LivingEntity.aiStep: the head's own interpolation.
            if let Some((yaw, steps)) = head_lerps.get_mut(&(m.id as i32))
                && *steps > 0
            {
                m.head_yaw += wrap_degrees(*yaw - m.head_yaw) / *steps as f32;
                *steps -= 1;
            }
            // Fox.tick: the head tilt and crouch follow the synched flags.
            if m.species() == horizon_client::mobs::Species::Fox {
                horizon_client::mobs::fox::client_tick(m);
            }
            // Cat.tick: the lying and relaxing amounts follow the synched flags.
            if m.species() == horizon_client::mobs::Species::Cat {
                m.anim.vo = m.anim.v;
                horizon_client::mobs::tame::cat_amounts(m);
                horizon_client::mobs::tame::cat_lying_on_sleeper(m, &sleepers);
            }
            // Chicken.aiStep: the wings follow the server's on-ground flag.
            if m.species() == horizon_client::mobs::Species::Chicken {
                horizon_client::mobs::farm::chicken_flap(m);
            }
            // Parrot.calculateFlapping: the wings follow the server's
            // on-ground flag and the shoulder it sits on (the glide is the
            // server's).
            if m.species() == horizon_client::mobs::Species::Parrot {
                horizon_client::mobs::parrot::flap(m);
            }
            // LivingEntity.updatingUsingItem: the ticks the item has been
            // used before this one. Another player's use particles; a mob's
            // drawn bow only needs the count (its pull follows it).
            if player {
                // AvatarRenderer.getArmPose: the used item's use animation
                // poses the arm.
                m.extra.use_animation = animation;
            }
            if m.extra.using_item > 0 {
                if player {
                    let item = if m.extra.using_off_hand {
                        m.extra.slot_stack(horizon_items::equipment::OFFHAND)
                    } else {
                        m.extra.slot_stack(horizon_items::equipment::MAINHAND)
                    }
                    .map(|s| s.item.clone());
                    if let Some(item) = item {
                        let eye = m.eye_height();
                        let at = [m.position.x, m.position.y, m.position.z];
                        using.push((
                            item,
                            m.extra.using_item as u32 - 1,
                            at,
                            eye,
                            m.head_pitch,
                            m.yaw,
                        ));
                    }
                }
                m.extra.using_item += 1;
            }
            // LivingEntity.tick's tickHeadTurn: a mob's BodyRotationControl,
            // a player's body following its walk and look.
            if player {
                remote_player_head_turn(m);
                // LivingEntity.tick: the glide count, wings and swim amount.
                m.tick_avatar_pose(m.fluids.in_water);
                // LivingEntity.tick: a sleeper's xRot is 0.
                if m.extra.ints[1] != 0 {
                    m.head_pitch = 0.0;
                    m.previous_head_pitch = 0.0;
                }
            } else if m.alive() {
                m.body_rotation_tick();
            }
            // Squid.aiStep; WalkAnimationState.update from the horizontal distance.
            horizon_client::mobs::aquatic::squid_client_step(m, &mut mob_fx);
            let d = (m.position.x - m.previous.x).hypot(m.position.z - m.previous.z) as f32;
            let (target, factor) = if m.species() == horizon_client::mobs::Species::Frog {
                (horizon_client::mobs::frog::walk_target(m, d), 0.4)
            } else if horizon_client::mobs::camel::is_camel(m.species()) {
                horizon_client::mobs::camel::walk_target(m, d)
            } else {
                ((d * 4.0).min(1.0), 0.4)
            };
            m.limb_speed += (target - m.limb_speed) * factor;
            m.limb_position += m.limb_speed;
            if m.species() == horizon_client::mobs::Species::Rabbit {
                horizon_client::mobs::rabbit::post_tick(m);
            }
            if m.species() == horizon_client::mobs::Species::Bat {
                horizon_client::mobs::bat::setup_animation_states(m);
            }
            // The animation states a server computes in its own `ai_step`,
            // and the particles vanilla draws on the client.
            horizon_client::mobs::sync::client_tick(m, game_time, &mut mob_fx);
            match m.species() {
                // WitherBoss.aiStep: the side heads turn toward their targets.
                horizon_client::mobs::Species::Wither => {
                    horizon_client::mobs::wither::client_tick(m, |id| {
                        wither_targets
                            .iter()
                            .find(|t| t.0 == id)
                            .map(|t| (t.1, t.2))
                    });
                }
                // Hoglin and piglin `isConverting` outside the Nether.
                horizon_client::mobs::Species::Hoglin
                | horizon_client::mobs::Species::Piglin
                | horizon_client::mobs::Species::PiglinBrute => {
                    horizon_client::mobs::piglin::client_converting(m, nether);
                }
                horizon_client::mobs::Species::Panda => {
                    horizon_client::mobs::panda::client_tick(m, thundering);
                }
                // Shulker.tick's onPeekAmountChange: the lid moves this
                // client's player, whose movement is its own.
                horizon_client::mobs::Species::Shulker => {
                    if let Some(player) = shulker_pushes {
                        horizon_client::mobs::shulker::client_push(m, player, &mut mob_fx);
                    }
                }
                _ => {}
            }
            if let Some(player) = player_ref {
                self.native_player_tick_post(mod_api::__private::PlayerTickPostCall { player });
            }
        }
        // ExperienceOrb.tick runs on a client too (all but the merges and the
        // push out of a block, which are the server's): gravity, buoyancy,
        // collisions and the pull of the nearest player between the server's
        // updates; the server removes it on pickup.
        let (world, facts, collision) = (&self.world, self.res.facts(), self.collision);
        let border = Some(self.border.geometry());
        let mut lava_random = None;
        let mut next_float = || {
            lava_random
                .get_or_insert_with(random_source::entity)
                .next_float()
        };
        if running {
            self.orbs.retain_mut(|o| {
                let distance = |feet: &[f64; 3]| -> f64 {
                    (feet[0] - o.position.x).powi(2)
                        + (feet[1] - o.position.y).powi(2)
                        + (feet[2] - o.position.z).powi(2)
                };
                let target = followed
                    .iter()
                    .filter(|(feet, _)| distance(feet) < 64.0)
                    .min_by(|a, b| distance(&a.0).total_cmp(&distance(&b.0)))
                    .map(|(_, target)| *target);
                // Entity.commonTick: setOldPosAndRot, then
                // LinearInterpolationHandler.doInterpolate moves 1/remaining of
                // the way to the server's position, so the frame draws the step
                // from the old position.
                let old = o.position;
                if let Some(l) = lerps.get_mut(&(o.id as i32)).filter(|l| l.steps > 0) {
                    let alpha = 1.0 / f64::from(l.steps);
                    o.position = Vec3::new(
                        o.position.x + (l.target[0] - o.position.x) * alpha,
                        o.position.y + (l.target[1] - o.position.y) * alpha,
                        o.position.z + (l.target[2] - o.position.z) * alpha,
                    );
                    l.steps -= 1;
                }
                let before = o.position;
                let kept = o.tick(
                    world,
                    facts,
                    collision,
                    target,
                    false,
                    &mut next_float,
                    border,
                    &self.res.motion().fluids,
                    &self.loop_context,
                    &mut |_, _| {},
                );
                // `tick` set `previous` to the stepped position: keep the
                // pre-step one, as `xo` is in vanilla.
                o.previous = old;
                // AbstractInterpolationHandler.applyPredictedMovement: the orb's
                // own move carries the target along (vanilla also checks the
                // moved target for collisions; that check is left out here).
                if let Some(l) = lerps.get_mut(&(o.id as i32)).filter(|l| l.steps > 0) {
                    l.target[0] += o.position.x - before.x;
                    l.target[1] += o.position.y - before.y;
                    l.target[2] += o.position.z - before.z;
                }
                kept
            });
        }
        // ItemEntity.tick runs on a client too: gravity, collisions and
        // friction move the item between the server's updates (every 20
        // ticks). Ordinary removal remains server-authoritative. Mods can opt
        // into predicting consumption at block contact.
        self.remote
            .as_mut()
            .unwrap()
            .head_lerps
            .retain(|_, (_, steps)| *steps > 0);
        let mut entered = Vec::new();
        if running {
            let mut items = self.dropped.lift();
            items.retain_mut(|item| {
                // The server decides entity collisions. This copy predicts block movement.
                let (_, discarded) =
                    self.tick_item_with_block_contacts(item, None, &[], true, &mut |splash| {
                        entered.push(splash);
                    });
                !discarded
            });
            self.put_back_dropped(items);
        }
        // Entity.doWaterSplashEffect on this window: Level.addParticle draws
        // the bubbles and splashes. ClientLevel.playSeededSound drops
        // Entity.playSound, so the sound stays the server's.
        for splash in entered {
            self.apply_item_splash(splash);
        }
        let _ = self.handle_mob_events_reusable(&mut mob_fx);
        if let Some(r) = self.remote.as_mut() {
            r.tick_fluids = fluids;
            r.tick_rain = rain;
            r.tick_standings = standings;
            r.tick_water_fx = mob_fx;
        }
        for (item, ticks, at, eye, pitch, yaw) in using {
            self.remote_player_eating(&item, ticks, at, eye, pitch, yaw);
        }
        // ClientLevel.tick: the sky flash fades in a frozen level too.
        self.flash = self.flash.saturating_sub(1);
        if running {
            self.guardian_bubbles();
            // The dragon: LivingEntity's interpolation, inside its aiStep.
            let dragon_lerp = self
                .dragon
                .as_ref()
                .and_then(|d| lerps.get_mut(&(d.id as i32)))
                .filter(|l| l.steps > 0)
                .map(|l| {
                    let steps = l.steps;
                    l.steps -= 1;
                    (l.target, l.rotation.map(|r| r.0), steps)
                });
            self.tick_remote_dragon(dragon_lerp);
            // Armor stands: ClientLevel.tickNonPassenger's setOldPosAndRot,
            // then their SteppedInterpolationHandler's step, as the mobs'.
            if let Some(r) = self.remote.as_mut() {
                for a in &mut self.armor_stands {
                    let step = r
                        .stepped
                        .get_mut(&(a.id as i32))
                        .and_then(|h| h.interpolate());
                    a.interpolate(step.map(|at| {
                        let p = at.position;
                        (Vec3::new(p[0], p[1], p[2]), at.yaw)
                    }));
                }
            }
            // The yaw of a boat eases toward the server's over the same steps
            // as its position, read before `step` counts one off.
            let turns: HashMap<i32, (f32, u32)> = lerps
                .iter()
                .filter(|(_, l)| l.steps > 0)
                .filter_map(|(id, l)| l.rotation.map(|r| (*id, (r.0, l.steps))))
                .collect();
            // Boats, minecarts, projectiles and the rest ease toward the
            // server's position over three ticks (`lerps`).
            let mut step = |id: u32, at: Vec3| match lerps.get_mut(&(id as i32)) {
                Some(l) if l.steps > 0 => {
                    let n = f64::from(l.steps);
                    l.steps -= 1;
                    Vec3::new(
                        at.x + (l.target[0] - at.x) / n,
                        at.y + (l.target[1] - at.y) / n,
                        at.z + (l.target[2] - at.z) / n,
                    )
                }
                _ => at,
            };
            self.tick_remote_objects(
                &mut step,
                |id, yaw| {
                    turns.get(&(id as i32)).map_or(yaw, |&(target, steps)| {
                        super::remote_objects::lerp_yaw_step(yaw, target, steps)
                    })
                },
                |c| {
                    cart_lerps
                        .get_mut(&(c.id as i32))
                        .is_some_and(|l| l.tick(c))
                },
            );
            self.tick_remote_projectiles(&mut step);
            // LightningBolt.tick on a client.
            self.tick_bolts();
        }
        lerps.retain(|_, l| l.steps > 0);
        if let Some(r) = self.remote.as_mut() {
            for (id, l) in lerps {
                r.lerps.entry(id).or_insert(l);
            }
            for (id, l) in cart_lerps {
                r.cart_lerps.entry(id).or_insert(l);
            }
        }
    }
    /// `AbstractMinecartRenderer.newExtractState`: sample the same weighted
    /// steps at the frame's partial tick, including corners within one tick.
    pub(super) fn remote_minecart_lerp(
        &self,
        id: u32,
        partial: f32,
    ) -> Option<([f64; 3], f32, f32)> {
        let lerp = self.remote.as_ref()?.cart_lerps.get(&(id as i32))?;
        if lerp.current.is_empty() {
            return None;
        }
        let (p, _, yaw, pitch) = lerp.at(partial);
        Some(([p.x, p.y, p.z], yaw, pitch))
    }

    /// `ClientPacketListener.handleMinecartAlongTrack`: the steps join the ones
    /// the minecart has queued, if it runs `NewMinecartBehavior`.
    fn remote_move_minecart(&mut self, p: play::MoveMinecartAlongTrack) {
        if !self.minecart_improvements() || !self.carts.iter().any(|c| c.id == p.id as u32) {
            return;
        }
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let at = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
        r.cart_lerps
            .entry(p.id)
            .or_insert_with(CartLerp::new)
            .pending
            .extend(p.steps.iter().map(|s| CartStep {
                position: at(s.position),
                movement: at(s.movement),
                yaw: horizon_protocol::buf::unpack_degrees(s.yaw),
                pitch: horizon_protocol::buf::unpack_degrees(s.pitch),
                weight: s.weight,
            }));
    }
    /// `Consumable.onConsumeTick` for another player using a consumable: the
    /// crumbs at their mouth every 4 ticks once 21.875% of its use has passed
    /// (`shouldEmitParticlesAndSounds`; the sounds come from the server).
    fn remote_player_eating(
        &mut self,
        item: &str,
        ticks: u32,
        at: [f64; 3],
        eye: f64,
        pitch: f32,
        yaw: f32,
    ) {
        let Some(consumable) = self
            .res
            .items()
            .get(item)
            .and_then(|i| i.consumable.as_ref())
            .filter(|c| c.particles)
        else {
            return;
        };
        let duration = consumable.ticks().max(0) as u32;
        let remaining = duration - ticks.min(duration);
        if !consumable.should_emit(remaining as i32) {
            return;
        }
        let options = self.item_particle(&ItemStack::new(item, 1));
        let mut spawned = Vec::new();
        horizon_client::particles::item_use_particles(
            &mut self.world.random,
            pitch,
            yaw,
            at,
            at[1] + eye,
            5,
            |p, v| spawned.push((p, v)),
        );
        for (p, v) in spawned {
            self.particles.add(&options, p, v);
        }
    }
    /// The end of the player's tick: `AbstractBoat.tick`'s paddles (the vehicle ticks before its
    /// passenger's `sendChanges`), then `LocalPlayer.sendChanges`: the input, and a passenger's look,
    /// vehicle and sprint, or `sendPosition`'s sprint and move. The held slot is not sent here:
    /// `MultiPlayerGameMode.tick` sends it at the start of the next tick (`remote_handle_input`).
    fn remote_send_state(&mut self, input: u8) {
        let Some(p) = self.player.as_ref() else {
            return;
        };
        let (pos, on_ground, collision, sprinting) = (
            [p.position.x, p.position.y, p.position.z],
            p.ground.on_ground,
            p.collision.horizontal,
            p.sprinting,
        );
        let steer = self.remote.as_ref().and_then(|r| r.steer_yaw);
        // A finger aiming with tap to interact: the server sees the player look at what it acts on.
        let (look_yaw, look_pitch) = self.interaction_rotation();
        let (yaw, pitch) = (steer.unwrap_or(look_yaw), look_pitch);
        let riding = self.riding.is_some();
        let controls_camera = self.controls_camera();
        let (paddles, vehicle): (Vec<S>, Vec<S>) = self
            .remote_vehicle_packets()
            .into_iter()
            .partition(|p| matches!(p, S::PaddleBoat(_)));
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let mut out = paddles;
        if input != r.last_input {
            r.last_input = input;
            out.push(S::PlayerInput(play::PlayerInput { flags: input }));
        }
        // LocalPlayer.sendIsSprintingIfNeeded.
        let sprint = |r: &mut Remote, out: &mut Vec<S>| {
            if sprinting != r.was_sprinting {
                r.was_sprinting = sprinting;
                out.push(S::PlayerCommand(play::PlayerCommand {
                    id: r.entity_id,
                    action: if sprinting { 1 } else { 2 },
                    data: 0,
                }));
            }
        };
        let mut moved_or_rotated = false;
        if riding {
            // LocalPlayer.sendChanges: a passenger sends its look, then the
            // vehicle it controls and, only then, its sprint.
            out.push(S::MovePlayerRot(play::MovePlayerRot {
                yaw,
                pitch,
                flags: u8::from(on_ground) | u8::from(collision) << 1,
            }));
            if !vehicle.is_empty() {
                out.extend(vehicle);
                sprint(r, &mut out);
            }
            moved_or_rotated = true;
        } else {
            sprint(r, &mut out);
            // LocalPlayer.sendPosition: no move while the view is another
            // entity's (the server keeps the player on it).
            if controls_camera
                && let Some(packet) = {
                    if let Some(told) = r.told_rotation.take() {
                        r.position_sender.told_rotation(told);
                    }
                    r.position_sender
                        .update(pos, (yaw, pitch), on_ground, collision)
                }
            {
                r.count_rate(1, 0);
                if r.trail.len() == TRAIL {
                    r.trail.pop_front();
                }
                r.trail
                    .push_back((pos, on_ground, r.last_input, Instant::now()));
                log_movement_gap(&mut r.moved_at, r.play_began);
                out.push(packet);
                moved_or_rotated = true;
            }
        }
        if moved_or_rotated && r.session.is_recording() {
            let pkt = horizon_protocol::packets::play::Clientbound::PlayerPosition(
                horizon_protocol::packets::play::PlayerPosition {
                    teleport: 0,
                    position: pos,
                    delta: [0.0, 0.0, 0.0],
                    yaw,
                    pitch,
                    relatives: 0,
                },
            );
            r.session.record_clientbound(&pkt);
        }
        if r.session.is_recording() {
            let is_using = self.using.is_some();
            let living_flags: u8 = if is_using {
                1 | if self.hand == Hand::Off { 2 } else { 0 }
            } else {
                0
            };
            let crouching = p.local_crouching || matches!(p.pose, crate::pose::Pose::Crouching);
            let mut shared: u8 = 0;
            if crouching {
                shared |= 0x02;
            }
            if sprinting {
                shared |= 0x08;
            }
            if p.swimming {
                shared |= 0x10;
            }
            if p.fall_flying {
                shared |= 0x80;
            }
            if shared != r.was_shared_flags
                || living_flags != r.was_living_flags
                || r.last_recorded_pose != Some(p.pose)
            {
                r.was_shared_flags = shared;
                r.was_living_flags = living_flags;
                r.last_recorded_pose = Some(p.pose);
                use horizon_protocol::entity_data::{Value, serializer};
                let pkt = horizon_protocol::packets::play::Clientbound::SetEntityData(
                    horizon_protocol::packets::play::SetEntityData {
                        id: r.entity_id,
                        values: vec![
                            (0, serializer::BYTE, Value::Byte(shared as i8)),
                            (6, serializer::POSE, Value::Id(p.pose.id())),
                            (8, serializer::BYTE, Value::Byte(living_flags as i8)),
                        ],
                    },
                );
                r.session.record_clientbound(&pkt);
            }
        }
        // Creative inventory edits are sent slot by slot (SetCreativeModeSlot).
        let mut creative_slots = Vec::new();
        if self.creative {
            // A server's open menu takes clicks, which it applies itself: what
            // this client predicts for them isn't an edit to send (it would
            // overwrite the server's stacks whenever the prediction is wrong).
            let in_menu = r.menu.as_ref().is_some();
            let stacks = creative_sync_stacks(&self.inventory, &self.offhand, &self.armor);
            for (i, stack) in stacks.into_iter().enumerate() {
                if stack != r.synced[i] {
                    r.synced[i] = stack.clone();
                    if !in_menu {
                        creative_slots.push((creative_sync_slot(i), stack));
                    }
                }
            }
        }
        for packet in out {
            self.send(packet);
        }
        for (slot, stack) in creative_slots {
            let item = self.to_network(&stack);
            self.send(S::SetCreativeModeSlot(play::SetCreativeModeSlot {
                slot,
                item,
            }));
        }
    }
    fn creative_stacks(&self) -> Vec<Option<ItemStack>> {
        creative_sync_stacks(&self.inventory, &self.offhand, &self.armor)
    }
    /// `CreativeInventoryListener.slotChanged`: `handleCreativeModeItemAdd` for every slot the
    /// player's inventory menu changed. `CreativeModeInventoryScreen.slotClicked` ends in
    /// `broadcastChanges`, so a click's slots go out with the click, right behind a thrown
    /// stack, and not at the end of the tick: a slot update the server sends meanwhile (the
    /// pickup of what a held Q threw) can then never land on a change still unsent and undo it.
    pub(super) fn send_creative_slots(&mut self) {
        if !self.creative {
            return;
        }
        let stacks = self.creative_stacks();
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let mut changed = Vec::new();
        for (i, stack) in stacks.into_iter().enumerate() {
            if stack != r.synced[i] {
                r.synced[i] = stack.clone();
                changed.push((creative_sync_slot(i), stack));
            }
        }
        for (slot, stack) in changed {
            let item = self.to_network(&stack);
            self.send(S::SetCreativeModeSlot(play::SetCreativeModeSlot {
                slot,
                item,
            }));
        }
        self.remote_creative_menu_sync();
    }
    /// Inventory as the server last described it.
    fn remote_synced(&mut self) {
        let stacks = self.creative_stacks();
        let menu = self.creative_menu_slots();
        if let Some(r) = self.remote.as_mut() {
            r.synced = stacks;
            if let Some(m) = r.creative_menu.as_mut() {
                *m = menu;
            }
        }
    }
    /// The slots a server packet changed since `before` are as the server has them; the
    /// others keep what was last sent, so a creative click made since still goes out.
    fn remote_synced_since(&mut self, before: &[Option<ItemStack>]) {
        let now = self.creative_stacks();
        let menu = self.creative_menu_slots();
        if let Some(r) = self.remote.as_mut() {
            mark_synced(&mut r.synced, before, now);
            if let Some(m) = r.creative_menu.as_mut() {
                *m = menu;
            }
        }
    }
    /// The inventory menu's slots 1 to 8 and 45: the crafting grid, the armour (head first) and
    /// the off hand.
    fn creative_menu_slots(&self) -> [Option<ItemStack>; 9] {
        let g = &self.craft_grid;
        let [head, chest, legs, feet] = self.armor.clone();
        [
            g[0].clone(),
            g[1].clone(),
            g[2].clone(),
            g[3].clone(),
            head,
            chest,
            legs,
            feet,
            self.offhand.clone(),
        ]
    }
    /// CreativeModeInventoryScreen's slot listener for the slots the inventory's sync leaves
    /// out (the crafting grid, the armour, the off hand): each one a creative screen's clicks
    /// changed is sent as SetCreativeModeSlot, while one is open and once more as it closes.
    pub(super) fn remote_creative_menu_sync(&mut self) {
        const MENU: [i16; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 45];
        let open = self.creative && self.screen == Screen::Creative;
        let now = self.creative_menu_slots();
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let Some(before) = r.creative_menu.as_mut() else {
            if open {
                r.creative_menu = Some(now);
            }
            return;
        };
        let changed: Vec<usize> = (0..MENU.len()).filter(|&i| before[i] != now[i]).collect();
        *before = now.clone();
        if !open {
            r.creative_menu = None;
        }
        for i in changed {
            let item = self.to_network(&now[i]);
            self.send(S::SetCreativeModeSlot(play::SetCreativeModeSlot {
                slot: MENU[i],
                item,
            }));
        }
    }

    /// `MultiPlayerGameMode.stopDestroyBlock`: aborts the break and starts the attack strength
    /// again, if one is running (`if (!isDestroying) return;`), and says whether it did. Only the
    /// progress goes: `destroyBlockPos` and `destroyingItem` keep the target, so a dig resumed on
    /// the same block sends no START and the server's clock runs on (#4124).
    pub(super) fn stop_remote_mining(&mut self) -> bool {
        let kept = self.remote.as_mut().and_then(|r| {
            if !std::mem::take(&mut r.is_destroying) {
                return None;
            }
            let (pos, face, _) = r.destroying?;
            r.destroying = Some((pos, face, 0.0));
            r.destroying
        });
        let Some((pos, _, _)) = kept else {
            return false;
        };
        let state = self.world.block(pos[0], pos[1], pos[2]);
        self.tutorial_destroy(state, -1.0);
        self.send(S::PlayerAction(play::PlayerAction {
            action: 2,
            pos,
            direction: 0,
            sequence: 0,
        }));
        self.reset_attack_strength();
        true
    }

    /// MultiPlayerGameMode.startDestroyBlock / continueDestroyBlock.
    fn remote_mining(&mut self, start: bool) -> bool {
        let debug_stick = self
            .held()
            .is_some_and(|stack| &*stack.item == "minecraft:debug_stick");
        if !self
            .res
            .items()
            .can_destroy_block(self.held(), self.creative)
        {
            self.stop_remote_mining();
            self.mining = None;
            return true;
        }
        let Some(r) = self.remote.as_mut() else {
            return true;
        };
        if !start && r.destroy_delay > 0 {
            r.destroy_delay -= 1;
            return true;
        }
        let Some(hit) = self.pick() else {
            self.mining = None;
            return true;
        };
        let face = face_direction(hit.face);
        let state = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
        if self.res.facts().is_air(state) {
            return true;
        }
        // startDestroyBlock refuses a restricted break before predicting it.
        if self.remote_break_restricted(hit.pos) {
            self.mining = None;
            return true;
        }
        if self.creative {
            if self.veto_mining_prediction(hit.pos, hit.face, true, 1.0) {
                return false;
            }
            let seq = self.next_sequence();
            // DebugStickItem.canDestroyBlock never lets the client break the
            // block; the server only selects the property on this START.
            if !debug_stick {
                self.predict_break(hit.pos, state, seq);
            }
            self.send(S::PlayerAction(play::PlayerAction {
                action: 0,
                pos: hit.pos,
                direction: face,
                sequence: seq,
            }));
            if let Some(r) = self.remote.as_mut() {
                r.destroy_delay = 5;
            }
            return true;
        }
        // sameDestroyTarget: the same block and still the item the break began with.
        let held = self.main_hand_key();
        let same = self.remote.as_ref().is_some_and(|r| {
            r.destroying.is_some_and(|(p, _, _)| p == hit.pos) && r.destroying_item == held
        });
        // startDestroyBlock leaves only a break that is still running alone (`isDestroying &&
        // sameDestroyTarget`); a click on the block a stopped break still names starts it again,
        // so the server re-anchors its clock. A held button carries the same target on, and that
        // one keeps the clock (#4124).
        let running = self.remote.as_ref().is_some_and(|r| r.is_destroying);
        if start && same && running {
            return true;
        }
        // The server's own getDestroyProgress: under water, off the ground,
        // with haste, fatigue and efficiency, so the client stops digging
        // when the server is done too (#1254).
        let progress = self.destroy_progress(hit.pos);
        if !same || start {
            let proposed = if progress >= 1.0 { 1.0 } else { 0.0 };
            if self.veto_mining_prediction(hit.pos, hit.face, true, proposed) {
                return false;
            }
            // startDestroyBlock aborts only a break that is running; a target left by a stop that
            // ended one is just replaced by the START below.
            let aborted = self.remote.as_mut().and_then(|r| {
                if !std::mem::take(&mut r.is_destroying) {
                    return None;
                }
                r.destroying.take()
            });
            if let Some((old, _, _)) = aborted {
                let old_state = self.world.block(old[0], old[1], old[2]);
                self.tutorial_destroy(old_state, -1.0);
                self.send(S::PlayerAction(play::PlayerAction {
                    action: 2,
                    pos: old,
                    direction: face,
                    sequence: 0,
                }));
            }
            let seq = self.next_sequence();
            self.send(S::PlayerAction(play::PlayerAction {
                action: 0,
                pos: hit.pos,
                direction: face,
                sequence: seq,
            }));
            self.tutorial_destroy(state, 0.0);
            // startDestroyBlock: BlockState.attack on the client, so a redstone
            // ore's dust (RedStoneOreBlock.attack) comes before the break.
            self.redstone_ore_particles(hit.pos);
            if progress >= 1.0 {
                // startDestroyBlock: instant survival breaks have no destroy delay.
                if !debug_stick {
                    self.predict_break(hit.pos, state, seq);
                }
                return true;
            }
            if let Some(r) = self.remote.as_mut() {
                r.destroying = Some((hit.pos, face, 0.0));
                r.destroying_item = held;
                r.is_destroying = true;
            }
            // startDestroyBlock plays the first hit and crack itself; the
            // server's level events 2019 and 2020 do the rest.
            self.breaking_block_effects(hit.pos, hit.face, true);
            self.mining = Some(Mining {
                pos: hit.pos,
                progress: 0.0,
                per_tick: 0.0,
                ticks: 0,
            });
            return true;
        }
        let Some((pos, last_face, done)) = self.remote.as_ref().and_then(|r| r.destroying) else {
            return true;
        };
        let done = done + progress;
        if self.veto_mining_prediction(pos, hit.face, false, done) {
            return false;
        }
        self.tutorial_destroy(state, done.clamp(0.0, 1.0));
        let ticks = self.mining.as_ref().map_or(0, |m| m.ticks) + 1;
        if done >= 1.0 {
            let seq = self.next_sequence();
            // DebugStickItem.canDestroyBlock: the server selects the property
            // and sends the block update instead; never predict a break.
            if !debug_stick {
                self.predict_break(pos, state, seq);
            }
            self.send(S::PlayerAction(play::PlayerAction {
                action: 3,
                pos,
                direction: face,
                sequence: seq,
            }));
            if let Some(r) = self.remote.as_mut() {
                // `destroyBlockPos` remains the last target after the break;
                // if that position is filled again before the next pick, a
                // held attack resumes it without sending a second START.
                r.destroying = Some((pos, face, 0.0));
                r.is_destroying = false;
                r.destroy_delay = 5;
            }
            self.mining = None;
        } else {
            if last_face != face {
                self.send(S::PlayerAction(play::PlayerAction {
                    action: 1,
                    pos,
                    direction: face,
                    sequence: 0,
                }));
            }
            if let Some(r) = self.remote.as_mut() {
                r.destroying = Some((pos, face, done));
            }
            self.mining = Some(Mining {
                pos,
                progress: done,
                per_tick: 0.0,
                ticks,
            });
        }
        true
    }

    fn veto_mining_prediction(
        &self,
        pos: [i32; 3],
        face: [i32; 3],
        starting: bool,
        progress: f32,
    ) -> bool {
        if !mod_api::__private::listened(mod_api::HookId::ClientMiningPredict) {
            return false;
        }
        let mut canceled = false;
        self.native_client_mining_predict(
            mod_api::ClientMiningPrediction {
                pos,
                face,
                starting,
                progress: progress.clamp(0.0, 1.0),
            },
            &mut canceled,
        );
        canceled
    }
    /// MultiPlayerGameMode.destroyBlock: the predicted break.
    fn predict_break(&mut self, pos: [i32; 3], state: BlockStateId, sequence: i32) {
        self.touch_vibrate(super::touch::BREAK_BUZZ);
        self.tutorial_destroy(state, 1.0);
        // BlockStatePredictionHandler.retainKnownServerState.
        let here = self.player_position();
        if let Some(r) = self.remote.as_mut() {
            r.predictions
                .entry(pos)
                .and_modify(|p| p.sequence = sequence)
                .or_insert(Prediction {
                    sequence,
                    state: state.0 as i32,
                    player_pos: here,
                });
        }
        // Block.playerWillDestroy's destroy event (particles and the break
        // sound), which BaseFireBlock skips: the server's extinguish sound
        // is all a player hears.
        let facts = self.res.facts().get(state);
        if !facts.has_class("BaseFireBlock") {
            let s = facts.sounds.clone();
            let at = [
                f64::from(pos[0]) + 0.5,
                f64::from(pos[1]) + 0.5,
                f64::from(pos[2]) + 0.5,
            ];
            self.audio.play(
                &s.break_sound,
                Some(at),
                (s.volume + 1.0) / 2.0,
                s.pitch * 0.8,
            );
            self.destroy_particles(pos, state);
        }
        // The server's block update follows (waterlogged blocks leave water).
        let air = self.world.air;
        if self.world.set(pos[0], pos[1], pos[2], air).is_ok() {
            self.world.block_entities.remove(&(pos[0], pos[1], pos[2]));
        }
        self.mining = None;
    }
    /// `Player.blockActionRestricted` for the block at `pos`: a spectator
    /// never breaks, an adventure player only with a main-hand item whose
    /// `can_break` matches the block (`ItemStack.canBreakBlockInAdventureMode`).
    /// A predicate this client could not decode (exact components) only
    /// tells that the server gave one, and the server sends the block back
    /// if it does not match.
    fn remote_break_restricted(&self, pos: [i32; 3]) -> bool {
        use horizon_protocol::item::{WirePatch, component::CAN_BREAK};
        if self
            .held()
            .and_then(|s| s.data())
            .is_some_and(|d| d.other("minecraft:can_break").is_some())
        {
            return self.block_action_restricted(pos);
        }
        self.spectator()
            || (!self.may_build()
                && !self
                    .held()
                    .and_then(|s| s.wire_patch())
                    .is_some_and(|p| WirePatch::adds(&p, CAN_BREAK)))
    }
    /// Minecraft.startUseItem: use one crosshair target, then the item itself.
    fn remote_use(&mut self, attempt: mod_api::InteractionAttempt) {
        // Minecraft.startUseItem returns before resetting rightClickDelay
        // while a survival block break is active.
        if self.remote.as_ref().is_some_and(|r| r.is_destroying) {
            return;
        }
        self.use_delay = 4;
        // Minecraft.startUseItem: nothing is used while the hands paddle a boat.
        if self.hands_busy() {
            return;
        }
        // InputEvent.InteractionKeyMappingTriggered for the main hand: a cancel
        // ends the whole click, the off hand gets no turn.
        if let Some(swing) = self.native_interaction(InteractionKey::Use, Hand::Main, attempt) {
            if swing {
                self.swing_hand(Hand::Main, SwingAnimation::DEFAULT);
            }
            return;
        }
        // An item a mod claims: its use is the mod's, and the server never hears of it.
        if self.mod_use_start() {
            return;
        }
        // Resolve the crosshair once. A passing entity interaction keeps
        // that target and may only fall back to using the item itself.
        let entity = self.remote_use_targets_entity();
        horizon_client::interaction_input::use_on_target(
            &mut RemoteUseWorld(self, attempt),
            entity,
        );
    }

    /// `Minecraft.startUseItem`'s entity turn for `hand`: Interact goes to
    /// the server, and whether the click ends there is what this client's
    /// copy of the entity answers (`MultiPlayerGameMode.interact`).
    fn remote_use_entity(&mut self, hand: Hand) -> bool {
        // A menu this use opens (a chest minecart's) is no clicked block's.
        if let Some(r) = self.remote.as_mut() {
            r.use_target = None;
        }
        let sneaking = self.remote_shift_down();
        // A boat, minecart, item frame, armor stand or painting nearer than
        // any mob: the server uses it (AbstractBoat.interact boards it).
        if let Some(hit) = self.remote_object_hit(false)
            && self.mob_hit().is_none_or(|(_, d, _)| hit.distance < d)
            && self.before_block(hit.distance)
        {
            // Interaction.interact answers SUCCESS when it responds and
            // CONSUME otherwise: both end the click (nothing behind the box
            // is used), but only SUCCESS swings; the packet carries the hand.
            let response = self.interactions.get(&hit.id).map(|i| i.response);
            // Objects whose interact passes (paintings, empty or fixed item frames,
            // boats sneaked at or out of control, carts that pass) leave each hand's
            // turn to its item: a crossbow aimed at it shoots.
            let is_painting = self.paintings.iter().any(|p| p.id as i32 == hit.id);
            let held = self.with_hand(hand, |g| g.held().cloned());
            let frame_passes = self
                .item_frames
                .iter()
                .find(|f| f.id as i32 == hit.id)
                .is_some_and(|f| {
                    item_frame_interact_passes(f.fixed, f.item.is_some(), held.is_some())
                });
            let boat_passes = self
                .boats
                .iter()
                .enumerate()
                .find(|(_, b)| b.id as i32 == hit.id)
                .is_some_and(|(i, b)| {
                    boat_interact_passes(
                        b.chest,
                        sneaking,
                        b.out_of_control >= 60.0,
                        self.boat_has_room(i),
                    )
                });
            let (cart_passes, is_rideable_cart) = self
                .carts
                .iter()
                .find(|c| c.id as i32 == hit.id)
                .map(|c| {
                    let ridden = self.riding == Some(c.id)
                        || self.mobs.iter().any(|m| m.extra.mount == Some(c.id));
                    (
                        c.kind
                            .interact_passes(sneaking, ridden, self.can_use_game_master_blocks()),
                        c.kind == super::minecarts::CartKind::Rideable,
                    )
                })
                .unwrap_or((false, false));
            // Cushion.interact on a client: PASS when sneaked at or when
            // something already sits on it, CONSUME (no swing) otherwise.
            let is_cushion = self.cushions.iter().any(|c| c.id as i32 == hit.id);
            let cushion_passes = is_cushion
                && (sneaking
                    || self.riding.is_some_and(|r| r as i32 == hit.id)
                    || self
                        .mobs
                        .iter()
                        .any(|m| m.extra.mount.is_some_and(|r| r as i32 == hit.id)));
            let passes =
                is_painting || frame_passes || boat_passes || cart_passes || cushion_passes;
            // Otherwise the main hand's turn takes it; there is no off hand turn.
            if response.is_none() && hand == Hand::Off && !passes {
                return false;
            }
            self.send(S::Interact(play::Interact {
                id: hit.id,
                hand: hand_id(hand),
                location: hit.location,
                secondary: sneaking,
            }));
            if passes {
                return false;
            }
            // Minecart.interact resets the passenger's easing baseline on
            // a successful boarding click, not on SetPassengers (commands
            // can mount a player without this interaction).
            if let Some(i) = self.carts.iter().position(|c| {
                c.id as i32 == hit.id && c.kind == super::minecarts::CartKind::Rideable
            }) {
                self.carts
                    .update(i, |c| c.player_rotation_offset = c.rotation_offset);
            }
            // MinecartCommandBlock.interact on the client: the editor opens.
            if self.carts.iter().any(|c| {
                c.id as i32 == hit.id && c.kind == super::minecarts::CartKind::CommandBlock
            }) {
                self.open_cart_command_block(hit.id as u32);
            }
            // Minecart.interact boards the cart and answers CONSUME on the client,
            // which ends the click without an arm swing. AbstractBoat.interact
            // answers SUCCESS there, which swings. Cushion.interact answers
            // CONSUME there too.
            if remote_object_swings(response, is_rideable_cart || is_cushion) {
                self.with_hand(hand, Game::swing_arm);
            }
            return true;
        }
        if let Some((i, distance, location)) = self.mob_hit()
            && self.before_block(distance)
        {
            let id = self.mobs[i].id as i32;
            self.send(S::Interact(play::Interact {
                id,
                hand: hand_id(hand),
                location,
                secondary: sneaking,
            }));
            // Player.interactOn: another player passes, and so does a mob
            // the held item means nothing to: the item is used instead.
            let player = self
                .remote
                .as_ref()
                .is_some_and(|r| r.level_players.contains_key(&id));
            if !player && self.remote_mob_takes(i, hand) {
                self.remote_feed_particles(i, hand);
                // As for the objects above, only SUCCESS swings the hand: a
                // CONSUME ends the click without one.
                if self.remote_mob_swings(i, hand) {
                    self.with_hand(hand, Game::swing_arm);
                }
                return true;
            }
        }
        false
    }

    /// `Minecraft.startUseItem`'s block turn for `hand`: a use that swung drops
    /// the hand (`LocalPlayer.itemUsed`) when it spent the held item or the
    /// player has infinite materials.
    fn remote_use_block(&mut self, hand: Hand) -> bool {
        let before = self.with_hand(hand, |g| g.held().map(|s| s.count));
        let swings = self.swings_started;
        let done = self.remote_use_item_on(hand);
        if done
            && self.swings_started != swings
            && let Some(after) = self.with_hand(hand, |g| g.held().map(|s| s.count))
            && (Some(after) != before || self.creative)
        {
            self.heights.item_used(hand);
        }
        done
    }

    /// MultiPlayerGameMode.useItemOn for `hand`: whether the click ends there
    /// (the block reacts, or a block item is placed).
    fn remote_use_item_on(&mut self, hand: Hand) -> bool {
        let held = self.with_hand(hand, |g| g.held().cloned());
        let held_block = held
            .as_ref()
            .and_then(|h| self.items().get(&h.item))
            .is_some_and(|i| i.block.is_some());
        let debug_stick = held
            .as_ref()
            .is_some_and(|stack| &*stack.item == "minecraft:debug_stick");
        if let Some(hit) = self.pick() {
            use mod_api::InteractionResult as Answer;
            use mod_api::TriState;
            let spectator = self.spectator();
            // RightClickBlock on this client, where performUseItemOn starts
            // (a spectator's use only opens menus on the server): it may force
            // the block's use or the item's useOn, or answer for both.
            let (mut canceled, mut use_block, mut use_item) =
                (None, TriState::Default, TriState::Default);
            if !spectator {
                match self.right_click_block(&hit, hand, held.as_ref()) {
                    super::native_hooks::BlockClickAnswer::Go {
                        use_block: block,
                        use_item: item,
                    } => (use_block, use_item) = (block, item),
                    super::native_hooks::BlockClickAnswer::Canceled(answer) => {
                        canceled = Some(answer);
                    }
                }
            }
            // The held item's onItemUseFirst on this client: the broadcast
            // event's ItemBeforeBlock phase answers for the block and the
            // item, as on the server (a Part's answer comes from the server).
            if canceled.is_none()
                && !spectator
                && use_item.or(true)
                && let Some(stack) = held.as_ref()
            {
                canceled = self.use_item_on_block(
                    &hit,
                    hand,
                    Some(stack),
                    mod_api::UsePhase::ItemBeforeBlock,
                );
            }
            // Player.isSecondaryUseActive with something in either hand
            // keeps the block from reacting.
            let sneaking = self.remote_shift_down();
            let hands_full = self.stack_in(self.selected).is_some() || self.offhand.is_some();
            let block_used = canceled.is_none() && use_block.or(!(sneaking && hands_full));
            let cooldown_forced = use_item == TriState::True;
            let item_used = canceled.is_none()
                && use_item.or(true)
                && held.as_ref().is_some_and(|stack| {
                    if !self.may_build()
                        && !self.with_hand(hand, |g| g.can_place_on_in_adventure(hit.pos))
                    {
                        return false;
                    }
                    self.items().get(&stack.item).is_some_and(|info| {
                        cooldown_forced
                            || !self
                                .cooldowns
                                .is_on_cooldown(&horizon_client::cooldowns::group(
                                    stack,
                                    info.use_cooldown.as_ref(),
                                ))
                    })
                });
            let block_answer = if !spectator && block_used {
                self.use_item_on_block(&hit, hand, held.as_ref(), mod_api::UsePhase::Block)
            } else {
                None
            };
            let vanilla_block = block_used && block_answer.is_none();
            let seq = self.next_sequence();
            // ShelfBlock.useItemOn consumes a nonempty main-hand click client-side
            // without predicting the inventory swap or swinging.
            let shelf_reacts = hand == Hand::Main
                && held.is_some()
                && vanilla_block
                && self.shelf_hit(&hit).is_some();
            let (result, changes) = if let Some(answer) = block_answer {
                if let Some(result) = super::native_hooks::block_use_result(answer) {
                    (result, DryUse::default())
                } else if hand == Hand::Main {
                    self.remote_block_template(&hit, held.as_ref(), true)
                } else {
                    (InteractionResult::Pass, DryUse::default())
                }
            } else if hand == Hand::Main && !spectator && vanilla_block {
                self.remote_block_reacts(&hit, held.as_ref())
            } else {
                // useItemOn runs for the off hand too (useWithoutItem doesn't):
                // RespawnAnchorBlock.useItemOn charges an anchor with its
                // glowstone, SUCCESS.
                let charges = vanilla_block && self.anchor_takes(hit.pos, held.as_ref());
                let result = if charges {
                    InteractionResult::Success
                } else {
                    InteractionResult::Pass
                };
                (result, DryUse::default())
            };
            // AbstractCauldronBlock.useItemOn, either hand: CauldronInteractions
            // answers on the client as on the server and only the server acts,
            // so a bucket the cauldron takes is not used again (UseItem).
            let result = if result.consumes_action() || spectator || !vanilla_block {
                result
            } else {
                match self.cauldron_outcome(hit.pos, held.as_ref()) {
                    horizon_client::cauldron::Outcome::Success => InteractionResult::Success,
                    horizon_client::cauldron::Outcome::Consume => InteractionResult::Consume,
                    horizon_client::cauldron::Outcome::Pass => result,
                }
            };
            // useItemOn on this client, unless a sneak with something in hand
            // skips it: RedStoneOreBlock's dust, and the wick smoke of a lit
            // candle an empty hand snuffs (CandleBlock.useItemOn) or a lit
            // candle cake (CandleCakeBlock.useItemOn).
            if hand == Hand::Main && !spectator && vanilla_block {
                self.redstone_ore_particles(hit.pos);
                let state = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
                let name = self.block_name(hit.pos);
                let lit =
                    self.pack.assets.registry.state(state).is_some_and(|s| {
                        s.properties.get("lit").map(String::as_str) == Some("true")
                    });
                let snuffs = if name.ends_with("candle_cake") {
                    hit.point[1] - f64::from(hit.pos[1]) > 0.5
                } else {
                    name.ends_with("candle") && self.may_build()
                };
                if held.is_none() && lit && snuffs {
                    self.candle_smoke(hit.pos, state);
                }
            }
            let result = if !spectator
                && vanilla_block
                && !result.consumes_action()
                && self.with_hand(hand, |g| g.predict_campfire_douse(&hit, held.as_ref(), seq))
            {
                InteractionResult::Success
            } else {
                result
            };
            let item_answer = if !spectator && !result.consumes_action() && item_used {
                self.use_item_on_block(&hit, hand, held.as_ref(), mod_api::UsePhase::ItemAfterBlock)
            } else {
                None
            };
            let item_used =
                item_used && !spectator && !result.consumes_action() && item_answer.is_none();
            // DebugStickItem.useOn once the block passed: SUCCESS on the client
            // too (the hand swings below); only the server changes the block.
            let (result, changes) = if debug_stick && item_used {
                (InteractionResult::Success, DryUse::default())
            } else {
                (result, changes)
            };
            let reacts = result.consumes_action();
            // ArmorStandItem.useOn runs in startPrediction before the packet
            // goes out (the server may spawn the stand at once).
            let stand_fits = item_used
                && held
                    .as_ref()
                    .is_some_and(|s| &*s.item == "minecraft:armor_stand")
                && held
                    .as_ref()
                    .is_some_and(|s| self.armor_stand_spawn(&hit, s).is_some());
            let cursor = [
                (hit.point[0] - f64::from(hit.pos[0])) as f32,
                (hit.point[1] - f64::from(hit.pos[1])) as f32,
                (hit.point[2] - f64::from(hit.pos[2])) as f32,
            ];
            self.send(S::UseItemOn(play::UseItemOn {
                hand: hand_id(hand),
                pos: hit.pos,
                face: i32::from(face_direction(hit.face)),
                cursor,
                inside: hit.inside,
                world_border: false,
                sequence: seq,
            }));
            // MultiPlayerGameMode.performUseItemOn: a spectator's use predicts
            // nothing and ends the click (CONSUME, no swing); the server opens
            // the block's menu.
            if spectator {
                return true;
            }
            // A cancelled RightClickBlock's answer, once the packet went out
            // (startPrediction sends it whatever the use): SUCCESS swings
            // here, and PASS goes on to useItem and the other hand.
            if let Some(answer) = canceled {
                if answer == Answer::Success {
                    self.with_hand(hand, Game::swing_arm);
                }
                return answer.consumes_action() || answer == Answer::Fail;
            }
            // Only a predicted placement (predict_place) marks a cell whose
            // resend plays a place sound: an empty hand or a non-block item
            // places nothing (BlockItem.place is where the sound comes from).
            if let Some(r) = self.remote.as_mut() {
                r.use_target = Some(hit.pos);
            }
            if let Some(answer) = item_answer {
                if answer == Answer::Success {
                    self.with_hand(hand, Game::swing_arm);
                }
                return answer.consumes_action() || answer == Answer::Fail;
            }
            if shelf_reacts {
                return true;
            }
            if reacts {
                self.predict_use(&changes, seq);
                // LeverBlock.useWithoutItem on the client: the predicted pull puffs.
                if vanilla_block || block_answer == Some(Answer::TryWithEmptyHand) {
                    if &*self.block_name(hit.pos) == "minecraft:lever" {
                        self.lever_puff(hit.pos);
                    }
                    self.open_command_block_if_editable((hit.pos[0], hit.pos[1], hit.pos[2]));
                    self.open_block_editor_if_editable((hit.pos[0], hit.pos[1], hit.pos[2]));
                }
                // MultiPlayerGameMode.useItemOn: SUCCESS SwingSource.CLIENT
                // (a chest, barrel, furnace, â€¦). CONSUME does not swing.
                if result.should_swing() && block_answer != Some(Answer::SuccessServer) {
                    self.with_hand(hand, Game::swing_arm);
                }
            } else if item_used && held.as_ref().is_some_and(|s| &*s.item == "minecraft:brush") {
                // BrushItem.useOn: brushing starts here (CONSUME, no swing); the
                // server brushes and the dust is this client's own.
                self.with_hand(hand, Game::start_using_item);
                return true;
            } else if item_used
                && let Some(stack) = held.as_ref().filter(|s| &*s.item == "minecraft:bone_meal")
                && self.with_hand(hand, |g| g.remote_bone_meal_succeeds(&hit, stack))
            {
                // BoneMealItem.useOn succeeded on this client's level (SUCCESS;
                // the server grows the plant): this hand swings and the click
                // ends, so the other hand (a torch) is never tried.
                self.with_hand(hand, Game::swing_arm);
                return true;
            } else if stand_fits {
                // ArmorStandItem.useOn on a client level: SUCCESS once the
                // stand fits, before the server spawns it. This hand swings
                // and the click ends.
                self.with_hand(hand, Game::swing_arm);
                return true;
            } else if let Some(stack) = held
                .as_ref()
                .filter(|_| item_used)
                .filter(|s| s.item.ends_with("_spawn_egg"))
            {
                // SpawnEggItem.useOn on a client level: FAIL for a mob that
                // cannot spawn (the click ends, no swing), else SUCCESS (swing).
                // Either way no UseItem follows, so the server does not spawn
                // a second mob from the fluid ray. ItemStack.useOn's adventure
                // predicate was checked before selecting this phase.
                if self.spawn_egg_allowed(&stack.item) {
                    self.with_hand(hand, Game::swing_arm);
                }
                return true;
            } else if let Some(stack) = held.as_ref().filter(|_| item_used)
                && &*stack.item == "minecraft:potion"
                && stack.potion() == Some("minecraft:water")
                && hit.face != [0, -1, 0]
                && self
                    .named_block_in_tag("minecraft:convertible_to_mud", &self.block_name(hit.pos))
                && (self.may_build()
                    || self.with_hand(hand, |g| g.can_place_on_in_adventure(hit.pos)))
            {
                // PotionItem.useOn on a client level: SUCCESS. The bottle is
                // poured out into the hand and the block turns to mud here at
                // once; the sounds (`playSound(null, ...)`) and the splash
                // particles (`sendParticles`) are the server's.
                if let Some(mud) = self.pack.assets.registry.default_state("minecraft:mud") {
                    let pos =
                        horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]);
                    self.predict_blocks(&vec![(pos, mud)], seq);
                }
                self.with_hand(hand, |g| {
                    g.filled_result(ItemStack::new("minecraft:glass_bottle", 1))
                });
                self.with_hand(hand, Game::swing_arm);
                return true;
            } else if let Some(stack) = held.as_ref().filter(|_| item_used)
                && (self.with_hand(hand, |g| g.predict_item_on_block(stack, &hit))
                    || self.predict_map_on_banner(stack, &hit))
            {
                // Item.useOn succeeded here (InteractionResult.SUCCESS):
                // this hand swings and the click ends, so the other hand
                // (a shield, a torch) is never tried.
                self.with_hand(hand, Game::swing_arm);
                return true;
            }
            if item_used && held_block {
                let consumable = held
                    .as_ref()
                    .and_then(|stack| self.items().get(&stack.item).cloned())
                    .filter(|info| info.consumable.is_some());
                let placed = if reacts {
                    None
                } else {
                    held.as_ref()
                        .and_then(|stack| self.remote_can_place(&hit, stack))
                };
                // BlockItem.useOn: a consumable that could not be placed is used
                // (Item.use), which the server does for this same packet. CONSUME
                // does not swing, and a full player's FAIL still ends the click.
                // Adventure's can_place_on leaves the placing to the server.
                if !reacts
                    && placed.is_none()
                    && self.may_build()
                    && let Some(info) = &consumable
                {
                    self.with_hand(hand, |g| g.start_eating(info));
                }
                if let Some(placed) = &placed {
                    self.predict_place(placed, seq);
                    if let Some(stack) = &held {
                        let pos = placed.0;
                        self.apply_predicted_item([pos.x, pos.y, pos.z], stack);
                    }
                    self.with_hand(hand, Game::consume_held);
                } else if let Some(r) = self.remote.as_mut() {
                    // BlockItem.place plays its sound only once the block is set: the
                    // server's resend of the target it refused is no placement.
                    r.placing = None;
                }
                if placed.is_some() {
                    self.with_hand(hand, Game::swing_arm);
                }
                // BlockItem.place returns FAIL when placement is rejected:
                // stop this click without trying the item or the other hand.
                return true;
            }
            return reacts;
        }
        false
    }

    /// `Item.useOn` on this client's level for the items whose effects are
    /// the clicker's own (`Level.levelEvent(player, ...)` and
    /// `playSound(player, ...)`: the server sends them to the others only):
    /// honeycomb waxing copper, and an axe, hoe or shovel transforming a
    /// block (stripping, scraping, removing wax, tilling, making a path).
    /// A cushion succeeds where it may be placed and spawns on the server, a
    /// firework rocket wherever it may be used, and the server launches it.
    /// Whether the item's use succeeded, which ends the click.
    /// The caller has selected Item.useOn after cooldown and adventure gates.
    fn predict_item_on_block(&mut self, stack: &ItemStack, hit: &super::Hit) -> bool {
        // FireworkRocketItem.useOn: SUCCESS unless gliding, when Item.use
        // boosts instead. The phase's cooldown/adventure gates already ran.
        if &*stack.item == "minecraft:firework_rocket" {
            return !self.player.as_ref().is_some_and(|p| p.fall_flying);
        }
        let at = [
            f64::from(hit.pos[0]) + 0.5,
            f64::from(hit.pos[1]) + 0.5,
            f64::from(hit.pos[2]) + 0.5,
        ];
        if &*stack.item == "minecraft:honeycomb" {
            // HoneycombItem.useOn.
            let registry = self.pack.assets.registry;
            let clicked = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
            let old = registry.state(clicked);
            let waxes = old
                .and_then(|old| self.res.transformers().waxed(registry, old))
                .is_some();
            if waxes {
                self.level_event_at(3003, hit.pos, 0);
                self.play_client_local_sound(
                    LevelSound::new("minecraft:item.honeycomb.wax_on", at, 1.0, 1.0)
                        .in_source(mod_api::SoundSource::Blocks),
                );
                // The other half of a double chest: its particles, and a second
                // sound from the clicked block.
                if let Some(other) =
                    old.and_then(|old| super::storage::chest_connected_pos(old, hit.pos))
                {
                    self.level_event_at(3003, other, 0);
                    self.play_client_local_sound(
                        LevelSound::new("minecraft:item.honeycomb.wax_on", at, 1.0, 1.0)
                            .in_source(mod_api::SoundSource::Blocks),
                    );
                }
            }
            waxes
        } else if let Some((_, sound, _, _, particle, _)) = self.block_transform(stack, hit) {
            self.transform_particle(&particle, hit.pos);
            self.play_client_local_sound(
                LevelSound::new(&sound, at, 1.0, 1.0).in_source(mod_api::SoundSource::Blocks),
            );
            true
        } else if self
            .res
            .items()
            .get(&stack.item)
            .is_some_and(|i| i.cushion_color.is_some())
        {
            // CushionItem.useOn on a client level: SUCCESS once
            // `canBePlacedAt` holds, before the server's own checks.
            self.cushion_spawn(hit).is_some()
        } else if &*stack.item == "minecraft:painting" {
            // HangingEntityItem.useOn on a client level: SUCCESS once
            // `Painting.create` finds a surviving variant, before the
            // server spawns it.
            self.painting_fits(hit, stack)
        } else {
            false
        }
    }

    /// `MapItem.useOn` on this client: a map used on a banner is SUCCESS
    /// whatever its data says (the server toggles the marker, or fails
    /// quietly), so this hand swings and the click ends. The caller applies
    /// ItemStack.useOn's adventure predicate before selecting this phase.
    fn predict_map_on_banner(&self, stack: &ItemStack, hit: &super::Hit) -> bool {
        super::maps::map_id(stack).is_some()
            && self.named_block_in_tag("minecraft:banners", &self.block_name(hit.pos))
    }

    /// `BoneMealItem.useOn` with the acting hand's `stack` on this client's
    /// level (a copy: nothing grows here): whether it succeeds.
    fn remote_bone_meal_succeeds(&mut self, hit: &super::Hit, stack: &ItemStack) -> bool {
        let pos = horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]);
        let face = horizon_core::geometry::Direction::ALL[face_direction(hit.face) as usize];
        let (ask, _) = self.held_bonemeal(pos, stack);
        self.with_level(|b, level, _| {
            b.bone_meal_succeeds_on_client(
                &mut DryLevel::new(level),
                pos,
                face,
                ask.crop,
                ask.check,
            )
        })
    }

    /// MultiPlayerGameMode.useItemOn inside startPrediction: the blocks the
    /// use changed on this client's level at once, each kept as predicted
    /// under the use's sequence until the server acknowledges it (a use the
    /// server refused goes back). Only the blocks a hand opens or flips: a
    /// wooden or copper door, trapdoor or fence gate, a lever or a button;
    /// whatever else the use changes waits for the server.
    fn predict_use(&mut self, (changes, sounds): &DryUse, seq: i32) {
        let facts = self.res.facts();
        let predicted = |state: BlockStateId| hand_operated(&facts.get(state).name);
        if changes.is_empty()
            || !changes
                .iter()
                .all(|(p, s)| predicted(*s) && predicted(self.world.block(p.x, p.y, p.z)))
        {
            return;
        }
        self.predict_blocks(changes, seq);
        // `level.playSound(player, ...)`: the clicker hears it from its own use; the
        // server sends it to the others only.
        for (pos, sound, volume, pitch) in sounds {
            let at = [
                f64::from(pos.x) + 0.5,
                f64::from(pos.y) + 0.5,
                f64::from(pos.z) + 0.5,
            ];
            let name = if sound.contains(':') {
                sound.to_string()
            } else {
                format!("minecraft:{sound}")
            };
            self.play_client_local_sound(
                LevelSound::new(&name, at, *volume, *pitch).in_source(mod_api::SoundSource::Blocks),
            );
        }
    }

    /// Another player of this level, whose copy passes every click (`Player.interactOn`).
    pub(super) fn is_level_player(&self, id: i32) -> bool {
        self.remote
            .as_ref()
            .is_some_and(|r| r.level_players.contains_key(&id))
    }

    /// Whether a main-hand right click with `held` would end on the block hit, for the gameplay
    /// guide: what `remote_block_reacts` answers on its dry level.
    pub(super) fn block_takes_click(&mut self, hit: &super::Hit, held: Option<&ItemStack>) -> bool {
        self.remote_block_reacts(hit, held).0.consumes_action()
    }
    /// `InteractionResult.CONSUME` for this click: the block keeps it and does nothing, so the
    /// item in hand never gets it either (an empty flower pot, a pressed button).
    pub(super) fn block_swallows_click(
        &mut self,
        hit: &super::Hit,
        held: Option<&ItemStack>,
    ) -> bool {
        matches!(
            self.remote_block_reacts(hit, held).0,
            InteractionResult::Consume
        )
    }

    /// Whether the block hit takes a main-hand right click with `held`
    /// (`BlockState.useItemOn`, then `useWithoutItem`), once the caller let
    /// the block react (no sneaking with full hands): a door, a button, a
    /// container, and the blocks it changed. Vanilla's client runs it on its
    /// own level first; this runs it on a copy, and `predict_use` keeps the
    /// changes it predicts.
    fn remote_block_reacts(
        &mut self,
        hit: &super::Hit,
        held: Option<&ItemStack>,
    ) -> (InteractionResult, DryUse) {
        // AbstractCauldronBlock.useItemOn (CauldronInteractions): a bucket,
        // bottle or dye the cauldron takes fills or lowers it, and the click
        // ends there instead of falling through to Item.use (which would
        // empty a water bucket into the face of the cauldron).
        if let Some(reacts) = self.remote_cauldron_reacts(hit, held) {
            return reacts;
        }
        if self.jukebox_takes(hit.pos, held) {
            return (InteractionResult::Success, DryUse::default());
        }
        // RespawnAnchorBlock.useItemOn: PASS when the off hand's glowstone is to
        // charge the anchor, so useWithoutItem doesn't run and the off hand's turn
        // comes (glowstone in this hand charges it in `use_item_on`).
        if !self.anchor_takes(hit.pos, held) && self.anchor_takes(hit.pos, self.offhand.as_ref()) {
            return (InteractionResult::Pass, DryUse::default());
        }
        // A content mod's machine takes the click where its server would (processing.rs): a
        // depot's top, a millstone with an empty hand. Nothing is placed against it.
        if self.machine_takes(hit.pos, hit.face, held) {
            return (InteractionResult::Success, DryUse::default());
        }
        // BeehiveBlock.useItemOn: shears or a glass bottle on a full hive
        // (honey level 5) is SUCCESS, so the click ends after UseItemOn and
        // no UseItem follows (the server does the harvest, see `use_beehive`).
        if held.is_some_and(|s| matches!(&*s.item, "minecraft:shears" | "minecraft:glass_bottle"))
            && self
                .hive_state(hit.pos)
                .is_some_and(|(honey, _)| honey == Some(5))
        {
            return (InteractionResult::Success, DryUse::default());
        }
        // ComposterBlock.useItemOn: a compostable item on a composter below
        // level 8 is SUCCESS on the client too (only the server adds the
        // layer), so the other hand (a torch) is never tried.
        if self.composter_takes(hit.pos, held) {
            return (InteractionResult::Success, DryUse::default());
        }
        self.remote_block_template(hit, held, false)
    }

    /// The shared dry-level boundary for a normal block use or the empty
    /// method requested by a canceled `BLOCK` phase's `TryWithEmptyHand`.
    pub(super) fn remote_block_template(
        &mut self,
        hit: &super::Hit,
        held: Option<&ItemStack>,
        without_item: bool,
    ) -> (InteractionResult, DryUse) {
        use horizon_core::geometry::Direction;
        let face = match hit.face {
            [0, 1, 0] => Direction::Up,
            [0, -1, 0] => Direction::Down,
            [0, 0, -1] => Direction::North,
            [0, 0, 1] => Direction::South,
            [-1, 0, 0] => Direction::West,
            _ => Direction::East,
        };
        let player = self.use_context();
        self.with_block_use_level(|b, level, _| {
            let mut level = DryLevel::new(level);
            let hit = horizon_core::geometry::BlockHit {
                block: horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]),
                direction: face,
                location: horizon_core::geometry::Vec3::new(
                    hit.point[0],
                    hit.point[1],
                    hit.point[2],
                ),
                inside: hit.inside,
            };
            let mut ticks = horizon_world::level::Ticks::default();
            let result = if without_item {
                use horizon_world::level::LevelAccess as _;
                let state = level.block_state(hit.block);
                b.use_without_item(&mut level, &mut ticks, state, hit, player)
            } else {
                b.use_item_on(&mut level, &mut ticks, hit, player, held.map(|s| &*s.item))
            };
            (result, (level.changed.into_iter().collect(), level.sounds))
        })
    }

    /// `AbstractCauldronBlock.useItemOn` (`CauldronInteractions`) on this
    /// client's level: the cauldron's new state and the sound it made, or
    /// None when the item passes and `Item.use` still has its turn. The
    /// interaction player is a copy: the hand and the inventory change only
    /// when the server confirms them.
    fn remote_cauldron_reacts(
        &self,
        hit: &super::Hit,
        held: Option<&ItemStack>,
    ) -> Option<(InteractionResult, DryUse)> {
        use horizon_client::cauldron::{self, Outcome};
        struct DryHand<'a> {
            creative: bool,
            hand: Option<ItemStack>,
            inventory: &'a [Option<ItemStack>],
        }
        impl cauldron::Player for DryHand<'_> {
            fn creative(&self) -> bool {
                self.creative
            }
            fn hand(&mut self) -> &mut Option<ItemStack> {
                &mut self.hand
            }
            fn add(&mut self, _: &mut ItemStack) -> bool {
                true
            }
            fn contains(&self, stack: &ItemStack) -> bool {
                self.inventory
                    .iter()
                    .flatten()
                    .any(|have| have.same_item_same_components(stack))
            }
            fn drop_stack(&mut self, _: ItemStack) {}
        }
        let registry = self.pack.assets.registry;
        let state = registry
            .state(self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]))
            .filter(|s| s.name.ends_with("cauldron"))?
            .clone();
        let held = held?;
        let above = self.world.block(hit.pos[0], hit.pos[1] + 1, hit.pos[2]);
        let under_water = registry
            .state(above)
            .is_some_and(|s| s.name == "minecraft:water");
        let removes_dye = cauldron::removes_dye(self.items(), &held.item);
        let mut hand = DryHand {
            creative: self.creative,
            hand: Some(held.clone()),
            inventory: &self.inventory,
        };
        let result = cauldron::interact(&state, under_water, removes_dye, &mut hand);
        let reaction = match result.outcome {
            Outcome::Pass => return None,
            Outcome::Success => InteractionResult::Success,
            Outcome::Consume => InteractionResult::Consume,
        };
        let pos = horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]);
        let mut changes = Vec::new();
        if let Some(block) = result.block
            && let Ok(new) = registry.resolve(&block)
        {
            changes.push((pos, new));
        }
        let sounds = match result.sound {
            Some(sound) => vec![(pos, std::sync::Arc::from(sound), 1.0, 1.0)],
            None => Vec::new(),
        };
        Some((reaction, (changes, sounds)))
    }

    /// MultiPlayerGameMode.useItemOn inside startPrediction, for a block item:
    /// the blocks `BlockItem.place` set at `pos` are placed on this client's
    /// level at once, so the next click of a fast bridge aims at them rather
    /// than past them (the server would refuse it). Its place sound is the
    /// placer's own (`playSound(player, ...)`: the server sends it to the
    /// others only).
    fn predict_place(&mut self, (pos, changes): &DryPlace, seq: i32) {
        let ticks = self.ticks;
        if let Some(r) = self.remote.as_mut() {
            // set_block plays the place sound of the block at `placing`.
            r.placing = Some(([pos.x, pos.y, pos.z], ticks));
        }
        self.predict_blocks(changes, seq);
    }
    /// The blocks a use changed, set on this client's level and kept as
    /// predicted under the use's sequence until the server acknowledges it
    /// (`BlockStatePredictionHandler.retainKnownServerState`): a use the
    /// server refused goes back.
    pub(super) fn predict_blocks(&mut self, changes: &DryChanges, seq: i32) {
        for &(p, state) in changes {
            let pos = [p.x, p.y, p.z];
            let before = self.world.block(p.x, p.y, p.z);
            let here = self.player_position();
            if let Some(r) = self.remote.as_mut() {
                r.predictions
                    .entry(pos)
                    .and_modify(|e| e.sequence = seq)
                    .or_insert(Prediction {
                        sequence: seq,
                        state: before.0 as i32,
                        player_pos: here,
                    });
            }
            self.set_block(pos, state.0 as i32);
        }
    }

    /// Run BlockItem.place on a disposable level: where it placed and the
    /// blocks it set. The real level supplies collision shapes and
    /// building-blocking entities; random draws, scheduled ticks and events
    /// stay in the preview.
    pub(super) fn remote_can_place(
        &mut self,
        hit: &super::Hit,
        stack: &ItemStack,
    ) -> Option<DryPlace> {
        // A crank takes the use (game/kinetics.rs): nothing is predicted against it.
        if !self.may_build() || self.kinetic_crank_at(hit.pos) {
            return None;
        }
        let ctx = horizon_world::level::PlaceContext {
            clicked_pos: horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]),
            face: horizon_core::geometry::Direction::ALL[face_direction(hit.face) as usize],
            hit: Vec3::new(hit.point[0], hit.point[1], hit.point[2]),
            inside: hit.face == [0; 3],
            yaw: self.interaction_rotation().0,
            pitch: self.interaction_rotation().1,
            sneaking: self.remote_shift_down(),
            item: stack.item.to_string(),
            can_use_game_master_blocks: self.can_use_game_master_blocks(),
        };
        self.with_level(|b, level, _| {
            let mut level = DryLevel::new(level);
            match b.place_block(
                &mut level,
                &mut horizon_world::level::Ticks::default(),
                &ctx,
            ) {
                horizon_world::level::PlaceOutcome::Placed { pos, .. } => {
                    Some((pos, level.changed.into_iter().collect()))
                }
                horizon_world::level::PlaceOutcome::Failed => None,
            }
        })
    }

    /// `MultiPlayerGameMode.useItem`'s local `ItemStack.use` for a bucket:
    /// `BucketItem.use` picks its own target and runs `pickupBlock` or
    /// `emptyContents` on this client's level inside `startPrediction`, so
    /// its success (which decides `itemUsed`) and the blocks it changes are
    /// the client's own. A fill into a waterloggable block changes nothing
    /// here: `SimpleWaterloggedBlock.placeLiquid` tests `isClientSide` and
    /// leaves the block update to the server.
    fn remote_use_bucket(
        &mut self,
        stack: &ItemStack,
    ) -> (bool, Option<DryChanges>, Vec<DrySound>) {
        // `BucketItem.use`: `level.mayInteract` and `mayUseItemAt`.
        // `Player.mayUseItemAt` tests the clicked block against `can_place_on`.
        if !self.may_build() {
            let aim = if &*stack.item == "minecraft:bucket" {
                self.bucket_hit()
            } else {
                self.pick()
            };
            if !aim.is_some_and(|hit| self.can_place_on_in_adventure(hit.pos)) {
                return (false, None, Vec::new());
            }
        }
        if &*stack.item == "minecraft:bucket" {
            let Some(hit) = self.bucket_hit() else {
                return (false, None, Vec::new());
            };
            let pos = horizon_core::position::BlockPos::new(hit.pos[0], hit.pos[1], hit.pos[2]);
            let creative = self.creative;
            return self.with_level(|b, level, ticks| {
                let mut level = DryLevel::new(level);
                let picked = b.bucket_pickup(&mut level, ticks, pos, creative).is_some();
                (
                    picked,
                    Some(level.changed.into_iter().collect()),
                    std::mem::take(&mut level.sounds),
                )
            });
        }
        let Some((content, sound, _)) = horizon_client::items::bucket_contents(&stack.item) else {
            return (false, None, Vec::new());
        };
        let Some(hit) = self.pick() else {
            return (false, None, Vec::new());
        };
        let target = self.bucket_target(content, &hit);
        let pos = horizon_core::position::BlockPos::new(target[0], target[1], target[2]);
        let [x, y, z] = hit.beside();
        let beside = Some(horizon_core::position::BlockPos::new(x, y, z));
        let sneaking = self.remote_shift_down();
        let creative = self.creative;
        self.with_level(|b, level, ticks| {
            let mut level = DryLevel::new(level);
            let emptied = b.bucket_empty(
                &mut level, ticks, pos, beside, content, sound, sneaking, creative,
            );
            // Water poured into a liquid container (where it went, which may
            // be the block beside a refusing one) fills it server-side only.
            let client_change = !(content.is_water()
                && emptied.is_some_and(|at| {
                    use horizon_world::level::LevelAccess as _;
                    b.is_liquid_container(level.block_state(at))
                }));
            (
                emptied.is_some(),
                client_change.then(|| level.changed.into_iter().collect()),
                std::mem::take(&mut level.sounds),
            )
        })
    }

    /// MultiPlayerGameMode.useItem for `hand`, if it holds something: whether
    /// the click ends there, as far as this client can tell (`use_ends_click`).
    fn remote_use_item(&mut self, hand: Hand) -> bool {
        let Some(held) = self.with_hand(hand, |g| g.held().cloned()) else {
            return false;
        };
        // A mod's claimed item is never used on the server; its own use
        // starts from the main hand only (`mod_use_start`).
        if self.native_input.claimant(&held.item).is_some() {
            return false;
        }
        let seq = self.next_sequence();
        self.send(S::UseItem(play::UseItem {
            hand: hand_id(hand),
            sequence: seq,
            yaw: self.yaw,
            pitch: self.pitch,
        }));
        let Some(info) = self.items().get(&held.item).cloned() else {
            return false;
        };
        // MultiPlayerGameMode.useItem's prediction: an item cooling down
        // passes (its packet still goes).
        let group = horizon_client::cooldowns::group(&held, info.use_cooldown.as_ref());
        if self.cooldowns.is_on_cooldown(&group) {
            return false;
        }
        if let Some(answer) = self.right_click_item(hand, &held, self.pick().is_some()) {
            let success = answer == mod_api::InteractionResult::Success;
            if success {
                self.with_hand(hand, Game::swing_arm);
            }
            return success;
        }
        // a mod's item used over time starts here too, as the server starts it.
        if self.with_hand(hand, |g| g.start_mod_use(&held)) {
            return true;
        }
        // Item.use: Equippable.swapWithEquipmentSlot runs on this client too, so
        // the armor slot and the hand change at once (the server confirms).
        if info.equip_slot.is_some() && self.with_hand(hand, |g| g.equip_from_hand(&held)) {
            // Equippable.swapWithEquipmentSlot's SUCCESS (SwingSource.CLIENT).
            self.with_hand(hand, Game::swing_arm);
            return true;
        }
        // `ItemStack.use` runs on this client too: a bucket's own ray and its
        // pick/empty contents decide whether the click succeeds (a failed
        // fill or pickup leaves the hand alone), and the blocks it sets are
        // this use's prediction.
        if &*held.item == "minecraft:bucket"
            || horizon_client::items::bucket_contents(&held.item).is_some()
        {
            let (succeeded, changes, sounds) = self.with_hand(hand, |g| g.remote_use_bucket(&held));
            if let Some(changes) = changes {
                self.predict_blocks(&changes, seq);
            }
            // `BucketItem.emptyContents` / evaporate: `playSound(player, â€¦)` on
            // this client; the server leaves the actor out (#4684).
            for (pos, sound, volume, pitch) in sounds {
                let at = [
                    f64::from(pos.x) + 0.5,
                    f64::from(pos.y) + 0.5,
                    f64::from(pos.z) + 0.5,
                ];
                let name = if sound.contains(':') {
                    sound.to_string()
                } else {
                    format!("minecraft:{sound}")
                };
                self.play_client_local_sound(
                    LevelSound::new(&name, at, volume, pitch)
                        .in_source(mod_api::SoundSource::Blocks),
                );
            }
            // BucketItem.use's SUCCESS (SwingSource.CLIENT) on this client.
            if succeeded {
                self.with_hand(hand, Game::swing_arm);
            }
            return succeeded;
        }
        // SpawnEggItem.use on the client succeeds after a SOURCE_ONLY block
        // hit; the server alone tests LiquidBlock and the placement permission.
        if held.item.ends_with("_spawn_egg") {
            let succeeded = self.spawn_egg_allowed(&held.item) && self.bucket_hit().is_some();
            if succeeded {
                self.with_hand(hand, Game::swing_arm);
            }
            return succeeded;
        }
        // KnowledgeBookItem.use on this client: the book is used up either
        // way. With recipes it is SUCCESS (swings, ends the click); with none
        // it is FAIL, and the other hand gets its turn. The recipes go as
        // their NBT on the wire, and the item's own default is an empty list,
        // so a stack whose patch sets them has some.
        if &*held.item == "minecraft:knowledge_book" {
            /// `DataComponents.RECIPES`' registry id (26.3).
            const RECIPES: i32 = 68;
            let has_recipes = held
                .wire_patch()
                .is_some_and(|p| horizon_protocol::item::WirePatch::adds(&p, RECIPES))
                || held
                    .data()
                    .and_then(|d| d.other("minecraft:recipes"))
                    .is_some_and(|t| matches!(t, Tag::List(l) if !l.is_empty()));
            self.with_hand(hand, Game::consume_held);
            if has_recipes {
                self.apply_use_cooldown(&held);
                self.with_hand(hand, Game::swing_arm);
            }
            return has_recipes;
        }
        // Food starts being eaten (Consumable.startConsuming). A food that
        // can't be eaten (full and not always-edible) fails the use
        // (InteractionResult.FAIL): startUseItem tries the other hand, and
        // no cooldown or use animation starts.
        let eating = self.with_hand(hand, |g| g.start_eating(&info));
        if info.food.is_some() && !eating {
            return false;
        }
        // A bucket, a glass bottle or a boat with nothing to fill, empty or
        // place PASSes or FAILs: startUseItem goes on to the off hand (a
        // Riptide trident there charges).
        if self.with_hand(hand, |g| g.use_finds_no_target(&held)) {
            return false;
        }
        // It runs the use here too: ItemStack.use starts the `use_cooldown`
        // of an item with no use time, and the server's
        // ClientboundCooldownPacket confirms it; one held in use starts it at
        // finishUsingItem (remote_complete_using).
        if !info.consumable.as_ref().is_some_and(|c| c.ticks() > 0) {
            self.apply_use_cooldown(&held);
        }
        if self.with_hand(hand, |g| g.remote_use_book(&held)) {
            // WritableBookItem.use's SUCCESS (SwingSource.CLIENT).
            self.with_hand(hand, Game::swing_arm);
            return true;
        }
        // BowItem.use and CrossbowItem.use FAIL with nothing to shoot: no
        // itemUsed (the hand doesn't dip) and the other hand's turn comes.
        if !self.with_hand(hand, |g| g.remote_start_held_use(&held)) {
            return false;
        }
        let gliding = self.player.as_ref().is_some_and(|p| p.fall_flying);
        // FireworkRocketItem.use while gliding: SUCCESS (SwingSource.PREDICTED),
        // so startUseItem swings this hand itself; a vanilla server swings it
        // for the other players only.
        if gliding && &*held.item == "minecraft:firework_rocket" {
            self.with_hand(hand, Game::swing_arm);
        }
        // A stack's own `blocks_attacks` raises it as a shield's does; a
        // use the server's patch gives it is CONSUME too (no swing).
        let blocks = self.blocks_attacks(&held).is_some() || self.patched_held_use(&held);
        let ends = use_ends_click(&info, gliding) || blocks;
        // MultiPlayerGameMode.useItem: `ItemStack.use`'s SUCCESS swings
        // (SwingSource.CLIENT); CONSUME does not â€” a bow, crossbow, trident
        // or spear starting its charge, and `ItemUtils.startUsingInstantly`'s
        // food, drink, spyglass, shield and goat horn.
        if ends && !blocks && remote_use_swings(&held, info.consumable.is_some()) {
            self.with_hand(hand, Game::swing_arm);
        }
        ends
    }

    /// Whether `held`'s `Item.use` needs a target this client sees none of
    /// (not SUCCESS): BucketItem.use (a fluid source or powder snow to pick
    /// up through `ClipContext.Fluid.SOURCE_ONLY`, a block to empty a full
    /// one against, `mayUseItemAt`), BottleItem.use (water to fill from) and
    /// BoatItem.use (a block or fluid surface with room for the boat).
    fn use_finds_no_target(&self, held: &ItemStack) -> bool {
        let id = &*held.item;
        let facts = |pos: [i32; 3]| {
            self.res
                .facts()
                .get(self.world.block(pos[0], pos[1], pos[2]))
        };
        let source = |want: Option<Fluid>| {
            let fluid_hit = self.pick_fluid();
            // pick_fluid steps through waterlogged blocks, but their water is
            // a source the SOURCE_ONLY ray stops at (and a BucketPickup).
            if let Some(hit) = self.pick()
                && facts(hit.pos).waterlogged
            {
                let eye = self.aim().map_or(hit.point, |(eye, _)| eye);
                let d = |p: [f64; 3]| (0..3).map(|a| (p[a] - eye[a]).powi(2)).sum::<f64>();
                if fluid_hit
                    .as_ref()
                    .is_none_or(|fluid| d(fluid.point) >= d(hit.point))
                {
                    return want.is_none_or(|want| want == Fluid::Water);
                }
            }
            fluid_hit.is_some_and(|hit| {
                let f = facts(hit.pos);
                let fluid = f.fluid != Fluid::None && f.fluid_level == 8 && !f.fluid_falling;
                match want {
                    Some(want) => fluid && f.fluid == want,
                    None => fluid || self.is_powder_snow(hit.pos),
                }
            })
        };
        // SolidBucketItem only has useOn: its use is Item.use's PASS.
        if id == "minecraft:powder_snow_bucket" {
            return true;
        }
        if id == "minecraft:bucket" {
            return !self.may_build() || !source(None);
        }
        if horizon_client::items::bucket_contents(id).is_some() {
            return !self.may_build() || self.pick().is_none();
        }
        if id == "minecraft:glass_bottle" {
            // A dragon's breath cloud never counts here: its owner is not
            // synced, so this client's BottleItem.use finds none and PASSes;
            // the packet still goes and the server fills the bottle.
            return !source(Some(Fluid::Water));
        }
        if super::boats::any_boat_wood(id).is_some() {
            return self.boat_to_place(held).is_none();
        }
        false
    }
    /// `Item.use` runs on the client too (MultiPlayerGameMode.useItem): an
    /// item held in use until the button lets go starts being used here, so
    /// that letting go sends RELEASE_USE_ITEM (Minecraft.handleKeybinds) and
    /// the server ends the use: a bow or a trident is let go, a crossbow's
    /// charge ends, a shield is lowered, a spyglass stops scoping. Their
    /// sounds, loaded arrows and thrown tridents come from the server.
    /// False when the use FAILs here: a bow or an uncharged crossbow with
    /// nothing to shoot.
    fn remote_start_held_use(&mut self, held: &ItemStack) -> bool {
        if self.using.is_some() {
            return true;
        }
        let bundle = self.items().get(&held.item).is_some_and(|i| i.bundle);
        let start = match &*held.item {
            // BundleItem.use: always starts; the server drops the contents.
            _ if bundle => true,
            // BowItem.use: with an arrow, or in creative.
            "minecraft:bow" => self.get_projectile(held).is_some() || self.creative,
            // CrossbowItem.use: charging when not charged and with ammunition;
            // a charged one shoots on the server.
            "minecraft:crossbow" => {
                if held.data().is_some_and(|d| !d.charged.is_empty()) {
                    return true;
                }
                self.get_projectile(held).is_some()
            }
            // TridentItem.use: not about to break, Riptide only in water or rain.
            "minecraft:trident" => {
                // FAIL: no dip, and the off hand gets its turn.
                return self.use_trident(held);
            }
            // SpyglassItem.use (its sound is the server's).
            "minecraft:spyglass" => true,
            // InstrumentItem.use: the horn plays here too (`playSound(player, ...)`
            // skips the player's own client), the server's copy tells the others.
            "minecraft:goat_horn" => {
                self.remote_use_instrument(held);
                return true;
            }
            // EnderEyeItem.use: startUsingItem unless an End portal frame is
            // aimed at, with or without a stronghold to throw the eye at.
            "minecraft:ender_eye" => !self.aims_at_end_portal_frame(),
            // Item.use: a kinetic weapon starts being used.
            id if horizon_client::spear::Kinetic::of(id).is_some() => true,
            // ShieldItem.use: anything that blocks attacks, off cooldown.
            // Item.use: a consumable or `blocks_attacks` the server's patch
            // adds starts the use (Consumable.startConsuming, startUsingItem).
            _ => {
                if !self.raise_shield(self.held_slot()) && self.patched_held_use(held) {
                    self.start_using_item();
                }
                return true;
            }
        };
        if start {
            self.start_using_item();
        }
        start || !matches!(&*held.item, "minecraft:bow" | "minecraft:crossbow")
    }
    /// `InstrumentItem.use` on this client: unless the horn is on cooldown,
    /// it is held up, plays its sound at the player and starts its cooldown.
    fn remote_use_instrument(&mut self, held: &ItemStack) {
        let Some(instrument) = self.res.items().instrument(held) else {
            return;
        };
        let group = horizon_client::cooldowns::group(held, None);
        if self.cooldowns.is_on_cooldown(&group) {
            return;
        }
        self.start_using_item();
        self.play_player_bound_sound(&instrument.sound, 2, instrument.range / 16.0, 1.0, true);
        if instrument.use_duration > 0.0 {
            self.add_cooldown(group, instrument.use_ticks());
        }
    }
    /// Whether `stack` stays in use until the button lets go (bows,
    /// crossbows, tridents, spyglasses, shields, eyes of ender, whose
    /// `getUseDuration` 0 never counts down to the end) or the server ends
    /// it (a mod's item): its use only counts ticks here.
    fn held_until_released(&self, stack: &ItemStack) -> bool {
        if self.mod_use(stack) {
            return true;
        }
        matches!(
            &*stack.item,
            "minecraft:bow"
                | "minecraft:crossbow"
                | "minecraft:trident"
                | "minecraft:spyglass"
                | "minecraft:ender_eye"
        ) || horizon_client::spear::Kinetic::of(&stack.item).is_some()
            || self.blocks_attacks(stack).is_some()
            || self.patched_held_use(stack)
    }
    /// Whether the server's patch gives `stack`, an item with no consumable
    /// of its own, a `consumable` with a use time or `blocks_attacks` (a
    /// sword made to block): `Item.use` starts a use of it
    /// (`Consumable.startConsuming`, `startUsingItem`) that lasts until the
    /// button lets go or the server stops it, as the client's
    /// `LivingEntity.updateUsingItem` never completes one.
    fn patched_held_use(&self, stack: &ItemStack) -> bool {
        let (ticks, blocks) = horizon_client::item_sync::patched_use(stack);
        ticks.map_or(blocks, |ticks| ticks > 0)
            && self
                .items()
                .get(&stack.item)
                .is_none_or(|i| i.consumable.is_none())
    }

    /// The rest of Minecraft.startUseItem once the main hand's target passed:
    /// the main hand's item, then the off hand's turn (its block, its item).
    /// The server does each packet's part with that packet's hand only.
    fn remote_use_hands(&mut self, attempt: mod_api::InteractionAttempt) {
        #[cfg(target_os = "android")]
        if self.mobile.touch_shield {
            let hand = if self
                .offhand
                .as_ref()
                .is_some_and(|s| self.blocks_attacks(s).is_some())
            {
                Hand::Off
            } else {
                Hand::Main
            };
            if self.remote_use_item(hand) {
                self.heights.item_used(hand);
            }
            return;
        }
        // Nothing in the crosshair: an empty hand's click is heard, by mods only.
        let miss = mod_api::__private::listened(mod_api::HookId::RightClickEmpty)
            && !self.entity_targeted()
            && self.pick().is_none();
        if miss && self.held().is_none() {
            self.native_right_click_empty(Hand::Main);
        }
        // A successful Item.use always drops the hand (LocalPlayer.itemUsed).
        if self.remote_use_item(Hand::Main) {
            self.heights.item_used(Hand::Main);
            return;
        }
        // The off hand's turn: the event again, then the entity or the block.
        if let Some(swing) = self.native_interaction(InteractionKey::Use, Hand::Off, attempt) {
            if swing {
                self.swing_hand(Hand::Off, SwingAnimation::DEFAULT);
            }
            return;
        }
        let off_target = if self.remote_use_targets_entity() {
            self.remote_use_entity(Hand::Off)
        } else {
            self.remote_use_block(Hand::Off)
        };
        if off_target {
            return;
        }
        if miss && self.offhand.is_none() {
            self.native_right_click_empty(Hand::Off);
        }
        if self.remote_use_item(Hand::Off) {
            self.heights.item_used(Hand::Off);
        }
    }
    /// `LivingEntity.updatingUsingItem` on a server client: an item held in
    /// use (`held_until_released`), and a consumable's eating animation and
    /// its crumbs until the server's USE_ITEM_COMPLETE. The server sends the
    /// stack, its hunger and the sounds.
    fn remote_tick_using(&mut self) {
        let Some((slot, ticks)) = self.using else {
            // A mod's use the game reset (the server stopping the use...).
            self.end_mod_use();
            return;
        };
        let stack = self.stack_in(slot).cloned();
        // The use stops when the hand no longer holds the item it began with
        // (ItemStack.isSameItem).
        let item = stack.as_ref().map(|s| Arc::clone(&s.item));
        if let Some(r) = self.remote.as_mut() {
            if ticks == 0 {
                r.use_item = item;
            } else if r.use_item != item {
                self.using = None;
                self.end_mod_use();
                return;
            }
        }
        // Another hotbar slot picked: LivingEntity.updatingUsingItem sees a
        // different stack in the hand. The off hand keeps its item.
        let in_hand = slot == self.selected || slot == super::shield::OFFHAND_SLOT;
        // A mod's item: held until the button lets go, counted for the mod.
        if self.mod_use_tick(slot, ticks, !self.right || !in_hand) {
            return;
        }
        // updatingUsingItem runs here too: a use a mod's tick ran out stays
        // frozen until let go (RELEASE_USE_ITEM) or the server finishes it.
        if let Some(s) = stack.as_ref().filter(|_| in_hand)
            && !self.use_item_ticked(s, ticks)
        {
            return;
        }
        // BrushItem.onUseTick: with no block aimed the use ends (the server
        // ends it too); a stroke shows its dust. Its sounds are the server's.
        if stack
            .as_ref()
            .is_some_and(|s| &*s.item == "minecraft:brush")
        {
            let elapsed = ticks + 1;
            let hit = self.pick().filter(|_| in_hand);
            if let Some(hit) = &hit
                && elapsed % 10 == 5
            {
                self.brush_dust(hit, slot == super::shield::OFFHAND_SLOT);
            }
            if hit.is_none() && in_hand {
                self.send(S::PlayerAction(play::PlayerAction {
                    action: 6,
                    pos: [0; 3],
                    direction: 0,
                    sequence: 0,
                }));
            }
            self.using = hit.and(Some((slot, elapsed))).filter(|_| elapsed < 200);
            return;
        }
        // BundleItem.getUseDuration: 200 ticks, the server drops the contents.
        if stack
            .as_ref()
            .is_some_and(|s| self.items().get(&s.item).is_some_and(|i| i.bundle))
        {
            self.using = (in_hand && ticks + 1 < 200).then_some((slot, ticks + 1));
            return;
        }
        // A drawn bow, a charging crossbow, a raised shield...: held until
        // the button lets go.
        if let Some(s) = stack.as_ref().filter(|s| self.held_until_released(s)) {
            if in_hand && (self.mod_use(s) || self.blocks_attacks(s).is_some()) {
                self.item_on_use_tick(s, ticks);
            }
            self.using = in_hand.then_some((slot, ticks + 1));
            return;
        }
        // A goat horn is held up for its instrument's use duration.
        if let Some(duration) = stack
            .as_ref()
            .and_then(|s| self.res.items().instrument(s))
            .map(|i| i.use_ticks().max(0) as u32)
        {
            self.using = (in_hand && ticks + 1 < duration).then_some((slot, ticks + 1));
            return;
        }
        let consumable = stack
            .as_ref()
            .and_then(|s| self.items().get(&s.item))
            .and_then(|i| i.consumable.clone());
        let (Some(stack), Some(consumable)) = (stack, consumable) else {
            self.using = None;
            return;
        };
        // Button release belongs to the next handleKeybinds phase, which
        // sends RELEASE_USE_ITEM. Do not silently clear a use begun by a
        // quick press/release between ticks.
        if !in_hand {
            self.using = None;
            return;
        }
        // Consumable.startConsuming eats one with no use time at once, without
        // a use (nor a USE_ITEM_COMPLETE).
        if consumable.ticks() <= 0 {
            self.using = None;
            return;
        }
        // Consumable.shouldEmitParticlesAndSounds, as in tick_used_item. The
        // countdown goes on past zero (LivingEntity.updateUsingItem completes
        // only on the server) until the server's USE_ITEM_COMPLETE.
        // getUseItemRemainingTicks: the duration this use started with.
        let remaining = self.use_duration - ticks as i32;
        if consumable.should_emit(remaining) {
            // `emit_consume` plays the eat sound and spawns the bites (when the
            // item has consume particles), as `tick_used_item` does on the
            // player's own server.
            self.emit_consume(&consumable, &stack, 5);
        }
        self.item_on_use_tick(&stack, ticks);
        self.using = Some((slot, ticks + 1));
    }
    /// `Player.handleEntityEvent(9)`: `completeUsingItem` on the client. A consumable's last crumbs, and FoodProperties.onConsume,
    /// which runs on the client too (Consumable.onConsume calls its
    /// ConsumableListeners on both sides): the client's FoodData eats at once,
    /// so a held right click finds `Player.canEat` false instead of starting
    /// another bite before the server's ClientboundSetHealth arrives.
    fn remote_complete_using(&mut self) {
        let Some((slot, _)) = self.using.take() else {
            return;
        };
        let Some(stack) = self.stack_in(slot).cloned() else {
            return;
        };
        let Some(info) = self.items().get(&stack.item).cloned() else {
            return;
        };
        // Consumable.onConsume's emitParticlesAndSounds(16): the last crumbs
        // and bite, which the server plays for the others only.
        if let Some(consumable) = &info.consumable {
            self.emit_consume(consumable, &stack, 16);
        }
        if let Some((nutrition, saturation)) = info.food
            && let Some(r) = self.remote.as_mut()
        {
            // FoodData.eat(nutrition, saturation), as finish_consuming does;
            // the next ClientboundSetHealth replaces it.
            let (_, food, sat) = &mut r.health;
            *food = (*food + nutrition as i32).clamp(0, 20);
            *sat = (*sat + saturation).clamp(0.0, *food as f32);
            self.apply_remote_health();
        }
        // ItemStack.finishUsingItem's applyAfterUseComponentSideEffects.
        self.apply_use_cooldown(&stack);
    }
    /// MouseHandler updates KeyMapping; handleKeybinds sends packets on a tick.
    pub(super) fn remote_mouse(&mut self, button: MouseButton, pressed: bool) -> Result<()> {
        use horizon_client::interaction_input::Button;
        let input = match button {
            MouseButton::Left => {
                self.toggles
                    .button(2, pressed, false, self.settings.toggle_attack);
                self.left = self.toggles.is_down(2);
                Some(Button::Attack)
            }
            MouseButton::Right => {
                self.toggles
                    .button(3, pressed, false, self.settings.toggle_use);
                self.right = self.toggles.is_down(3);
                Some(Button::Use)
            }
            MouseButton::Middle => Some(Button::Pick),
            _ => None,
        };
        if pressed
            && let Some(button) = input
            && let Some(remote) = self.remote.as_mut()
        {
            remote.input.press(button);
        }
        Ok(())
    }

    pub(super) fn request_remote_mining_stop(&mut self) {
        if let Some(r) = self.remote.as_mut() {
            r.stop_mining = true;
        }
    }

    fn remote_handle_input(&mut self) {
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let mut input = std::mem::take(&mut r.input);
        let stop = std::mem::take(&mut r.stop_mining);
        // Minecraft.tick runs handleKeybinds only while `gui.screen() == null`: a
        // book, a sign editor, a dialog or a prompt over the world (each a
        // `Screen` in vanilla) stops the attack and use mappings from acting,
        // and `KeyMapping.releaseAll` let go of the buttons when it opened
        // (`track_toggle_screens`), so a dig held into a book ends (#9729).
        let ready = r.loaded_sent
            && self.screen == Screen::Playing
            && self.sleeping.is_none()
            && !self.screen_over_playing();
        // MultiPlayerGameMode.tick, first in Minecraft.tick, screen or not: the slot
        // chosen since the last tick (the wheel, the last tick's number keys, the
        // server's SetHeldSlot) goes out before anything else of this tick.
        self.remote_send_carried();
        if stop || !ready {
            self.stop_remote_mining();
        }
        if !ready {
            // A screen's opening released the keys (`KeyMapping.releaseAll`).
            self.hotbar_presses.clear();
            return;
        }
        let (attack, use_item) = (self.left, self.right);
        // handleKeybinds: keySwapOffhand.consumeClick, before attacking and using (an
        // anticheat's Post check wants it before the tick's move, not whenever the key went down).
        let swaps = self
            .remote
            .as_mut()
            .map_or(0, |r| std::mem::take(&mut r.swap_presses));
        if !self.spectator() {
            for _ in 0..swaps {
                self.remote_swap_hands();
            }
        }
        input.tick(&mut RemoteInputWorld(self), attack, use_item);
        self.remote_queued_keys();
        if let Some(r) = self.remote.as_mut() {
            r.input = input;
        }
    }

    /// `MultiPlayerGameMode.ensureHasSentCarriedItem`: the server hears of a
    /// new selected slot before anything the slot acts with.
    fn remote_send_carried(&self) {
        if let Some(r) = &self.remote
            && r.carried.get() != Some(self.selected)
        {
            r.carried.set(Some(self.selected));
            self.send_play(S::SetCarriedItem(play::SetCarriedItem {
                slot: self.selected as i16,
            }));
        }
    }

    fn remote_queued_keys(&mut self) {
        // handleKeybinds' keybind polls: the queued presses act now, in tick
        // order, so no packet goes out between CLIENT_TICK_END and a pong.
        let pending = self
            .remote
            .as_mut()
            .map(|r| std::mem::take(&mut r.pending_keys))
            .unwrap_or_default();
        for key in pending {
            match key {
                KeyCode::KeyF if !self.spectator() => self.remote_swap_hands(),
                KeyCode::KeyQ if !self.spectator() => {
                    // Minecraft.hasControlDown: the physical keys, so a toggled
                    // sprint on Left Control does not make Q drop the stack.
                    let all = self.has_control_down() || self.input_takes_stack_drop();
                    self.remote_drop(all);
                }
                KeyCode::KeyE
                    if !self.in_portal_transition()
                        && (self.riding_mount_with_menu() || self.riding_chest_boat()) =>
                {
                    let id = self.remote.as_ref().map_or(0, |r| r.entity_id);
                    self.send(S::PlayerCommand(play::PlayerCommand {
                        id,
                        action: 5,
                        data: 0,
                    }));
                }
                _ => {}
            }
        }
        if self.clicks.miss_time > 0 {
            self.clicks.miss_time -= 1;
        }
    }

    pub(in crate::game) fn remote_start_attack(&mut self) -> bool {
        if self.clicks.miss_time > 0 || self.hands_busy() {
            return false;
        }
        if self.spectator() {
            // Minecraft.startAttack: a spectator's click on an entity asks
            // to watch through it (MultiPlayerGameMode.spectate); a click on
            // nothing still tells the server (spectatorNoAction).
            let target = self.spectator_click_target();
            self.send(S::SpectatorAction(play::SpectatorAction { target }));
            return true;
        }
        let spear = self
            .held()
            .and_then(|s| horizon_client::spear::Kinetic::of(&s.item));
        if spear.is_some() && self.cannot_attack_with_item(self.hand_strength.ticker, 0) {
            return false;
        }
        // InputEvent.InteractionKeyMappingTriggered, after the checks that stop
        // an attack: a cancel ends it there, swinging unless the mod said not to.
        if let Some(swing) = self.native_interaction(
            InteractionKey::Attack,
            Hand::Main,
            mod_api::InteractionAttempt::Press,
        ) {
            if swing {
                let animation = spear.map_or(SwingAnimation::DEFAULT, |k| k.attack_animation());
                self.swing_hand(Hand::Main, animation);
            }
            return false;
        }
        // A spear jabs (`MultiPlayerGameMode.piercingAttack`): the server
        // stabs, this client swings, plays the sound and starts its charge
        // again.
        if let Some(kinetic) = spear {
            // ServerboundPlayerActionPacket.Action.STAB.
            self.send(S::PlayerAction(play::PlayerAction {
                action: 8,
                pos: [0; 3],
                direction: 0,
                sequence: 0,
            }));
            self.hand_strength.reset_attack();
            self.swing_hand(Hand::Main, kinetic.attack_animation());
            self.play_at_feet(kinetic.attack_sound);
            return true;
        }
        if self.mod_attack_targets() {
            self.swing_arm();
            self.send(S::Punch(Empty));
            return false;
        }
        let mut instant = false;
        // A ghast's fireball or a wind charge nearer than any mob: the
        // server turns it (game/items/deflect.rs).
        if let Some((i, distance)) = self.redirectable_hit()
            && self.mob_hit().is_none_or(|(_, d, _)| distance <= d)
            && self.before_block(distance)
            && self.creatures.projectiles[i].id != 0
        {
            self.remote_attack_entity(self.creatures.projectiles[i].id as i32, distance);
        } else if let Some(hit) = self.remote_object_hit(true)
            && self.mob_hit().is_none_or(|(_, d, _)| hit.distance < d)
            && self.before_block(hit.distance)
        {
            self.remote_attack_entity(hit.id, hit.distance);
        } else if let Some((i, distance, _)) = self.mob_hit()
            && self.before_block(distance)
        {
            self.remote_attack_entity(self.mobs[i].id as i32, distance);
        } else if !self.entity_targeted()
            && let Some(hit) = self.pick()
        {
            // A block hit on air falls through to MISS. `instant` below reads the
            // block again after the dig, which may have broken it.
            if mod_api::__private::listened(mod_api::HookId::LeftClickEmpty)
                && self
                    .res
                    .facts()
                    .is_air(self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]))
            {
                self.native_left_click_empty();
            }
            if !self.remote_mining(true) {
                return false;
            }
            instant = self
                .res
                .facts()
                .is_air(self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]));
        } else {
            if mod_api::__private::listened(mod_api::HookId::LeftClickEmpty)
                && !self.entity_targeted()
            {
                self.native_left_click_empty();
            }
            if !self.creative {
                self.clicks.miss_time = 10;
            }
            self.reset_attack_strength();
        }
        self.remote_swing_packet();
        instant
    }

    /// `MultiPlayerGameMode.attack` runs `Player.attack` on this client too: its
    /// `isSweepAttack` for a damaging hit on `target` (`living`: a
    /// `LivingEntity`, which may be hit critically), heard by native mods on
    /// both sides. Only the server sweeps.
    fn remote_sweep_event(&mut self, target: &str, living: bool) {
        if mod_api::__private::mods().is_empty() {
            return;
        }
        let speed = self
            .own_attribute("minecraft:attack_speed")
            .unwrap_or_else(|| self.main_hand_attack_speed());
        let scale = self.hand_strength.scale(0.5, speed);
        let spin = self.player.as_ref().map_or(0.0, |p| p.auto_spin_attack_dmg);
        let damage = if spin > 0.0 {
            spin
        } else {
            self.own_attribute("minecraft:attack_damage")
                .map_or(1.0, |v| v as f32)
        };
        let magic = scale * (self.enchanted_damage(target, damage, None) - damage);
        if damage <= 0.0 && magic <= 0.0 {
            return;
        }
        let crit = living && self.critical_attack(scale);
        self.sweep_decision(scale, crit, target);
    }
    /// The entity type of the boat, minecart, armor stand or End crystal with
    /// server id `id` that an attack reaches `Player.attack`'s sweep check on,
    /// and whether it is living. A painting or item frame is not: on this
    /// client `skipAttackInteraction` hurts it and stops the attack there.
    fn remote_attacked_object(&self, id: i32) -> Option<(String, bool)> {
        let id = id as u32;
        if let Some(b) = self.boats.iter().find(|b| b.id == id) {
            return Some((b.entity_type(), false));
        }
        if let Some(c) = self.carts.iter().find(|c| c.id == id) {
            return Some((c.kind.id().to_owned(), false));
        }
        if self.armor_stands.iter().any(|a| a.id == id) {
            return Some(("minecraft:armor_stand".to_owned(), true));
        }
        self.crystals
            .iter()
            .any(|c| c.id == id)
            .then(|| ("minecraft:end_crystal".to_owned(), false))
    }
    /// `MultiPlayerGameMode.attack` runs `Player.attack` on this client too.
    /// For a mob `hurtOrSimulate` fails here (`Entity.hurtClient`), so the
    /// hit has no attacker side; but `RemotePlayer.hurtClient` is true, so a
    /// hit on another player with extra knockback runs the attacker's side of
    /// `causeExtraKnockback`: the 0.6 horizontal speed cut and the sprint
    /// stop. Without it the next move outruns the server's player.
    fn remote_player_attack(&mut self, mob: usize) {
        // Java dispatches this through the target's RemotePlayer type. The
        // level_players map is profile bookkeeping and can lag the entity;
        // mobs must stay out because their hurtClient returns false.
        if self.mobs[mob].kind.id != "minecraft:player" {
            return;
        }
        // `getAttackStrengthScale(0.5)`: here the ticker is `hand_strength`'s
        // (`attack_ticks` only counts in the local world), and the attack
        // speed is the player's synced attribute, which a server may change.
        let speed = self
            .own_attribute("minecraft:attack_speed")
            .unwrap_or_else(|| self.main_hand_attack_speed());
        let strength = self.hand_strength.scale(0.5, speed);
        // Player.attack's damage: a riptide spin's, or this client's own
        // ATTACK_DAMAGE, which the held item and effects don't change here.
        let spin = self.player.as_ref().map_or(0.0, |p| p.auto_spin_attack_dmg);
        let damage = if spin > 0.0 {
            spin
        } else {
            self.own_attribute("minecraft:attack_damage")
                .map_or(1.0, |v| v as f32)
        };
        let knockback = self
            .own_attribute("minecraft:attack_knockback")
            .map_or(0.0, |v| v as f32);
        let sprinting = self.player.as_ref().is_some_and(|p| p.sprinting);
        // attackVisualEffects: a RemotePlayer's hurtClient is true, so a
        // critical hit (full strength, canCriticalAttack) with damage left
        // after baseDamageScaleFactor runs LocalPlayer.crit, this client's
        // own CRIT emitter beside the one the server's Animate brings.
        // magicBoost is 0 on a client, so no ENCHANTED_HIT one.
        if damage * (0.2 + strength * strength * 0.8) > 0.0 && self.critical_attack(strength) {
            use horizon_client::particles::{ParticleOptions, ParticleType};
            let m = &self.mobs[mob];
            self.particles.tracking_emitter(
                ParticleOptions::simple(ParticleType::Crit),
                TrackingTarget::Entity(m.id as i32),
                mob_tracking_body(m),
            );
        }
        if !horizon_client::attack_strength::attacker_slows_after_hit(
            damage, knockback, strength, sprinting,
        ) {
            return;
        }
        if let Some(p) = self.player.as_mut() {
            let slowdown = horizon_client::attack_strength::ATTACKER_SLOWDOWN;
            p.velocity = Vec3::new(
                p.velocity.x * slowdown,
                p.velocity.y,
                p.velocity.z * slowdown,
            );
        }
        self.stop_sprint_after_extra_knockback();
    }
    /// The latency the server's player list gives this player
    /// (`PlayerInfo.getLatency`). A new entry reads 0 until the server's first
    /// latency update (every 600 ticks), so 0 shows nothing rather than `0 ms`.
    pub(super) fn own_latency(&self) -> Option<i32> {
        let r = self.remote.as_ref()?;
        let me = r.tab_players.get(&r.profile)?;
        (me.latency > 0).then_some(me.latency)
    }

    /// An attribute's value on this client's player, if its map has the attribute.
    fn own_attribute(&mut self, id: &str) -> Option<f64> {
        let p = self.player.as_mut()?;
        let key = p.attributes.key(id)?;
        p.attributes.lookup(Some(&key))?;
        p.attributes
            .value(Some(key), horizon_world::attribute_map::ValueKind::Computed)
            .ok()
    }

    fn remote_continue_attack(&mut self, down: bool) {
        if !down {
            self.clicks.miss_time = 0;
        }
        // A piercing weapon neither breaks blocks nor swings on its own.
        if self.clicks.miss_time > 0
            || self.using.is_some()
            || self
                .held()
                .is_some_and(|s| horizon_client::spear::Kinetic::of(&s.item).is_some())
        {
            return;
        }
        let entity = self.entity_targeted();
        if down
            && !entity
            && let Some(hit) = self.pick()
        {
            let state = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
            let air = self.res.facts().is_air(state);
            // InputEvent.InteractionKeyMappingTriggered, before the dig goes on:
            // a cancel digs nothing and swings unless the mod said not to.
            let canceled = if air {
                None
            } else {
                self.native_interaction(
                    InteractionKey::Attack,
                    Hand::Main,
                    mod_api::InteractionAttempt::Held,
                )
            };
            if canceled.is_none() && !air {
                // MultiPlayerGameMode.continueDestroyBlock starts with ensureHasSentCarriedItem.
                self.remote_send_carried();
            }
            if let Some(swing) = canceled {
                if swing {
                    self.remote_swing_packet();
                }
            } else if self.remote_break_restricted(hit.pos) {
                // continueDestroyBlock returned false: no swing.
            } else if !air {
                if self.remote_mining(false) {
                    self.remote_swing_packet();
                }
            } else {
                if let Some(r) = self.remote.as_mut() {
                    r.destroying = None;
                    r.is_destroying = false;
                }
                self.mining = None;
            }
        } else {
            // `stopDestroyBlock` returns at once for a break that is not running: a pick that
            // missed once a break stopped ends nothing, so the dig's accumulated progress stays
            // and goes on growing on the jumps that follow (#4124).
            if self.stop_remote_mining() {
                self.mining = None;
            }
        }
    }
    /// Keys with a multiplayer meaning; returns true when consumed.
    pub(super) fn remote_key(&mut self, key: KeyCode) -> Result<bool> {
        if self.screen != Screen::Playing {
            if (key == KeyCode::KeyE || key == KeyCode::Escape) && self.screen == Screen::Inventory
            {
                self.remote_close_container();
                self.set_screen(Screen::Playing);
                self.capture(true);
                return Ok(true);
            }
            return Ok(false);
        }
        match key {
            KeyCode::KeyT | KeyCode::Slash | KeyCode::Enter => {
                // ChatScreen, over the server's command tree.
                self.open_chat_method(if key == KeyCode::Slash { "/" } else { "" });
                // `Minecraft.handleKeybinds` opens it in the call that then
                // releases an active use (Gui.handleKeybinds runs before the
                // isUsingItem branch).
                self.release_use_for_keybind_screen();
                Ok(true)
            }
            KeyCode::KeyQ => {
                if self.spectator() {
                    return Ok(true);
                }
                if let Some(r) = self.remote.as_mut() {
                    r.pending_keys.push(key);
                }
                Ok(true)
            }
            // Open the ridden vehicle's own screen (`HasCustomInventoryScreen`)
            // instead of the player's inventory: riding a horse, llama or
            // chest boat, the inventory key sends
            // ServerboundPlayerCommandPacket OPEN_INVENTORY.
            KeyCode::KeyE if self.riding_mount_with_menu() || self.riding_chest_boat() => {
                if self.in_portal_transition() {
                    return Ok(true);
                }
                if let Some(r) = self.remote.as_mut() {
                    r.pending_keys.push(key);
                }
                Ok(true)
            }
            KeyCode::KeyE => {
                if self.in_portal_transition() {
                    return Ok(true);
                }
                // The inventory opens without telling the server.
                if self.tutorial.open_inventory() {
                    self.persist_tutorial_step();
                }
                self.recipe_ui.reset_screen();
                if self.creative {
                    self.set_screen(Screen::Creative);
                    self.refresh_creative_tabs();
                    self.creative_screen.reopen();
                    self.refresh_saved_hotbars();
                } else {
                    self.set_screen(Screen::Inventory);
                }
                self.craft_size = 2;
                self.capture(false);
                // `Minecraft.handleKeybinds` opens the inventory in the same
                // call that then releases an active use.
                self.release_use_for_keybind_screen();
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    /// `Minecraft.handleKeybinds`: a key that opens a screen runs
    /// `Gui.setScreen`, whose `KeyMapping.releaseAll` clears the use key, and
    /// the same invocation then releases an active use
    /// (`player.isUsingItem() && !keyUse.isDown()` â†’ `releaseUsingItem`).
    /// Only the screens opened from those keybinds do this (inventory, chat,
    /// advancements); the pause menu, server containers, mount inventories and
    /// dialogs release on the first tick after closing instead, as vanilla
    /// does.
    pub(super) fn release_use_for_keybind_screen(&mut self) {
        // Only a server's client holds a use to release.
        if self.remote.is_none() {
            return;
        }
        // `KeyMapping.releaseAll` clears key.use; without this the button still
        // reads held and a use would start again on the first tick back.
        self.right = false;
        horizon_client::interaction_input::release_on_screen_open(&mut RemoteInputWorld(self));
    }
    /// `MultiPlayerGameMode.releaseUsingItem`: sends `RELEASE_USE_ITEM` and
    /// runs the client half of `Item.releaseUsing` (a Riptide launch moves the
    /// player itself). Vanilla reaches it both when the use key is let go and,
    /// in the same tick, when a keybind opens a screen.
    fn release_remote_use(&mut self) {
        self.send(S::PlayerAction(play::PlayerAction {
            action: 6,
            pos: [0; 3],
            direction: 0,
            sequence: 0,
        }));
        // a mod's Stop may skip the item's release, and a mod's own release follows.
        let hand = self.used_hand();
        if let Some((slot, ticks)) = self.using.take()
            && let Some(hand) = hand
            && let Some(stack) = self.stack_in(slot).cloned()
        {
            self.with_hand(hand, |g| {
                if !g.use_item_stopped(&stack, ticks) {
                    return;
                }
                if &*stack.item == "minecraft:trident" {
                    g.release_trident(g.ticks_held(&stack, ticks));
                }
                g.mod_release_using(&stack, ticks);
            });
        }
    }
    /// ClientPacketListener.sendChat / sendCommand, signed with a chat
    /// session (game/network/remote_chat.rs). The Chat option's Hidden and System
    /// levels keep chat out and Hidden keeps the commands out (ChatAbilities).
    pub(super) fn send_signed_chat_input(&mut self, message: &str) {
        let message = message.trim();
        if message.is_empty() || !self.chat_abilities().allows(message) {
            return;
        }
        if let Some(command) = message.strip_prefix('/') {
            if self.studio_command(command) {
                return;
            }
            self.send_command(command);
        } else {
            self.send_signed_chat(message);
        }
    }
    /// key.swapOffhand pressed: the next tick sends it (`remote_swap_hands`).
    pub(super) fn remote_swap_pressed(&mut self) {
        if let Some(r) = self.remote.as_mut() {
            r.swap_presses += 1;
        }
    }
    /// ChatScreen.handleChatInput fires the client event for chat, not commands.
    pub(super) fn send_chat_from_screen(&mut self, message: &str) {
        if message.starts_with('/') || !mod_api::__private::listened(mod_api::HookId::ClientChat) {
            self.send_signed_chat_input(message);
        } else if let Some(uuid) = self.remote.as_ref().map(|r| r.profile) {
            let id = self.own_entity_id();
            let sent = chat_after_event(message, |outgoing, cancel| {
                self.native_client_chat(id, uuid, message, outgoing, cancel);
            });
            if let Some(outgoing) = sent {
                self.send_signed_chat(&outgoing);
            }
        }
    }
    /// MultiPlayerGameMode.dropItem (key.drop): predict the removal, then send
    /// the action; swing only if a nonempty stack was removed.
    pub(super) fn remote_drop(&mut self, all: bool) {
        // ensureHasSentCarriedItem: the server drops from the slot just chosen.
        self.remote_send_carried();
        let dropped =
            horizon_client::items::take_held_drop(&mut self.inventory[self.selected], all, false);
        self.send(S::PlayerAction(play::PlayerAction {
            action: if all { 4 } else { 5 },
            pos: [0; 3],
            direction: 0,
            sequence: 0,
        }));
        if dropped.is_some() {
            self.swing_hand(Hand::Main, SwingAnimation::DEFAULT);
        }
    }
    /// Minecraft.handleKeybinds, key.swapOffhand: the server swaps the hands
    /// (ServerboundPlayerActionPacket SWAP_ITEM_WITH_OFFHAND at BlockPos.ZERO, facing down) and
    /// sends both slots back; nothing is predicted.
    fn remote_swap_hands(&mut self) {
        self.send(S::PlayerAction(play::PlayerAction {
            action: 7,
            pos: [0; 3],
            direction: 0,
            sequence: 0,
        }));
        // Equal stacks swap with no slot update to wait for.
        let (main, off) = (self.stack_in(self.selected), self.offhand.as_ref());
        let moves = !super::hand::matches_for_swap(main, off);
        if let Some(r) = self.remote.as_mut() {
            r.swap_answer = [moves; 2];
        }
    }
    /// The server set inventory `slot` (`shield::OFFHAND_SLOT` for the off
    /// hand), `changed` from the stack this client held there. The stack
    /// that answers a swap (`setItemInHand`) is another object, which the
    /// hand shows after the swap animation (`hand_stack_replaced`); other
    /// updates of a hand's slot count as the held stack changing in place,
    /// as this client doesn't predict every use the way vanilla's does
    /// (a thrown snowball's stack shrinks only when the server says so).
    fn remote_hand_slot_set(&mut self, slot: usize, changed: bool) {
        let hand = if slot == super::shield::OFFHAND_SLOT {
            1
        } else if slot == self.selected {
            0
        } else {
            return;
        };
        let answer = self
            .remote
            .as_mut()
            .is_some_and(|r| std::mem::take(&mut r.swap_answer[hand]));
        if answer && changed {
            self.hand_stack_replaced(slot);
        }
    }
    /// A container click: the server answers with the authoritative slots.
    pub(super) fn remote_click(&mut self, right: bool) {
        let Some(size) = self
            .renderer
            .as_ref()
            .map(|r| [r.config.width, r.config.height])
        else {
            return;
        };
        let batch = HudBatch::new(size, self.settings.gui_scale_inputs());
        let Some(internal) = self.slot_at(&batch, self.cursor.0, self.cursor.1) else {
            return;
        };
        let input = if self.has_shift_down() {
            Input::QuickMove
        } else {
            Input::Pickup
        };
        if self.screen == Screen::Creative {
            // CreativeModeInventoryScreen.slotClicked: the Survival Inventory tab clicks the
            // player's own InventoryMenu, and its CreativeInventoryListener sends each slot that
            // changed (SetCreativeModeSlot). No ContainerClick: a server holding the carried
            // stack would give it back after the Destroy Item slot cleared it (#3307).
            self.click_open_inventory(Some(internal), i32::from(right), input);
            return;
        }
        self.remote_menu_ops(&[(Some(internal), i32::from(right), input)]);
    }
    /// `AbstractContainerMenu.setItem`/`initializeContents`: the state id a
    /// click answers with is the open menu's, not another window's.
    fn remote_menu_state(&mut self, container: i32, state: i32) {
        let open = self.remote_menu().0;
        if let Some(r) = self.remote.as_mut().filter(|_| container == open) {
            r.state_id = state;
        }
    }
    /// The server's open menu: its window id and menu type (0, "inventory"
    /// when none is open).
    pub(super) fn remote_menu(&self) -> (i32, String) {
        match self.remote.as_ref().and_then(|r| r.menu.as_ref().cloned()) {
            Some((id, name)) => (id, name),
            None => (0, "inventory".to_owned()),
        }
    }
    /// `MultiPlayerGameMode.handleContainerInput` for each click of `ops` on
    /// the server's open menu (a quick craft drag is several): the verified
    /// menu engine predicts them on this client's copy, then each goes out as
    /// a `ContainerClick` with the slots it changed and the carried stack
    /// after it; the server corrects anything it sees differently. A menu the
    /// engine doesn't model sends the clicks unpredicted. False when a click's
    /// slot isn't one of the menu's.
    pub(super) fn remote_menu_ops(&mut self, ops: &[(Option<usize>, i32, Input)]) -> bool {
        // CrafterScreen.slotClicked: a slot toggle goes out before the click.
        if let [(Some(slot), button, input)] = ops {
            self.crafter_slot_clicked(*slot, *button, *input);
        }
        let (_, menu) = self.remote_menu();
        let mut clicks = Vec::with_capacity(ops.len());
        for &(slot, button, input) in ops {
            let slot = match slot {
                None => -999,
                Some(internal) => match Self::menu_slot_index(&menu, internal) {
                    Some(s) => s as i16,
                    None => return false,
                },
            };
            clicks.push((slot, button, input));
        }
        self.remote_menu_clicks(&clicks);
        true
    }
    /// [`Game::remote_menu_ops`]'s clicks by the menu's own slot ids (a mod's `slots.click`).
    pub(super) fn remote_menu_clicks(&mut self, clicks: &[(i16, i32, Input)]) {
        let (container, _) = self.remote_menu();
        // Unpredicted: nothing changed, the carried stack as it is.
        let mut sent: Vec<Predicted> = clicks
            .iter()
            .map(|&(slot, button, input)| (slot, button, input, Vec::new(), self.carried.clone()))
            .collect();
        let mut threw = false;
        // The engine's slot indices are the menu's own, as on the network.
        self.menu_session(|m, env, _| {
            let Some((predicted, dropped)) = predict_clicks(m, env, clicks) else {
                // The server throws on it too: nothing is predicted.
                return Some(false);
            };
            sent = predicted;
            threw = dropped;
            Some(true)
        });
        self.send_clicks(container, sent);
        // LivingEntity.drop(.., Prediction.PREDICTED): the client swings its own
        // main hand; the server shows the swing only to the other players.
        if threw {
            self.swing_hand(Hand::Main, SwingAnimation::DEFAULT);
        }
    }
    /// A QUICK_MOVE on the server's open menu with nothing predicted: the
    /// client's slots stay as they are and every slot it moved comes back from
    /// the server (`broadcastChanges`), the answer the next step waits for.
    /// A prediction the server agrees with gets no answer at all.
    pub(super) fn remote_quick_move_unpredicted(&mut self, internal: usize) -> bool {
        let (container, menu) = self.remote_menu();
        let Some(slot) = Self::menu_slot_index(&menu, internal) else {
            return false;
        };
        let carried = self.carried.clone();
        let click = (slot as i16, 0, Input::QuickMove, Vec::new(), carried);
        self.send_clicks(container, vec![click]);
        true
    }
    /// A Shift double click on a server (`AbstractContainerScreen.mouseReleased`):
    /// a predicted QUICK_MOVE for every slot the engine moves, each sent.
    pub(super) fn remote_quick_move_all(&mut self, slot: usize, moved: &ItemStack) {
        let (container, _) = self.remote_menu();
        let mut sent: Vec<Predicted> = Vec::new();
        self.menu_session(|m, env, to_menu| {
            let clicked = to_menu(slot)?;
            let mut before = menu_items(m);
            let mut each = |m: &horizon_client::menu::Menu, t: usize| {
                let changed = changed_slots(&before, m);
                sent.push((t as i16, 0, Input::QuickMove, changed, m.carried.clone()));
                before = menu_items(m);
            };
            Some(
                m.quick_move_all_each(env, clicked, moved, &mut each)
                    .is_ok(),
            )
        });
        self.send_clicks(container, sent);
    }
    fn send_clicks(&mut self, container: i32, clicks: Vec<Predicted>) {
        // A result the engine claims to have changed is the one the server
        // agrees to (`click_agreed`), so it sends none: keep it. A result the
        // engine doesn't claim (a recipe it doesn't know) the server sends.
        if matches!(
            self.remote_menu().1.as_str(),
            "inventory" | "minecraft:crafting"
        ) && let Some(r) = self.remote.as_mut()
        {
            for (.., changed, _) in &clicks {
                if let Some((_, result)) = changed.iter().find(|(slot, _)| *slot == 0) {
                    r.craft_result = result.clone();
                }
            }
        }
        let state = self.remote.as_ref().map_or(0, |r| r.state_id);
        let hashed = |g: &Self, s: &Option<ItemStack>| g.hashed_stack(s);
        for (slot, button, input, changed, carried) in clicks {
            let changed = changed.iter().map(|(i, s)| (*i, hashed(self, s))).collect();
            let carried = hashed(self, &carried);
            self.send(S::ContainerClick(play::ContainerClick {
                container,
                state,
                slot,
                button: button as i8,
                input: input_ordinal(input),
                changed,
                carried,
            }));
        }
    }
    /// The server's open menu is the inventory or a chest-like container
    /// (chests, barrels, ender chests: `generic_9xN`; shulker boxes).
    pub(super) fn remote_storage_open(&self) -> bool {
        match self.remote.as_ref().and_then(|r| r.menu.as_ref()) {
            None => true,
            Some((_, name)) => {
                name == "inventory"
                    || name.starts_with("minecraft:generic_9x")
                    || name == "minecraft:shulker_box"
            }
        }
    }
    /// A click outside a server's window (slot -999): predicted like
    /// `MultiPlayerGameMode.handleContainerInput` (the carried stack after
    /// `doClick`; the server drops the stack and swings the hand).
    pub(super) fn remote_throw(&mut self, right: bool, input: horizon_client::menu::Input) {
        use horizon_client::menu::Input;
        let container = self
            .remote
            .as_ref()
            .and_then(|r| r.menu.as_ref())
            .map_or(0, |m| m.0);
        if input == Input::Pickup
            && let Some(c) = self.carried.as_mut()
        {
            if right && c.count > 1 {
                c.count -= 1;
            } else {
                self.carried = None;
            }
            self.swing_arm();
        }
        let carried = self.hashed_stack(&self.carried);
        let state = self.remote.as_ref().map_or(0, |r| r.state_id);
        self.send(S::ContainerClick(play::ContainerClick {
            container,
            state,
            slot: -999,
            button: i8::from(right),
            input: if input == Input::Throw { 4 } else { 0 },
            changed: Vec::new(),
            carried,
        }));
    }
    /// `CreativeModeInventoryScreen.slotClicked` outside the window: the server knows nothing of
    /// a stack picked from the creative screen, so the carried stack (an item tab's right click:
    /// one of it) is dropped through `handleCreativeModeItemDrop`, a SetCreativeModeSlot to slot
    /// -1, and the hand swings as `Player.drop` predicts.
    pub(super) fn remote_creative_drop(&mut self, right: bool) {
        let one = right && !self.creative_inventory_open();
        let Some(carried) = self.carried.as_mut() else {
            return;
        };
        let thrown = if one && carried.count > 1 {
            carried.count -= 1;
            ItemStack {
                count: 1,
                ..carried.clone()
            }
        } else {
            self.carried.take().expect("carried")
        };
        self.remote_creative_throw(thrown);
    }
    /// `MultiPlayerGameMode.handleCreativeModeItemDrop`: a stack thrown out of the creative
    /// screen goes to the server as a SetCreativeModeSlot to slot -1, which drops it from the
    /// player's hand; the client only swings (`Player.drop` on a client).
    pub(super) fn remote_creative_throw(&mut self, thrown: ItemStack) {
        if !self.creative {
            return;
        }
        self.swing_arm();
        let item = self.to_network(&Some(thrown));
        self.send(S::SetCreativeModeSlot(play::SetCreativeModeSlot {
            slot: -1,
            item,
        }));
    }
    pub(super) fn remote_close_container(&mut self) {
        let id = self
            .remote
            .as_ref()
            .and_then(|r| r.menu.as_ref().map(|m| m.0))
            .unwrap_or(0);
        self.send(S::ContainerClose(play::ContainerClose { container: id }));
        self.remote_close_local();
    }
    /// `MultiPlayerGameMode.handleInventoryButtonClick`: button `button` of
    /// the menu the server opened goes to it (`ContainerButtonClick`), whose
    /// menu decides what it does (`clickMenuButton`). No sound: that is the
    /// screen's. False while no such menu is open.
    pub(super) fn remote_button_click(&self, button: i32) -> bool {
        let Some(container) = self
            .remote
            .as_ref()
            .and_then(|r| r.menu.as_ref())
            .map(|(w, _)| *w)
        else {
            return false;
        };
        self.send(S::ContainerButtonClick(play::ContainerButtonClick {
            container,
            button,
        }));
        true
    }
    fn remote_reject_container(&mut self, container: i32) {
        self.remote_close_local();
        if self.screen == Screen::Inventory {
            self.set_screen(Screen::Playing);
            self.capture(true);
        }
        self.send(S::ContainerClose(play::ContainerClose { container }));
    }
    fn remote_close_local(&mut self) {
        // `Gui.setScreen` clears the focus of the screen it takes down: the
        // container's keyboard focus must not outlive it (a Tab bound to the
        // inventory moves the focus and closes the container in one press).
        self.pad_clear_focus();
        self.recipe_ui.reset_screen();
        if let Some(r) = self.remote.as_mut() {
            r.menu.set(None);
            r.set_menu_title(None);
            r.craft_result = None;
        }
        self.mount_menu_reset();
        // The client's menu has no level access: what it holds is the server's.
        self.station = None;
        // AbstractContainerMenu.removed: the server gives the grid and the
        // carried stack back to the inventory and syncs it; the copies here
        // are stale the moment the window closes.
        self.craft_grid.fill(None);
        self.carried = None;
        self.craft_size = 2;
        if let Some(key) = self.container.take() {
            // Server containers stay renderable but lose their copied slots.
            if let Some(entity) = self.world.block_entities.get_mut(&key) {
                for s in entity.slots_mut() {
                    *s = None;
                }
            }
        }
    }
    /// A server's trading window (MerchantScreen over a ClientSideMerchant).
    fn remote_open_merchant(
        &mut self,
        container: i32,
        title: String,
        text: horizon_client::text::Text,
    ) {
        if let Some(r) = self.remote.as_mut() {
            r.menu
                .set(Some((container, "minecraft:merchant".to_owned())));
            r.set_menu_title(Some(text));
        }
        let session = horizon_client::merchant::Session::new(Vec::new(), true);
        self.open_remote_merchant(session, (!title.is_empty()).then_some(title));
    }
    /// ClientboundMerchantOffersPacket.
    fn remote_merchant_offers(&mut self, p: play::MerchantOffers) {
        if self.open_menu_name(p.container) != "minecraft:merchant" {
            return;
        }
        let cost = |g: &Self, c: &play::NetCost| {
            g.to_stack(&Some(c.clone().into()))
                .map(|stack| horizon_client::merchant::ItemCost { stack })
        };
        let offers: Vec<_> = p
            .offers
            .iter()
            .filter_map(|o| {
                let mut offer = horizon_client::merchant::MerchantOffer {
                    base_cost_a: cost(self, &o.cost_a)?,
                    cost_b: match &o.cost_b {
                        Some(b) => Some(cost(self, b)?),
                        None => None,
                    },
                    result: self.to_stack(&Some(o.result.clone()))?,
                    uses: o.uses,
                    max_uses: o.max_uses,
                    reward_exp: true,
                    special_price_diff: o.special_price,
                    demand: o.demand,
                    price_multiplier: o.price_multiplier,
                    xp: o.xp,
                };
                // setToOutOfStock.
                if o.out_of_stock {
                    offer.uses = offer.max_uses;
                }
                Some(offer)
            })
            .collect();
        if let Some(m) = self.station.as_mut().and_then(|s| s.merchant.as_mut()) {
            m.session.offers = offers;
            m.session.level = p.level;
            m.session.xp = p.xp;
            m.session.villager = p.show_progress;
        }
    }
    /// ClientboundOpenScreenPacket for the menus Horizon draws.
    fn remote_open_screen(&mut self, p: play::OpenScreen) {
        // A new RecipeBookComponent owns fresh ghost slots and an overlay,
        // even when the server replaces a menu without a close packet.
        self.recipe_ui.reset_screen();
        let menu = self
            .remote
            .as_ref()
            .and_then(|r| {
                r.builtin
                    .get("minecraft:menu")?
                    .name(p.menu)
                    .map(str::to_owned)
            })
            .unwrap_or_default();
        let target = self
            .remote
            .as_mut()
            .and_then(|r| r.use_target.take())
            .map(|t| (t[0], t[1], t[2]));
        // The server names a mod menu's block itself: it, not the last
        // clicked block, is the one the menu is of. Only for this window:
        // any other's is stale.
        let extra = self
            .remote
            .as_mut()
            .and_then(|r| r.menu_extra.take())
            .filter(|e| e.window == p.container);
        let target = extra
            .as_ref()
            .map(|e| (e.pos[0], e.pos[1], e.pos[2]))
            .or(target);
        if menu == "minecraft:lectern" {
            // LecternScreen: the book and page come in the menu's content and data.
            if let Some(r) = self.remote.as_mut() {
                r.menu.set(Some((p.container, menu)));
                r.set_menu_title(None);
            }
            self.remote_open_lectern(target.unwrap_or_default());
            return;
        }
        if let Some(kind) = Self::remote_station_kind(&menu) {
            let title = menu_title(&p.title);
            let pos = target.map_or([0, 0, 0], |t| [t.0, t.1, t.2]);
            self.remote_open_station(p.container, menu, kind, title, pos);
            return;
        }
        if menu == "minecraft:merchant" {
            // A trading window: its offers follow in a MerchantOffers packet.
            let title = plain(
                &p.title,
                &self
                    .remote
                    .as_ref()
                    .map(|r| r.lang.clone())
                    .unwrap_or_default(),
            );
            self.remote_open_merchant(p.container, title, menu_title(&p.title));
            return;
        }
        let mod_type = horizon_client::containers::mod_menu_type(&menu);
        let entity = match menu.as_str() {
            "minecraft:crafting" => None,
            "minecraft:furnace" | "minecraft:blast_furnace" | "minecraft:smoker" => {
                let kind = match menu.as_str() {
                    "minecraft:blast_furnace" => FurnaceKind::BlastFurnace,
                    "minecraft:smoker" => FurnaceKind::Smoker,
                    _ => FurnaceKind::Furnace,
                };
                Some(BlockEntity::Furnace(Furnace {
                    kind,
                    items: [None, None, None],
                    lit_remaining: 0,
                    lit_total: 0,
                    cooking_timer: 0,
                    cooking_total: 0,
                    speed: 1.0,
                    recipes_used: std::collections::HashMap::new(),
                }))
            }
            // HopperMenu, and DispenserMenu (MenuType.GENERIC_3x3: dispensers
            // and droppers).
            "minecraft:hopper" => Some(BlockEntity::Chest(vec![None; 5])),
            // BrewingStandMenu: its times come in ContainerSetData.
            "minecraft:brewing_stand" => Some(BlockEntity::Brewing(Box::new(
                horizon_client::brewing::BrewingStand::new(),
            ))),
            "minecraft:generic_3x3" => Some(BlockEntity::Chest(vec![None; 9])),
            // CrafterMenu: the disabled slots and triggered come in ContainerSetData.
            "minecraft:crafter_3x3" => Some(BlockEntity::Crafter(Default::default())),
            m if m.starts_with("minecraft:generic_9x") || m == "minecraft:shulker_box" => {
                let rows = m
                    .strip_prefix("minecraft:generic_9x")
                    .and_then(|r| r.parse::<usize>().ok())
                    .unwrap_or(3);
                Some(BlockEntity::Chest(vec![None; rows * 9]))
            }
            // A content mod's menu type: its container's slots, which its
            // mod's screen places.
            _ if mod_type.is_some() => Some(BlockEntity::Mod(Box::new(
                horizon_client::containers::ModBlockEntity::new(
                    String::new(),
                    horizon_client::containers::ModBlockEntityFields {
                        items: vec![None; mod_type.map_or(0, |t| t.slots)],
                        ..Default::default()
                    },
                ),
            ))),
            _ => {
                // Unsupported menu: tell the server we closed it.
                self.remote_reject_container(p.container);
                self.add_chat_line(
                    horizon_client::text::Text::literal(format!(
                        "(Horizon cannot show {menu} screens yet)"
                    )),
                    horizon_client::chat::ChatSource::Client,
                );
                return;
            }
        };
        let title = menu_title(&p.title);
        if let Some(r) = self.remote.as_mut() {
            r.menu.set(Some((p.container, menu.clone())));
            r.set_menu_title(Some(title));
        }
        self.craft_size = if menu == "minecraft:crafting" { 3 } else { 2 };
        self.container = None;
        if let Some(entity) = entity {
            // The clicked block shows the menu when it holds that container;
            // an ender chest's or a double chest's slots are no one block's,
            // so they go under a key no block has.
            let holds = |k: &(i32, i32, i32)| {
                let name = &self.res.facts().get(self.world.block(k.0, k.1, k.2)).name;
                BlockEntity::for_block(name).is_some_and(|e| {
                    e.id() == entity.id() && e.slots().len() == entity.slots().len()
                })
            };
            let key = target.filter(holds).unwrap_or((i32::MIN, 0, 0));
            self.world.block_entities.insert(key, entity);
            self.container = Some(key);
            self.open_native_screen(key, target, p.container, &menu, extra);
            // A mod menu type shows through its mod's screen only: without one
            // (`MenuScreens.create` finds no screen) the window closes again.
            if self.mod_screen.is_none() && mod_type.is_some() {
                self.world.block_entities.remove(&key);
                self.container = None;
                if let Some(r) = self.remote.as_mut() {
                    r.menu.set(None);
                    r.set_menu_title(None);
                }
                self.send(S::ContainerClose(play::ContainerClose {
                    container: p.container,
                }));
                return;
            }
        }
        self.set_screen(Screen::Inventory);
        self.capture(false);
    }
    /// `ClientPacketListener.handleMountScreenOpen`: a horse's or llama's
    /// inventory; for another entity the window is closed again.
    fn remote_open_mount(&mut self, p: play::MountScreenOpen) {
        let columns = usize::try_from(p.columns).unwrap_or(0).min(5);
        if !self.open_remote_mount_menu(p.entity as u32, columns) {
            self.remote_reject_container(p.container);
            return;
        }
        if let Some(r) = self.remote.as_mut() {
            r.menu
                .set(Some((p.container, format!("{MOUNT_MENU}{columns}"))));
            r.set_menu_title(None);
        }
    }
    /// The type of the menu a server opened (`MenuType`'s id).
    pub(super) fn remote_menu_type(&self) -> Option<&str> {
        Some(self.remote.as_ref()?.menu.as_ref()?.1.as_str())
    }
    /// The title the server gave the menu it opened.
    pub(super) fn remote_menu_title(&self) -> Option<horizon_client::text::Text> {
        self.remote.as_ref()?.menu_title.clone()
    }
    /// A language change reaches the server-sent components too: their titles
    /// are resolved when the packet arrives, with the session's language.
    pub(super) fn refresh_remote_language(&mut self) {
        if let Some(lang) = horizon_client::lang::current()
            && let Some(r) = self.remote.as_mut()
        {
            r.lang = Arc::new(lang.map().clone());
        }
    }
    /// A server's chest-like menu (`MenuType.GENERIC_9xN`): its rows and title.
    pub(super) fn remote_chest_menu(&self) -> Option<(usize, horizon_client::text::Text)> {
        let r = self.remote.as_ref()?;
        let rows = r
            .menu
            .as_ref()?
            .1
            .strip_prefix("minecraft:generic_9x")?
            .parse()
            .ok()?;
        Some((
            rows,
            r.menu_title
                .clone()
                .unwrap_or_else(horizon_client::text::Text::empty),
        ))
    }
    /// ServerboundPlayerCommandPacket STOP_SLEEPING.
    pub(super) fn remote_stop_sleeping(&mut self) {
        let id = self.remote.as_ref().map_or(0, |r| r.entity_id);
        self.send(S::PlayerCommand(play::PlayerCommand {
            id,
            action: 0,
            data: 0,
        }));
    }
    pub(super) fn remote_respawn(&mut self) {
        self.send(S::ClientCommand(play::ClientCommand { action: 0 }));
    }

    /// A connection's screen is over the world: connecting, reconfiguring, or
    /// the level loading screen, a portal's included (`remote_overlay`).
    pub(super) fn remote_screen_up(&self) -> bool {
        !self.replay.playing()
            && self
                .remote
                .as_ref()
                .is_some_and(|r| self.player.is_none() || r.level_load.is_active())
    }
    /// Connecting / loading screens; true when it covered the frame.
    pub(super) fn remote_overlay(&mut self, batch: &mut HudBatch) -> bool {
        if !self.remote_screen_up() {
            return false;
        }
        let Some(r) = &self.remote else { return false };
        let loading = r.level_load.is_active();
        if self.player.is_some() && !loading {
            return false;
        }
        let (w, h) = (batch.width, batch.height);
        if let Some(since) = r.reconfiguring {
            // ServerReconfigScreen narrates its title.
            horizon_client::widgets::begin_screen(|| {
                horizon_client::widgets::ScreenInfo::new(
                    "reconfiguring",
                    super::screen_info::tr("connect.reconfiguring", &[]),
                )
            });
            self.draw_reconfiguring(batch, since);
            return true;
        }
        if r.joined || loading {
            // LevelLoadingScreen: the world's terrain comes in. A server's
            // chunk status view and progress are the integrated server's alone
            // (`LevelLoadTracker.statusView`, `hasProgress`), and so is what
            // the narrator says of it.
            let progress = self
                .integrated
                .as_ref()
                .map(|i| i.load_view())
                .filter(|v| v.has_progress())
                .map(|v| v.server_progress());
            horizon_client::widgets::begin_screen(|| {
                super::screen_info::level_loading(progress, &super::screen_info::tr)
            });
            if self.draw_dimension_transition_screen(
                batch,
                &r.level_load,
                progress.map(|_| self.loading_progress),
            ) {
                return true;
            }
            self.draw_level_loading_background(batch, r.level_load.reason);
            let atlas = &self.gui.atlas;
            match self.integrated.as_ref().map(|i| i.load_view()) {
                Some(view) => {
                    let status = |x: i32, z: i32| view.status(x, z);
                    let progress = view.has_progress().then_some(self.loading_progress);
                    horizon_client::level_loading::draw(batch, atlas, Some(&status), progress);
                }
                None => horizon_client::level_loading::draw(batch, atlas, None, None),
            }
            return true;
        }
        let cancel = self.connect_cancel((w, h)).is_some();
        // ConnectScreen.extractRenderState: "Joining the world" every 2 s (a
        // server's: single player's isn't joined through ConnectScreen).
        if self.integrated.is_none()
            && let Some(r) = self.remote.as_mut()
            && r.joining_narrated
                .is_none_or(|t| t.elapsed() > Duration::from_millis(2000))
        {
            r.joining_narrated = Some(Instant::now());
            self.say_system_now(&horizon_client::text::Text::translate(
                "narrator.joining",
                &[],
            ));
        }
        let Some(r) = &self.remote else { return false };
        // ConnectScreen: `GameNarrator.NO_TITLE`, then its Cancel; single
        // player's server is joined under its LevelLoadingScreen.
        let single_player = self.integrated.is_some();
        horizon_client::widgets::begin_screen(|| {
            if single_player {
                super::screen_info::level_loading(None, &super::screen_info::tr)
            } else {
                horizon_client::widgets::ScreenInfo::new("connect", "")
            }
        });
        let atlas = &self.gui.atlas;
        let status = horizon_client::text::Text::untrusted(r.status.as_str());
        horizon_client::level_loading::draw_connecting(batch, atlas, &status, self.cursor, cancel);
        true
    }
    /// Connected or connecting with no player yet: `ConnectScreen`, then
    /// `LevelLoadingScreen`, over the panorama with its blur.
    pub(super) fn remote_loading(&self) -> bool {
        if self.replay.playing() {
            return false;
        }
        use horizon_client::level_loading::Reason;
        self.remote.as_ref().is_some_and(|r| {
            if r.level_load.is_active() {
                // The portal screens have no panorama, so no blur.
                r.level_load.reason == Reason::Other
            } else {
                self.player.is_none()
            }
        })
    }
    /// `ServerReconfigScreen`: a vertical layout of cells padded by 10, the
    /// title (9 high) over Disconnect (150 x 20), centred. The title's and the
    /// button's top left corners.
    fn reconfigure_layout(&self, (w, h): (f32, f32)) -> ((f32, f32), (f32, f32)) {
        let title = horizon_client::text::Text::translate("connect.reconfiguring", &[]);
        let title_w = self.gui.atlas.font.width_text(&title).ceil() as i32;
        let width = (title_w + 20).max(170);
        let left = ((w as i32 - width) as f32 * 0.5) as i32;
        let top = ((h as i32 - 69) as f32 * 0.5) as i32;
        let across = |child: i32| (10.0 + ((width - child - 10) as f32 - 10.0) * 0.5) as i32;
        (
            ((left + across(title_w)) as f32, (top + 10) as f32),
            ((left + across(150)) as f32, (top + 39) as f32),
        )
    }
    /// `ServerReconfigScreen` over the panorama; Disconnect wakes up after
    /// `DISCONNECT_TIME` (600 ticks).
    fn draw_reconfiguring(&self, batch: &mut HudBatch, since: u64) {
        let atlas = &self.gui.atlas;
        let (w, h) = (batch.width, batch.height);
        horizon_client::ui::menu_background(batch, atlas, 0.0, 0.0, w, h, false);
        let title = horizon_client::text::Text::translate("connect.reconfiguring", &[]);
        let ((tx, ty), (bx, by)) = self.reconfigure_layout((w, h));
        batch.text_styled(atlas, &title, tx, ty, u32::MAX, true);
        // CommonComponents.GUI_DISCONNECT
        let label = horizon_client::text::Text::translate("menu.disconnect", &[]);
        let active = self.ticks.saturating_sub(since) >= 600;
        let hover = horizon_client::ui::hovered(self.cursor, bx, by, 150.0, 20.0);
        horizon_client::ui::button(batch, atlas, bx, by, 150.0, 20.0, &label, active, hover);
    }
    /// `ConnectScreen`'s Cancel: a server's (single player's server isn't
    /// joined through `ConnectScreen`), until the login is done.
    fn connect_cancel(&self, (w, h): (f32, f32)) -> Option<(f32, f32)> {
        let r = self.remote.as_ref()?;
        (self.player.is_none() && !r.joined && self.integrated.is_none())
            .then(|| horizon_client::level_loading::connect_cancel(w, h))
    }
    /// A click on `ConnectScreen`: Cancel aborts the connection
    /// (`connect.aborted`) and goes back. Escape does nothing there
    /// (`shouldCloseOnEsc` is false).
    pub(super) fn connect_screen_click(&mut self) -> Result<()> {
        // ServerReconfigScreen's Disconnect, once it is active.
        if let Some(since) = self.remote.as_ref().and_then(|r| r.reconfiguring) {
            let (_, (x, y)) = self.reconfigure_layout(self.gui_size());
            if self.ticks.saturating_sub(since) >= 600
                && horizon_client::ui::hovered(self.cursor, x, y, 150.0, 20.0)
            {
                self.audio.play_button_click();
                self.leave_server(None)?;
            }
            return Ok(());
        }
        let Some((x, y)) = self.connect_cancel(self.gui_size()) else {
            return Ok(());
        };
        if horizon_client::ui::hovered(self.cursor, x, y, 200.0, 20.0) {
            self.audio.play_button_click();
            self.leave_server(None)?;
        }
        Ok(())
    }
    /// The server's death message; chat uses the shared ChatComponent.
    pub(super) fn remote_hud(&self, batch: &mut HudBatch) {
        let Some(r) = &self.remote else { return };
        let atlas = &self.gui.atlas;
        let w = batch.width;
        if self.screen == Screen::Dead
            && let Some(message) = &r.death_message
        {
            // DeathScreen draws the styled cause component.
            let tw = atlas.font.width_text(message);
            batch.text_styled(
                atlas,
                message,
                ((w - tw) / 2.0).floor(),
                85.0,
                u32::MAX,
                true,
            );
        }
    }
    /// Shared flag 6 of an entity on a server's client (`None` without a server).
    pub(super) fn remote_glowing(&self, id: u32) -> Option<bool> {
        let r = self.remote.as_ref()?;
        Some(r.glowing.contains(&(id as i32)))
    }
    /// The profile name of another player the server sent, by entity id.
    pub(super) fn remote_player_name(&self, id: u32) -> Option<&str> {
        let r = self.remote.as_ref()?;
        r.level_player_name(id as i32, &r.player_uuid(id)?)
    }
    pub(super) fn remote_player_uuid(&self, id: u32) -> Option<u128> {
        let remote = self.remote.as_ref()?;
        if self.player_model.as_ref().is_some_and(|m| m.id == id) {
            Some(remote.profile)
        } else {
            remote.level_players.get(&(id as i32)).copied()
        }
    }
    /// Status for `horizon-ctl status`.
    /// The other players' entities in the level and their names (their
    /// profiles', which outlive the player list's entries).
    pub(super) fn remote_player_names(&self) -> Vec<(u32, String)> {
        let Some(r) = self.remote.as_ref() else {
            return Vec::new();
        };
        r.level_players
            .iter()
            .filter(|(_, uuid)| **uuid != r.profile)
            .filter_map(|(id, uuid)| Some((*id as u32, r.level_player_name(*id, uuid)?.to_owned())))
            .collect()
    }
    /// `ClientClockManager.tick`: the clock moves on by the game time that
    /// passed since its last tick, at its rate. A game time the server sets
    /// forward (its sync every second) moves the clock with it.
    fn remote_clock_tick(&mut self) {
        let Some(r) = self.remote.as_mut() else {
            return;
        };
        let delta = self.world.game_time - r.clock_game_time;
        r.clock_game_time = self.world.game_time;
        for clock in r.clocks.values_mut() {
            let partial = f64::from(clock.partial_tick) + delta as f64 * f64::from(clock.rate);
            let full = partial.floor();
            clock.partial_tick = (partial - full) as f32;
            clock.total_ticks += full as i64;
        }
        self.world.day_time = r.clock_time(r.default_clock);
    }
    pub(super) fn remote_status(&self) -> Option<serde_json::Value> {
        let r = self.remote.as_ref()?;
        Some(serde_json::json!({
            "address": horizon_client::net::shown_address(&r.session.address),
            "name": r.session.name,
            "joined": r.joined,
            // ServerboundPlayerLoadedPacket sent: the level around the
            // player is ready, and the client ticks and sends its moves.
            "loaded": r.loaded_sent,
            "status": r.status,
            "entity_id": r.entity_id,
            "dimension": r.dimension.name,
            "game_mode": r.game_mode,
            "chunks": self.world.chunks.len(),
            "entities": r.entities.values().fold(std::collections::BTreeMap::<&str, u32>::new(), |mut m, t| {
                *m.entry(t.kind.as_str()).or_default() += 1;
                m
            }),
            "players": r.players.values().collect::<Vec<_>>(),
            // Where the other players' entities are (harnesses check movement).
            // What the other players hold and wear, as this client draws them.
            "player_equipment": self
                .mobs
                .iter()
                .filter(|m| r.level_players.contains_key(&(m.id as i32)))
                .map(|m| {
                    let e = &m.extra;
                    let armor = horizon_items::equipment::ARMOR.map(|slot| e.slot_id(slot));
                    serde_json::json!({
                        "id": m.id,
                        "swinging": m.ai.swinging,
                        "using": m.extra.using_item,
                        "armor": armor,
                        "main_hand": e.main_hand_id(),
                        "off_hand": e.off_hand_id(),
                    })
                })
                .collect::<Vec<_>>(),
            "player_positions": r
                .entities
                .values()
                .filter(|t| t.kind == "minecraft:player")
                .map(|t| t.position)
                .collect::<Vec<_>>(),
            // The other players this client draws dead (lying down, after EntityEvent DEATH).
            "dead_players": self
                .mobs
                .iter()
                .filter(|m| r.level_players.contains_key(&(m.id as i32)) && !m.alive())
                .map(|m| m.id)
                .collect::<Vec<_>>(),
            // The other players this client draws lying in a bed (SLEEPING_POS).
            "lying_players": self
                .mobs
                .iter()
                .filter(|m| r.level_players.contains_key(&(m.id as i32)) && m.extra.ints[1] != 0)
                .map(|m| m.id)
                .collect::<Vec<_>>(),
            // The player's own model exists (third person and the inventory draw it).
            "player_model": self.player_model.is_some(),
            // The world view draws its body now (third person, or lying in a bed;
            // not while invisible).
            "own_body": self.player_model_shown() && self.own_body_visible(),
            "text_display_vertices": self.text_display_vertices,
            "text_displays": self.text_displays.len(),
            "health": r.health.0,
            "food": r.health.1,
            "decode_errors": r.decode_errors,
            "chat": r.chat.iter().rev().take(10).rev().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
        }))
    }
}

/// `Entity.lookAt`'s rotation from `from` towards `to`, as (yaw, pitch).
fn look_at_rotation(from: [f64; 3], to: [f64; 3]) -> (f32, f32) {
    use horizon_core::geometry::trig::atan2;
    let (dx, dy, dz) = (to[0] - from[0], to[1] - from[1], to[2] - from[2]);
    let horizontal = (dx * dx + dz * dz).sqrt();
    // `* 180.0F / (float)Math.PI` on the double.
    let degrees = |r: f64| r * 180.0 / f64::from(std::f32::consts::PI);
    let pitch = wrap_degrees((-degrees(atan2(dy, horizontal))) as f32);
    let yaw = wrap_degrees(degrees(atan2(dz, dx)) as f32 - 90.0);
    (yaw, pitch)
}

/// `PositionMoveRotation.calculateAbsolute`'s pitch: the packet's pitch,
/// added to `source` when it is relative, clamped to +-90.
fn position_packet_pitch(source: f32, change: f32, relative: bool) -> f32 {
    let offset = if relative { source } else { 0.0 };
    (offset + change).clamp(-90.0, 90.0)
}

/// `Mth.wrapDegrees`.
fn wrap_degrees(a: f32) -> f32 {
    let a = a % 360.0;
    if a >= 180.0 {
        a - 360.0
    } else if a < -180.0 {
        a + 360.0
    } else {
        a
    }
}

/// `LivingEntity.tick`'s body target and `tickHeadTurn` for another player:
/// the body turns toward where it walks (backwards when walking back), or to
/// the look while the arm swings, and stays within 50 degrees of the look.
fn remote_player_head_turn(m: &mut Mob) {
    let (dx, dz) = (m.position.x - m.previous.x, m.position.z - m.previous.z);
    let mut target = m.body_yaw;
    if (dx * dx + dz * dz) as f32 > 0.002_500_000_2 {
        let walk = (dz.atan2(dx) as f32).to_degrees() - 90.0;
        let facing = (wrap_degrees(m.yaw) - walk).abs();
        target = if 95.0 < facing && facing < 265.0 {
            walk - 180.0
        } else {
            walk
        };
    }
    if m.anim.attack_anim > 0.0 {
        target = m.yaw;
    }
    m.body_yaw += wrap_degrees(target - m.body_yaw) * 0.3;
    let head = wrap_degrees(m.yaw - m.body_yaw);
    if head.abs() > 50.0 {
        m.body_yaw += head - head.signum() * 50.0;
    }
}

/// A click sent to a server: its slot, button and input, the slots it
/// changed and the carried stack after it.
type Predicted = (
    i16,
    i32,
    Input,
    Vec<(i16, Option<ItemStack>)>,
    Option<ItemStack>,
);

/// How many stacks the creative screen keeps in step with the server (`Remote::synced`).
const CREATIVE_SYNCED: usize = 41;

/// The player's stacks in `Remote::synced` order: the 36 inventory slots, the off hand, then
/// the armour from the head down. `CreativeInventoryListener.slotChanged` sends every
/// InventoryMenu slot the creative screen changes, the armour slots too.
fn creative_sync_stacks(
    inventory: &[Option<ItemStack>],
    offhand: &Option<ItemStack>,
    armor: &[Option<ItemStack>; 4],
) -> Vec<Option<ItemStack>> {
    let mut stacks: Vec<Option<ItemStack>> = inventory.iter().take(36).cloned().collect();
    stacks.resize(36, None);
    stacks.push(offhand.clone());
    stacks.extend(armor.iter().cloned());
    stacks
}

/// Entries of `creative_sync_stacks` that went from `before` to `now` are synced as `now`.
fn mark_synced(
    synced: &mut [Option<ItemStack>],
    before: &[Option<ItemStack>],
    now: Vec<Option<ItemStack>>,
) {
    for (i, stack) in now.into_iter().enumerate() {
        if before.get(i) != Some(&stack)
            && let Some(s) = synced.get_mut(i)
        {
            *s = stack;
        }
    }
}

/// InventoryMenu's slot for entry `i` of `creative_sync_stacks`: the hotbar is 36..=44, the
/// main inventory 9..=35, the off hand 45 and the armour 5..=8 (head first).
fn creative_sync_slot(i: usize) -> i16 {
    (match i {
        0..9 => 36 + i,
        36 => 45,
        37.. => 5 + (i - 37),
        _ => i,
    }) as i16
}

/// `MultiPlayerGameMode.handleContainerInput` for each click on the engine's
/// copy of the menu: the click with the slots it changed and the carried
/// stack after it, and whether a click dropped a stack. Every drop of
/// `AbstractContainerMenu.doClick` (a THROW, a click outside the window, a
/// swap or a result move with no room left) is `Prediction.PREDICTED`.
/// None when the engine refuses a click.
fn predict_clicks(
    m: &mut horizon_client::menu::Menu,
    env: &horizon_client::menu::Env,
    clicks: &[(i16, i32, Input)],
) -> Option<(Vec<Predicted>, bool)> {
    let mut predicted = Vec::with_capacity(clicks.len());
    let mut dropped = false;
    for &(slot, button, input) in clicks {
        let before = menu_items(m);
        let thrown = m.dropped.len();
        m.click(env, i32::from(slot), button, input).ok()?;
        dropped |= m.dropped.len() > thrown;
        let changed = changed_slots(&before, m);
        predicted.push((slot, button, input, changed, m.carried.clone()));
    }
    Some((predicted, dropped))
}

pub(super) fn menu_items(m: &horizon_client::menu::Menu) -> Vec<Option<ItemStack>> {
    (0..m.slot_count()).map(|i| m.slot(i).cloned()).collect()
}

/// The slots whose stack is no longer `before`'s (`ItemStack.matches`).
/// `MultiPlayerGameMode.handleContainerInput`'s changed slots, in the order the packet writes them.
fn changed_slots(
    before: &[Option<ItemStack>],
    m: &horizon_client::menu::Menu,
) -> Vec<(i16, Option<ItemStack>)> {
    let changed: Vec<i16> = (0..m.slot_count())
        .filter(|&i| m.slot(i) != before.get(i).and_then(Option::as_ref))
        .map(|i| i as i16)
        .collect();
    open_hash_order(&changed)
        .into_iter()
        .map(|i| (i, m.slot(i as usize).cloned()))
        .collect()
}

/// The order a new fastutil `Int2ObjectOpenHashMap` iterates `keys` put in this order: vanilla
/// collects a click's changed slots in one and writes them as it iterates. The key 0 comes first,
/// then the table from its last position down; a key sits at `HashCommon.mix(key) & mask` or the
/// next free position. The table starts at 32 (16 expected, load factor 0.75) and doubles when an
/// insert finds it holding 24 (`size++ >= maxFill`), re-inserting from its last position down.
fn open_hash_order(keys: &[i16]) -> Vec<i16> {
    fn mix(key: i16) -> usize {
        let h = i32::from(key).wrapping_mul(0x9E37_79B9_u32 as i32);
        (h ^ ((h as u32) >> 16) as i32) as u32 as usize
    }
    fn put(table: &mut [Option<i16>], key: i16) {
        let mask = table.len() - 1;
        let mut pos = mix(key) & mask;
        while table[pos].is_some() {
            pos = (pos + 1) & mask;
        }
        table[pos] = Some(key);
    }
    // HashCommon.maxFill(n, 0.75): min(ceil(n * 0.75), n - 1).
    let max_fill = |n: usize| (n * 3).div_ceil(4).min(n - 1);
    let mut table = vec![None; 32];
    let (mut zero, mut size) = (false, 0);
    for &key in keys {
        if key == 0 {
            zero = true;
        } else {
            put(&mut table, key);
        }
        size += 1;
        // `size++ >= maxFill`: the size before this key.
        if size > max_fill(table.len()) {
            // HashCommon.arraySize(size + 1, 0.75): the power of two holding size + 1 at that load.
            let n = ((size + 1) * 4).div_ceil(3).next_power_of_two();
            let mut grown = vec![None; n];
            for &key in table.iter().rev().flatten() {
                put(&mut grown, key);
            }
            table = grown;
        }
    }
    let rest = table.iter().rev().flatten().copied();
    zero.then_some(0).into_iter().chain(rest).collect()
}

/// ServerboundContainerClickPacket's `ContainerInput` ordinal.
pub(super) fn input_ordinal(input: Input) -> i32 {
    match input {
        Input::Pickup => 0,
        Input::QuickMove => 1,
        Input::Swap => 2,
        Input::Clone => 3,
        Input::Throw => 4,
        Input::QuickCraft => 5,
        Input::PickupAll => 6,
    }
}

mod dry_level;
use dry_level::DryLevel;

type DrySound = (
    horizon_core::position::BlockPos,
    std::sync::Arc<str>,
    f32,
    f32,
);
/// The blocks a use on the dry level set.
type DryChanges = Vec<(horizon_core::position::BlockPos, BlockStateId)>;
/// What a use on the dry level changed, and the sounds it played.
type DryUse = (DryChanges, Vec<DrySound>);
/// Where `BlockItem.place` on the dry level placed, and the blocks it set.
type DryPlace = (horizon_core::position::BlockPos, DryChanges);

/// The blocks a guest's hand flips that its use predicts (and hears at once): a wooden or
/// copper door, trapdoor or fence gate, a lever, a button or a cauldron, whose fill level
/// changes as the bucket, bottle or dye lands. An iron door or trapdoor has no hand use.
pub(super) fn hand_operated(name: &str) -> bool {
    let name = name.trim_start_matches("minecraft:");
    ((name.ends_with("_door") || name.ends_with("_trapdoor")) && !name.starts_with("iron_"))
        || name.ends_with("_fence_gate")
        || name.ends_with("_button")
        || name == "lever"
        || name.ends_with("cauldron")
}

/// `InteractionHand` on the wire.
fn hand_id(hand: Hand) -> i32 {
    match hand {
        Hand::Main => 0,
        Hand::Off => 1,
    }
}

/// A hit, a use, a block placed or broken, an item dropped: what Free Direction steering
/// rests after (`Game::send`).
fn is_interaction(packet: &S) -> bool {
    matches!(
        packet,
        S::PlayerAction(_) | S::Attack(_) | S::Interact(_) | S::UseItemOn(_) | S::UseItem(_)
    )
}

/// Whether Free Direction holds `packet` for the next tick (`Game::send`): an interaction
/// made while the last move steered (`steered`), and whatever comes after one held
/// (`queued`), in order. Digging (START_DESTROY_BLOCK to STOP_DESTROY_BLOCK, actions 0 to 3)
/// goes at once: this client has broken the block on its own level already, and this tick's
/// move may go into the space it left (flying up against the block just dug), which the
/// server and its anticheat take only once they have the break, as vanilla sends it before
/// the move. The server reads no look from it, and Grim's RotationBreak checks it against the
/// look of the move after it, the view's (the steering rests from it on).
fn waits_for_the_view(packet: &S, steered: bool, queued: bool) -> bool {
    let digging = matches!(packet, S::PlayerAction(a) if (0..=3).contains(&a.action));
    (is_interaction(packet) && !digging && steered) || queued
}

/// Whether `Item.use` of `info` ends `Minecraft.startUseItem`'s click, so the
/// off hand gets no turn. `FireworkRocketItem.use` answers SUCCESS while the
/// player glides (the rocket boosts them) and PASS otherwise.
fn use_ends_click(info: &horizon_client::items::ItemInfo, fall_flying: bool) -> bool {
    item_has_use(info) || (fall_flying && &*info.id == "minecraft:firework_rocket")
}

/// Whether `Item.use` of this item answers SUCCESS on this client, which
/// swings (`MultiPlayerGameMode.useItem`'s `result.shouldSwing()`), rather
/// than CONSUME, which only ends the click: a bow, crossbow, trident or
/// spear starting its charge, `ItemUtils.startUsingInstantly`'s food, drink,
/// spyglass, shield and goat horn, and the ender eye's SUCCESS_SERVER (the
/// server's swing source, not this client's).
fn remote_use_swings(held: &ItemStack, consumable: bool) -> bool {
    let id = &*held.item;
    !matches!(
        id,
        "minecraft:bow"
            | "minecraft:crossbow"
            | "minecraft:trident"
            | "minecraft:spyglass"
            | "minecraft:goat_horn"
            | "minecraft:ender_eye"
            | "minecraft:brush"
    ) && !consumable
        && horizon_client::spear::Kinetic::of(id).is_none()
}

/// Whether an object interaction on the client predicts a hand swing (`InteractionResult.SUCCESS`).
/// Boarding a minecart answers CONSUME and does not swing.
fn remote_object_swings(response: Option<bool>, is_boarding: bool) -> bool {
    response.unwrap_or(!is_boarding)
}

/// Whether right-clicking an item frame passes (`InteractionResult.PASS`).
/// A fixed frame or an empty frame when clicking with an empty hand passes.
fn item_frame_interact_passes(fixed: bool, frame_has_item: bool, hand_has_item: bool) -> bool {
    fixed || (!frame_has_item && !hand_has_item)
}

/// Whether right-clicking a boat passes (`InteractionResult.PASS`) on a client.
/// `AbstractBoat.interact` passes when sneaking or out of control (60 ticks), and
/// answers SUCCESS otherwise without checking for room (only the server's
/// `startRiding` does). `AbstractChestBoat.interact` opens the chest instead of
/// passing, unless it is out of control, not sneaked at and has room.
fn boat_interact_passes(chest: bool, sneaking: bool, out_of_control: bool, has_room: bool) -> bool {
    if chest {
        !sneaking && out_of_control && has_room
    } else {
        sneaking || out_of_control
    }
}

/// Whether `Item.use` usually acts on this item (eats, drinks, equips, raises,
/// throws, shoots, fills or places), which ends `Minecraft.startUseItem` at
/// its hand. Tools and materials pass, and the off hand gets its turn. The
/// server decides; this is the client's guess of whether to go on. A carrot or warped fungus
/// on a stick is not here: `FoodOnAStickItem.use` passes on a client, and only the server's
/// answer to a boost that can run swings the hand.
pub(super) fn item_has_use(info: &horizon_client::items::ItemInfo) -> bool {
    let id = info.id.trim_start_matches("minecraft:");
    id.ends_with("_spear")
        || info.is("minecraft:fishing_rod") // a content mod's rod too
        || info.consumable.is_some()
        || info.equip_slot.is_some()
        || info.blocks_attacks.is_some()
        || horizon_client::items::bucket_contents(&info.id).is_some()
        || id.ends_with("_boat")
        || id.ends_with("_raft")
        || id.ends_with("_bundle")
        || matches!(
            id,
            "bow"
                | "crossbow"
                | "trident"
                | "spyglass"
                | "snowball"
                | "egg"
                | "blue_egg"
                | "brown_egg"
                | "ender_pearl"
                | "ender_eye"
                | "wind_charge"
                | "splash_potion"
                | "lingering_potion"
                | "experience_bottle"
                | "bucket"
                | "glass_bottle"
                | "map"
                | "writable_book"
                | "written_book"
                | "goat_horn"
                | "bundle"
                | "brush"
                | "knowledge_book"
        )
}

/// The view distance a client asks for: its render distance, which Horizon
/// lets go past vanilla's 32 (a vanilla server still caps it at its own).
fn client_view_distance(render_distance: i32) -> i8 {
    render_distance.clamp(2, i32::from(i8::MAX)) as i8
}

/// `ClientPacketListener.handleBlockEntityData` loads the tag into the existing
/// block entity, so the state the client builds itself stays: copied from `old`
/// into `new`, the block entity decoded from the tag.
fn keep_client_state(old: Option<&BlockEntity>, new: &mut BlockEntity) {
    match (old, new) {
        // The turn of a vault's item and a trial spawner's mob (VaultClientData).
        (Some(BlockEntity::Vault(old)), BlockEntity::Vault(new)) => {
            new.spin = old.spin;
            new.o_spin = old.o_spin;
        }
        (Some(BlockEntity::TrialSpawner(old)), BlockEntity::TrialSpawner(new)) => {
            new.spin = old.spin;
            new.o_spin = old.o_spin;
        }
        // A beacon's beam (`BeaconBlockEntity.loadAdditional` reads only the
        // powers): a fresh one draws nothing until the next tick walks it again.
        (Some(BlockEntity::Beacon(old)), BlockEntity::Beacon(new)) => {
            new.sections.clone_from(&old.sections);
            new.checking.clone_from(&old.checking);
            (new.last_check_y, new.levels) = (old.last_check_y, old.levels);
        }
        _ => {}
    }
}

/// `LocalPlayer.onSyncedDataUpdated(DATA_LIVING_ENTITY_FLAGS)`: the server's
/// flags (1 using, 2 the off hand) start a use the client is not running
/// (`startUsingItem`, which needs an item in that hand), and stop one it is
/// (`stopUsingItem`). `using` is the slot in use and its ticks.
fn synced_local_use(
    using: Option<(usize, u32)>,
    flags: i8,
    selected: usize,
    has_item: impl Fn(usize) -> bool,
) -> Option<(usize, u32)> {
    let server_using = flags & 1 != 0;
    match using {
        None if server_using => {
            let slot = if flags & 2 != 0 {
                super::shield::OFFHAND_SLOT
            } else {
                selected
            };
            has_item(slot).then_some((slot, 0))
        }
        Some(_) if !server_using => None,
        using => using,
    }
}

/// The game as the tick's input sees it.
struct RemoteInputWorld<'g, 'a>(&'g mut Game<'a>);

struct RemoteUseWorld<'g, 'a>(&'g mut Game<'a>, mod_api::InteractionAttempt);

impl horizon_client::interaction_input::TargetUseWorld for RemoteUseWorld<'_, '_> {
    fn use_entity(&mut self) -> bool {
        #[cfg(target_os = "android")]
        if self.0.mobile.touch_shield {
            return false;
        }
        self.0.remote_use_entity(Hand::Main)
    }
    fn use_block(&mut self) -> bool {
        #[cfg(target_os = "android")]
        if self.0.mobile.touch_shield {
            let transformer = self
                .0
                .stack_in(self.0.selected)
                .and_then(|s| self.0.items().get(&s.item))
                .is_some_and(|i| i.block_transformer.is_some());
            if self.0.mobile.controls.settings.shield_button
                || self.0.mobile.controls.shield_held()
                || !transformer
                || self.0.pick().is_none()
            {
                return false;
            }
        }
        self.0.remote_use_block(Hand::Main)
    }
    fn use_item(&mut self) {
        self.0.remote_use_hands(self.1);
    }
}

impl horizon_client::interaction_input::InteractionWorld for RemoteInputWorld<'_, '_> {
    fn pick_hit_result(&mut self) {
        self.0.pick_tick_targets();
    }
    fn release_hit_result(&mut self) {
        self.0.release_tick_targets();
    }
    fn using_item(&self) -> bool {
        self.0.using.is_some()
    }
    fn release_using(&mut self) {
        if let Some((slot, ticks)) = self.0.using
            && self.0.mod_use_tick(slot, ticks, true)
        {
            return;
        }
        self.0.release_remote_use();
    }
    fn start_attack(&mut self) -> bool {
        self.0.remote_start_attack()
    }
    fn start_use(&mut self, held: bool) {
        let attempt = if held {
            mod_api::InteractionAttempt::Held
        } else {
            mod_api::InteractionAttempt::Press
        };
        self.0.remote_use(attempt);
    }
    fn pick(&mut self) {
        // Minecraft.pickBlockOrEntity: an entity before the block is asked
        // for (`handlePickItemFromEntity`), else the block.
        if let Some(entity) = self.0.pick_entity_hit() {
            if let Some(id) = entity {
                if self
                    .0
                    .native_interaction(
                        InteractionKey::PickBlock,
                        Hand::Main,
                        mod_api::InteractionAttempt::Press,
                    )
                    .is_some()
                {
                    return;
                }
                self.0.send(S::PickItemFromEntity(play::PickItemFromEntity {
                    id,
                    include_data: self.0.has_control_down(),
                }));
            }
        } else if let Some(hit) = self.0.pick() {
            if self
                .0
                .native_interaction(
                    InteractionKey::PickBlock,
                    Hand::Main,
                    mod_api::InteractionAttempt::Press,
                )
                .is_some()
            {
                return;
            }
            self.0.send(S::PickItemFromBlock(play::PickItemFromBlock {
                pos: hit.pos,
                include_data: self.0.has_control_down(),
            }));
        }
    }
    fn use_delay(&self) -> u32 {
        self.0.use_delay
    }
    fn continue_attack(&mut self, down: bool) {
        self.0.remote_continue_attack(down);
    }
}

#[derive(Default)]
pub(super) struct ClientMenu {
    current: Option<(i32, String)>,
    pub(super) generation: u64,
}

impl ClientMenu {
    pub(super) fn set(&mut self, current: Option<(i32, String)>) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("menu generation exhausted");
        self.current = current;
    }

    pub(super) fn as_ref(&self) -> Option<&(i32, String)> {
        self.current.as_ref()
    }
}

impl Remote {
    pub(super) fn set_menu_title(&mut self, title: Option<horizon_client::text::Text>) {
        self.menu_title_snapshot = title.try_into_api().ok().flatten().map(Arc::new);
        self.menu_title = title;
    }

    pub(super) fn menu_title_snapshot(&self) -> Option<&mod_api::Text> {
        self.menu_title_snapshot.as_deref()
    }

    pub(super) fn shared_menu_title_snapshot(&self) -> Option<Arc<mod_api::Text>> {
        self.menu_title_snapshot.clone()
    }

    /// Resolve registry keys only when registries or the dimension change.
    fn resolve_clocks(&mut self) {
        self.overworld_clock = self
            .registries
            .id("minecraft:world_clock", "minecraft:overworld");
        self.default_clock = self
            .dimension
            .default_clock
            .as_deref()
            .and_then(|name| self.registries.id("minecraft:world_clock", name));
        for id in [self.overworld_clock, self.default_clock]
            .into_iter()
            .flatten()
        {
            self.clocks.entry(id).or_insert(play::ClockState {
                clock: id,
                total_ticks: 0,
                partial_tick: 0.0,
                rate: 1.0,
            });
        }
    }

    fn clock_time(&self, id: Option<i32>) -> i64 {
        id.and_then(|id| self.clocks.get(&id))
            .map_or(0, |c| c.total_ticks)
    }

    pub(super) fn overworld_clock_time(&self) -> i64 {
        self.clock_time(self.overworld_clock)
    }

    /// `Minecraft.level != null`: from `handleLogin` until the next
    /// `handleConfigurationStart` (`clearClientLevel`). Minecraft.tick sends
    /// CLIENT_TICK_END only then, so a client waiting for the server's Login
    /// (a busy server takes seconds to place a new player) sends nothing.
    pub(in crate::game) fn has_level(&self) -> bool {
        client_has_level(self.logged_in, self.reconfiguring)
    }
    /// An entity the server added to this client level: its type
    /// (`minecraft:text_display`) and its position as last sent.
    pub(super) fn tracked_entity(&self, id: i32) -> Option<(&str, [f64; 3])> {
        self.entities
            .get(&id)
            .map(|t| (t.kind.as_str(), t.position))
    }
    /// The profile of the player entity `id`: this window's own player, or
    /// another one by the UUID it was added with and its profile name.
    pub(super) fn level_player(&self, id: i32) -> Option<(u128, Option<&str>)> {
        if id == self.entity_id {
            return Some((self.profile, Some(self.profile_name.as_str())));
        }
        let uuid = *self.level_players.get(&id)?;
        Some((uuid, self.level_player_names.get(&id).map(String::as_str)))
    }
    /// `PlayerInfo.getMessageValidator().updateAndValidate(message)` for `message.sender`.
    fn validate_chat(
        &mut self,
        message: &secure_chat::Incoming,
        now: i64,
    ) -> Option<secure_chat::Validated> {
        let sender = message.sender;
        if let Some(validator) = self.chat_validators.get_mut(&sender) {
            return validator.update_and_validate(message, now);
        }
        let (mut validator, settled) = self.new_chat_validator(sender);
        let validated = validator.update_and_validate(message, now);
        // While the services' keys are still on their way the validator is a stopgap, not kept.
        if settled {
            self.chat_validators.insert(sender, validator);
        }
        validated
    }
    /// `ClientPacketListener.initializeChatSession` for `sender`'s entry: a session the services'
    /// keys vouch for makes a keyed validator, anything else the fallback of
    /// `enforcesSecureChat()` (`canValidateProfileKeys() && serverEnforcesSecureChat`). `false`
    /// while the keys are not known yet.
    fn new_chat_validator(&self, sender: u128) -> (secure_chat::MessageValidator, bool) {
        let keys = self.services_keys.lock().unwrap_or_else(|e| e.into_inner());
        let (can_validate, settled) = match &*keys {
            ServicesKeys::Ready(_) => (true, true),
            ServicesKeys::Missing => (false, true),
            ServicesKeys::Pending => (false, false),
        };
        let fallback =
            secure_chat::MessageValidator::fallback(can_validate && self.enforces_secure_chat);
        let (ServicesKeys::Ready(keys), Some(Some((id, expires_at, key, signature)))) =
            (&*keys, self.chat_sessions.get(&sender))
        else {
            if matches!(&*keys, ServicesKeys::Missing)
                && self.chat_sessions.get(&sender).is_some_and(Option::is_some)
            {
                log::warn!(
                    "Ignoring chat session from {sender:032x} due to missing Services public key"
                );
            }
            return (fallback, settled);
        };
        let profile_key = secure_chat::ProfileKey {
            expires_at: *expires_at,
            key: key.clone(),
            signature: signature.clone(),
        };
        match secure_chat::ChatSession::validate(*id, profile_key, keys, sender) {
            Ok(session) => (secure_chat::MessageValidator::keyed(session), true),
            Err(why) => {
                log::error!("Failed to validate profile key for player {sender:032x}: {why}");
                (fallback, true)
            }
        }
    }
    /// How far the server sends chunks: its view distance and the ring
    /// around (`ChunkTrackingView`), at least two.
    pub(super) fn sent_radius(&self) -> i32 {
        self.view_distance.max(2).saturating_add(1)
    }
    /// A player entity's profile name: the one it was created with, else its
    /// player-list entry's (an entity the list named after it spawned).
    fn level_player_name(&self, id: i32, uuid: &u128) -> Option<&str> {
        self.level_player_names
            .get(&id)
            .or_else(|| self.players.get(uuid))
            .map(String::as_str)
    }
    /// Listed UUIDs always have a tab entry; removal clears both together.
    pub(super) fn listed_player_count(&self) -> usize {
        self.listed.len()
    }
    /// The local player's profile UUID.
    pub(super) fn profile(&self) -> u128 {
        self.profile
    }
    /// Another player's profile UUID, by their entity id (None for the local player).
    pub(super) fn player_uuid(&self, id: u32) -> Option<u128> {
        let uuid = *self.level_players.get(&(id as i32))?;
        (uuid != self.profile).then_some(uuid)
    }
    /// The entity id of a player in the level, by profile UUID.
    pub(super) fn player_entity(&self, uuid: u128) -> Option<u32> {
        self.level_players
            .iter()
            .find(|(_, u)| **u == uuid)
            .map(|(id, _)| *id as u32)
    }
    /// The kind and last sent position of a tracked server entity.
    pub(super) fn tracked(&self, id: i32) -> Option<(&str, [f64; 3])> {
        self.entities.get(&id).map(|t| (&*t.kind, t.position))
    }
}

fn object_skin_changed(published: &HashMap<String, String>, key: &str, skin: &str) -> bool {
    published.get(key).is_none_or(|previous| previous != skin)
}

/// The textures to resolve skins from: every list entry's, and for a player
/// entity whose entry is gone the profile it kept. `kept` first takes, for each
/// player entity without one, its entry's textures (`AbstractClientPlayer.getPlayerInfo`
/// looks the PlayerInfo up once and holds it, removed from the list or not).
fn skin_entries(
    level_players: &HashMap<i32, u128>,
    listed: &HashMap<u128, horizon_client::skins::Textures>,
    kept: &mut HashMap<i32, horizon_client::skins::Textures>,
) -> Vec<(u128, horizon_client::skins::Textures)> {
    for (id, uuid) in level_players {
        if !kept.contains_key(id)
            && let Some(textures) = listed.get(uuid)
        {
            kept.insert(*id, textures.clone());
        }
    }
    let mut entries: Vec<_> = listed.iter().map(|(u, t)| (*u, t.clone())).collect();
    for (id, textures) in kept.iter() {
        if let Some(uuid) = level_players.get(id)
            && !listed.contains_key(uuid)
            && !entries.iter().any(|(u, _)| u == uuid)
        {
            entries.push((*uuid, textures.clone()));
        }
    }
    entries
}

/// `AttributeInstance.calculateValue` before the attribute's range: the base,
/// `ADD_VALUE`s, then `ADD_MULTIPLIED_BASE`s of that, then each
/// `ADD_MULTIPLIED_TOTAL`.
fn snapshot_value(s: &play::AttributeSnapshot) -> f64 {
    let amounts = |op: i32| {
        s.modifiers
            .iter()
            .filter(move |m| m.operation == op)
            .map(|m| m.amount)
    };
    let base = s.base + amounts(0).sum::<f64>();
    let mut value = base + amounts(1).map(|a| base * a).sum::<f64>();
    for amount in amounts(2) {
        value *= 1.0 + amount;
    }
    value
}

/// `Util.getMillis` for the level load tracker's timeouts.
fn client_millis() -> u64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    ORIGIN
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// What `ChatScreen.handleChatInput` sends of typed chat once `ClientChatEvent` (`ask`, with
/// the outgoing message and the cancel flag) had its say: nothing when canceled or emptied.
fn chat_after_event(message: &str, ask: impl FnOnce(&mut String, &mut bool)) -> Option<String> {
    let mut outgoing = message.to_owned();
    let mut cancel = false;
    ask(&mut outgoing, &mut cancel);
    (!cancel && !outgoing.is_empty()).then_some(outgoing)
}
/// `Entity.isOnFire` for the window's player: while the server's flag is set
/// the predicted tick's countdown never takes `fire_ticks` below 1.
fn hold_server_fire(fire_ticks: i32, server_on_fire: bool) -> i32 {
    if server_on_fire {
        fire_ticks.max(1)
    } else {
        fire_ticks
    }
}

/// `BlockStatePredictionHandler.ServerVerifiedState`: the sequence of the action
/// that last changed the block, the state the server last said it has, and where
/// the player stood when this client first predicted a change there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Prediction {
    pub sequence: i32,
    pub state: i32,
    pub player_pos: [f64; 3],
}

/// `Entity.isColliding`: whether the collision `shape` of a block at `pos`
/// meets `entity`'s box. Shapes are block-local, so they move onto the block
/// first (`VoxelShape.move`), and any box that overlaps is a collision
/// (`Shapes.joinIsNotEmpty` with `BooleanOp.AND`).
pub(super) fn shape_meets(
    shape: &horizon_core::geometry::VoxelShape,
    pos: [i32; 3],
    entity: horizon_core::geometry::Aabb,
) -> bool {
    let at = Vec3::new(f64::from(pos[0]), f64::from(pos[1]), f64::from(pos[2]));
    shape
        .boxes()
        .iter()
        .any(|b| b.move_by(at).intersects(entity))
}

/// `ClientLevel.syncBlockState`'s snap: a restored block that meets the player
/// sends it back to the position the prediction remembered, unless the player
/// was teleported after that prediction was made
/// (`lastTeleportSequence < sequence`: the remembered position is stale).
pub(super) fn snap_target(
    recorded: [f64; 3],
    last_teleport_sequence: i32,
    sequence: i32,
    meets: bool,
) -> Option<[f64; 3]> {
    (meets && last_teleport_sequence < sequence).then_some(recorded)
}

/// [`Remote::has_level`]: logged in (a Login arrived) and no reconfiguration
/// since (`reconfiguring` is set by StartConfiguration, cleared by the next
/// Login).
fn client_has_level(logged_in: bool, reconfiguring: Option<u64>) -> bool {
    logged_in && reconfiguring.is_none()
}

/// `ClientChunkCache.Storage.inRange`: whether column `x`, `z` is within `calculateStorageRange`
/// (the server's radius, at least two, plus three) of the view centre. The radius and the centre
/// are the server's to send, any `i32`: the distances are taken without overflow, which panicked
/// a debug build and, wrapping in release, dropped every loaded chunk.
fn in_storage_range(view_distance: i32, center: (i32, i32), x: i32, z: i32) -> bool {
    let range = view_distance.max(2).saturating_add(3).unsigned_abs();
    x.abs_diff(center.0) <= range && z.abs_diff(center.1) <= range
}

// A packet's removals (`RemoveEntities`), apart: this file may not grow.
#[cfg(test)]
mod interaction_event_tests;

#[cfg(test)]
mod native_entity_tests;

#[cfg(test)]
mod effect_blend_tests;
#[cfg(test)]
mod level_sound_tests;

#[cfg(test)]
mod attachment_query_tests;
#[cfg(test)]
mod tests {
    use super::super::api_values::TryIntoApi;

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn menu_title_snapshot_tracks_open_replacement_and_close() {
        super::super::native_entity_refs::tests::with_client_game(|game| {
            let (client, _server) = horizon_protocol::connection::Pipe::pair();
            game.connect_local(client, "menu-title-test").unwrap();
            let remote = game.remote.as_mut().unwrap();
            for value in [Some("first"), Some("replacement"), None] {
                remote.set_menu_title(value.map(horizon_client::text::Text::literal));
                assert_eq!(
                    remote.menu_title_snapshot(),
                    remote.menu_title.try_into_api().unwrap().as_ref()
                );
            }
        });
    }
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn menu_title_snapshot_follows_packet_and_close_paths() {
        use horizon_core::nbt::Tag;
        use horizon_protocol::packets::play::{self, Clientbound as C};
        use std::sync::Arc;

        super::super::native_entity_refs::tests::with_client_game(|game| {
            let (client, _server) = horizon_protocol::connection::Pipe::pair();
            game.connect_local(client, "menu-title-paths").unwrap();
            let menu_id = |game: &super::Game<'_>, name: &str| {
                game.remote
                    .as_ref()
                    .unwrap()
                    .builtin
                    .get("minecraft:menu")
                    .unwrap()
                    .id(name)
                    .unwrap()
            };
            let chest = menu_id(game, "minecraft:generic_9x3");
            let lectern = menu_id(game, "minecraft:lectern");
            let open = |game: &mut super::Game<'_>, menu, title: &str| {
                game.handle_packet(C::OpenScreen(play::OpenScreen {
                    container: 7,
                    menu,
                    title: Tag::String(title.into()),
                }))
                .unwrap();
            };
            let snapshot = |game: &super::Game<'_>| {
                let shared = game
                    .remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .unwrap();
                let observed = game.test_observed_title().unwrap();
                assert!(Arc::ptr_eq(&observed, &shared));
                shared
            };
            open(game, chest, "first");
            let first = snapshot(game);
            open(game, chest, "replacement");
            let replacement = snapshot(game);
            assert!(!Arc::ptr_eq(&first, &replacement));
            assert_ne!(first, replacement);
            game.handle_packet(C::ContainerClose(play::ContainerClose { container: 7 }))
                .unwrap();
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            assert!(game.test_observed_title().is_none());
            open(game, chest, "again");
            snapshot(game);
            game.remote_close_container();
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            open(game, chest, "before reject");
            game.remote_reject_container(7);
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            open(game, chest, "before dimension change");
            spawn_menu_test(game);
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            open(game, chest, "before lectern");
            snapshot(game);
            open(game, lectern, "lectern");
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            game.remote_lectern_page(1);
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            game.handle_packet(C::ContainerClose(play::ContainerClose { container: 7 }))
                .unwrap();
            assert!(
                game.remote
                    .as_ref()
                    .unwrap()
                    .shared_menu_title_snapshot()
                    .is_none()
            );
            assert!(game.test_observed_title().is_none());
        });
    }
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn stopping_a_retained_client_dig_aborts_only_once() {
        use crate::game::{Hit, native_entity_refs::tests::with_client_game};
        use horizon_client::{chunk::Chunk, motion::PlayerMotion};
        use horizon_core::geometry::Vec3;
        use horizon_protocol::connection::Pipe;
        use mod_api::{
            ClientHost, ClientMiningPredictEvent, ClientPacketSendEvent, Flow, Hooks, ModEntry,
            testing::with_mods,
        };
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        struct CountActions {
            packets: Arc<AtomicUsize>,
            vetoes: Arc<AtomicUsize>,
        }
        #[mod_api::__subscriptions]
        impl Hooks for CountActions {
            fn client_mining_predict(
                &self,
                _: &dyn ClientHost,
                event: &mut ClientMiningPredictEvent,
            ) -> Flow {
                assert!(!event.prediction.starting);
                assert_eq!(event.prediction.pos, [0, 64, 0]);
                assert!(event.prediction.progress > 0.5);
                self.vetoes.fetch_add(1, Ordering::Relaxed);
                Flow::Cancel
            }

            fn client_packet_send(
                &self,
                _: &dyn ClientHost,
                event: &mut ClientPacketSendEvent<'_>,
            ) -> Flow {
                if event.packet.id.path() == "player_action" {
                    self.packets.fetch_add(1, Ordering::Relaxed);
                }
                Flow::Continue
            }
        }
        let count = Arc::new(AtomicUsize::new(0));
        let vetoes = Arc::new(AtomicUsize::new(0));
        let mods = vec![ModEntry::new(
            "count_actions",
            Box::new(CountActions {
                packets: count.clone(),
                vetoes: vetoes.clone(),
            }),
        )]
        .into();
        with_mods(mods, || {
            with_client_game(|game| {
                let (client, _server) = Pipe::pair();
                game.connect_local(client, "MiningTest").unwrap();
                let pos = [0, 64, 0];
                game.set_mode_id(0);
                game.player =
                    Some(PlayerMotion::new(game.motion, Vec3::new(0.5, 64.0, 3.5)).unwrap());
                game.world
                    .chunks
                    .insert((0, 0), Arc::new(Chunk::empty(0, 0)));
                let stone = game
                    .pack
                    .assets
                    .registry
                    .default_state("minecraft:stone")
                    .unwrap();
                game.world.set(0, 64, 0, stone).unwrap();
                game.tick_hit.set(Some(Some(Hit {
                    pos,
                    face: [0, 1, 0],
                    point: [0.5, 65.0, 0.5],
                    inside: false,
                })));
                let held = game.main_hand_key();
                let remote = game.remote.as_mut().unwrap();
                remote.destroying = Some((pos, 1, 0.5));
                remote.destroying_item = held;
                remote.is_destroying = true;
                game.remote_continue_attack(true);
                assert_eq!(vetoes.load(Ordering::Relaxed), 1);
                let remote = game.remote.as_ref().unwrap();
                assert!(remote.is_destroying);
                assert_eq!(remote.destroying, Some((pos, 1, 0.5)));
                assert_eq!(count.load(Ordering::Relaxed), 0);
                game.remote_continue_attack(false);
                game.remote_continue_attack(false);
                assert_eq!(
                    game.remote.as_ref().unwrap().destroying,
                    Some((pos, 1, 0.0))
                );
            })
        });
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
    /// The remote player animation uses the scoped item answer once per tick.
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn remote_player_use_animation_dispatches_once() {
        use mod_api::{ClientHost, Hooks, ItemPart, ModEntry, Mods, Registrar, UseAnimationEvent};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        #[derive(Clone)]
        struct Part(Arc<AtomicUsize>);
        #[mod_api::horizon_part]
        impl ItemPart for Part {
            fn use_animation(&self, _: &dyn ClientHost, event: &mut UseAnimationEvent<'_>) {
                self.0.fetch_add(1, Ordering::Relaxed);
                *event.animation = mod_api::UseAnimation::Drink;
            }
        }
        struct Mod(Arc<AtomicUsize>);
        #[mod_api::__subscriptions]
        impl Hooks for Mod {
            fn register(&self, r: &mut dyn Registrar, _: &mod_api::RegistrationContext) {
                let item = r.item_key("minecraft:bow");
                r.item_part(item, Part(self.0.clone()));
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mods: Mods = vec![ModEntry::new(
            "remote_animation",
            Box::new(Mod(calls.clone())),
        )]
        .into();
        crate::game::api_values::install_parts(&mods).unwrap();
        mod_api::testing::with_mods(mods, || {
            super::super::native_entity_refs::tests::with_client_game(|game| {
                let (client, _server) = horizon_protocol::connection::Pipe::pair();
     !(!menu.click(game, &[invalid]));
                assert_eq!(menu.slot(game, 36).unwrap().unwrap().count(), 3);
                assert_eq!(menu.carried(game), Some(None));
            }
            game.options.execution_role = ExecutionRole::IntegratedServer;
            assert!(!menu.click(game, &[pickup(36)]));
            game.options.execution_role = ExecutionRole::Client;
            let screen = game.screen;
            game.screen = Screen::Creative;
            assert!(!menu.click(game, &[pickup(36)]));
            game.screen = screen;
            game.remote.as_mut().unwrap().state_id = 17;
            assert!(menu.click(game, &[pickup(36)]));
            let (id, body) = reader.recv().unwrap();
            let play::Serverbound::ContainerClick(sent) =
                play::Serverbound::decode(id, &body).unwrap()
            else {
                panic!("expected the first valid click, not a refused request");
            };
            assert_eq!(
                (
                    sent.container,
                    sent.state,
                    sent.slot,
                    sent.button,
                    sent.input
                ),
                (0, 17, 36, 0, 0)
            );
            assert_eq!(sent.changed.len(), 1);
            assert_eq!(sent.changed[0].0, 36);
            assert!(sent.changed[0].1.is_none());
            assert_eq!(sent.carried.unwrap().count, 3);
            assert_eq!(menu.slot(game, 36), Some(None));
            assert_eq!(menu.carried(game).unwrap().unwrap().count(), 3);
            assert!(!menu.click(game, &[pickup(36), pickup(usize::MAX),]));
            assert_eq!(menu.slot(game, 36), Some(None));
            assert_eq!(menu.carried(game).unwrap().unwrap().count(), 3);
            assert!(menu.click(
                game,
                &[MenuClick::Drag {
                    kind: DragKind::Split,
                    slots: vec![37]
                },]
            ));
            assert_eq!(menu.slot(game, 37).unwrap().unwrap().count(), 3);
            assert_eq!(menu.carried(game), Some(None));
            assert!(menu.click(game, &[pickup(37)]));
            assert!(menu.click(game, &[pickup(36)]));
            assert_eq!(menu.carried(game), Some(None));
            assert_eq!(menu.slot(game, 36).unwrap().unwrap().count(), 3);
            game.handle_packet(C::SetPlayerInventory(play::SetPlayerInventory {
                slot: 0,
                item: Some(Stack::new(game.item_id("minecraft:stone"), 2)),
            }))
            .unwrap();
            assert_eq!(menu.slot(game, 36).unwrap().unwrap().count(), 2);
            assert!(menu.click(game, &[MenuClick::QuickMove { slot: 36 }]));
            assert_eq!(menu.slot(game, 36), Some(None));
            assert_eq!(menu.slot(game, 9).unwrap().unwrap().count(), 2);
            assert!(menu.click(
                game,
                &[MenuClick::Swap {
                    slot: 9,
                    with: SwapWith::Hotbar(1)
                }]
            ));
            assert_eq!(menu.slot(game, 9), Some(None));
            assert_eq!(menu.slot(game, 37).unwrap().unwrap().count(), 2);
            assert!(menu.click(
                game,
                &[MenuClick::Throw {
                    slot: 37,
                    whole_stack: true
                }]
            ));
            assert_eq!(menu.slot(game, 37), Some(None));
            game.handle_packet(C::SetCursorItem(play::SetCursorItem {
                item: stone.clone(),
            }))
            .unwrap();
            assert_eq!(game.menu_carried(menu).unwrap().unwrap().count, 3);
            let crafting_id = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:menu")
                .unwrap()
                .id("minecraft:crafting")
                .unwrap();
            let open = || {
                C::OpenScreen(play::OpenScreen {
                    container: 7,
                    menu: crafting_id,
                    title: Tag::String("Crafting".into()),
                })
            };
            game.handle_packet(open()).unwrap();
            assert_eq!(game.menu_slot_count(menu), None);
            assert!(!menu.click(game, &[pickup(36)]));
            let crafting = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            assert_eq!(game.menu_slot_count(crafting), Some(46));
            game.handle_packet(C::ContainerSetSlot(play::ContainerSetSlot {
                container: 7,
                state: 0,
                slot: 1,
                item: stone.clone(),
            }))
            .unwrap();
            assert_eq!(game.menu_slot(crafting, 1).unwrap().unwrap().count, 3);
            game.handle_packet(C::ContainerSetSlot(play::ContainerSetSlot {
                container: 7,
                state: 0,
                slot: 0,
                item: stone,
            }))
            .unwrap();
            assert_eq!(game.menu_slot(crafting, 0).unwrap().unwrap().count, 3);
            game.handle_packet(open()).unwrap();
            assert_eq!(game.menu_slot_count(crafting), None);
            assert!(!crafting.click(game, &[pickup(1)]));
            let crafting = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            game.handle_packet(C::Respawn(play::Respawn {
                spawn: spawn.clone(),
                keep: 0,
            }))
            .unwrap();
            assert_eq!(game.menu_slot_count(crafting), None);
            assert!(!crafting.click(game, &[pickup(1)]));
            let inventory = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            assert_ne!(inventory, menu);
            let chest_id = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:menu")
                .unwrap()
                .id("minecraft:generic_9x3")
                .unwrap();
            game.handle_packet(C::OpenScreen(play::OpenScreen {
                container: 7,
                menu: chest_id,
                title: Tag::String("Chest".into()),
            }))
            .unwrap();
            let chest = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            assert_eq!(game.menu_slot_count(chest), Some(63));
            let stone = Some(Stack::new(game.item_id("minecraft:stone"), 4));
            game.handle_packet(C::ContainerSetSlot(play::ContainerSetSlot {
                container: 7,
                state: 0,
                slot: 2,
                item: stone,
            }))
            .unwrap();
            assert_eq!(game.menu_slot(chest, 2).unwrap().unwrap().count, 4);
            assert_eq!(game.menu_slot(chest, 3), Some(None));
            game.handle_packet(C::SetCursorItem(play::SetCursorItem { item: None }))
                .unwrap();
            assert!(chest.click(game, &[pickup(2)]));
            assert_eq!(chest.slot(game, 2), Some(None));
            assert_eq!(chest.carried(game).unwrap().unwrap().count(), 4);
            assert!(chest.click(game, &[pickup(2)]));
            assert_eq!(chest.slot(game, 2).unwrap().unwrap().count(), 4);
            assert_eq!(chest.carried(game), Some(None));
            for expected in [
                (0, 17, -999, 0, 5),
                (0, 17, 37, 1, 5),
                (0, 17, -999, 2, 5),
                (0, 17, 37, 0, 0),
                (0, 17, 36, 0, 0),
                (0, 17, 36, 0, 1),
                (0, 17, 9, 1, 2),
                (0, 17, 37, 1, 4),
                (7, 0, 2, 0, 0),
                (7, 0, 2, 0, 0),
            ] {
                let sent = (0..32)
                    .find_map(|_| {
                        let (id, body) = reader.recv().unwrap();
                        match play::Serverbound::decode(id, &body).unwrap() {
                            play::Serverbound::ContainerClick(sent) => Some(sent),
                            _ => None,
                        }
                    })
                    .expect("expected click packet");
                assert_eq!(
                    (
                        sent.container,
                        sent.state,
                        sent.slot,
                        sent.button,
                        sent.input
                    ),
                    expected
                );
                if expected == (0, 17, -999, 2, 5) {
                    assert!(sent.carried.is_none());
                    assert_eq!(sent.changed.len(), 1);
                    assert_eq!(sent.changed[0].0, 37);
                    assert_eq!(sent.changed[0].1.as_ref().unwrap().count, 3);
                }
            }
            reader
                .set_read_timeout(Some(std::time::Duration::from_millis(20)))
                .unwrap();
            assert!(reader.recv().unwrap_err().is_timed_out());
            game.remote
                .as_mut()
                .unwrap()
                .menu
                .set(Some((7, "minecraft:unknown".into())));
            assert!(
                game.local_player()
                    .and_then(|player| game.player_container_menu(player))
                    .is_none()
            );
            assert_eq!(game.menu_slot_count(inventory), None);
            assert_eq!(game.menu_slot_count(chest), None);
            game.remote.as_mut().unwrap().menu.set(None);
            let before_replacement = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            let mut other = spawn;
            other.dimension = "test:other_overworld".into();
            game.enter_remote_dimension(&other, true, Reason::Other);
            assert_eq!(game.menu_slot_count(before_replacement), None);
            assert!(!before_replacement.click(game, &[pickup(36)]));
            let before_logout = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            game.client_logging_out();
            assert_eq!(game.menu_slot_count(before_logout), None);
            assert!(!before_logout.click(game, &[pickup(36)]));
            assert!(
                game.local_player()
                    .and_then(|player| game.player_container_menu(player))
                    .is_none()
            );
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "MenuReadTest").unwrap();
            game.enter_remote_dimension(&other, false, Reason::Other);
            let remote = game.remote.as_mut().unwrap();
            remote.logged_in = true;
            remote.entity_id = 7;
            remote.profile = 42;
            let reconnected = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            assert_eq!(game.menu_slot_count(before_logout), None);
            assert_eq!(game.menu_slot_count(reconnected), Some(46));
            assert_ne!(reconnected, before_logout);
        });
    }

    /// `MenuRef::close` and `click_button` go through the screen's own close
    /// and the button packet, and refuse a menu that is not open or stale.
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_menu_close_and_button_click_reach_the_server() {
        use super::super::{
            Screen, execution_role::ExecutionRole, native_entity_refs::tests::with_game,
        };
        use horizon_core::nbt::Tag;
        use horizon_protocol::packets::play::{self, Clientbound as C};

        with_game(|game| {
            let (mut reader, _writer) = connect_menu_test(game);
            spawn_menu_test(game);
            let sent = |reader: &mut horizon_protocol::connection::ConnReader| {
                (0..32).find_map(|_| {
                    let (id, body) = reader.recv().ok()?;
                    match play::Serverbound::decode(id, &body).unwrap() {
                        p @ (play::Serverbound::ContainerClose(_)
                        | play::Serverbound::ContainerButtonClick(_)) => Some(p),
                        _ => None,
                    }
                })
            };
            // The inventory menu while playing: no screen to close, no buttons.
            let inventory = game
                .local_player()
                .and_then(|player| game.player_container_menu(player))
                .unwrap();
            assert_eq!(game.screen, Screen::Playing);
            assert!(!inventory.close(game));
            assert!(!inventory.click_button(game, 0));
            let loom = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:menu")
                .unwrap()
                .id("minecraft:loom")
                .unwrap();
            let open = |game: &mut super::Game<'_>| {
                game.handle_packet(C::OpenScreen(play::OpenScreen {
                    container: 5,
                    menu: loom,
                    title: Tag::String("Loom".into()),
                }))
                .unwrap();
                game.local_player()
                    .and_then(|player| game.player_container_menu(player))
                    .unwrap()
            };
            let menu = open(game);
            assert_eq!(game.screen, Screen::Inventory);
            // The old inventory handle is stale now that the loom replaced it.
            assert!(!inventory.close(game));
            assert!(!inventory.click_button(game, 0));
            game.options.execution_role = ExecutionRole::IntegratedServer;
            assert!(!menu.click_button(game, 3));
            assert!(!menu.close(game));
            game.options.execution_role = ExecutionRole::Client;
            assert_eq!(game.screen, Screen::Inventory);
            assert!(menu.click_button(game, 3));
            let Some(play::Serverbound::ContainerButtonClick(click)) = sent(&mut reader) else {
                panic!("expected the button click, not a refused request");
            };
            assert_eq!((click.container, click.button), (5, 3));
            assert!(menu.close(game));
            assert_eq!(game.screen, Screen::Playing);
            let Some(play::Serverbound::ContainerClose(close)) = sent(&mut reader) else {
                panic!("expected the close");
            };
            assert_eq!(close.container, 5);
            assert!(game.remote.as_ref().unwrap().menu.as_ref().is_none());
            // Closed: the handle is stale and nothing more is sent.
            assert!(!menu.close(game));
            assert!(!menu.click_button(game, 3));
            // A server-side close leaves a reopened menu's old handle stale too.
            let first = open(game);
            game.handle_packet(C::ContainerClose(play::ContainerClose { container: 5 }))
                .unwrap();
            let second = open(game);
            assert!(!first.close(game));
            assert!(!first.click_button(game, 3));
            assert!(second.close(game));
            reader
                .set_read_timeout(Some(std::time::Duration::from_millis(20)))
                .unwrap();
            let Some(play::Serverbound::ContainerClose(close)) = sent(&mut reader) else {
                panic!("expected the second close");
            };
            assert_eq!(close.container, 5);
            assert!(sent(&mut reader).is_none());
        });
    }

    /// A client names the level it shows by the server's key, even where `Dim` has
    /// no name for it.
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_handles_reach_a_custom_level() {
        use super::super::{execution_role::ExecutionRole, native_entity_refs::tests::with_game};
        use horizon_client::level_loading::Reason;
        use horizon_protocol::{connection::Pipe, packets::play::SpawnInfo};
        use mod_api::{DimensionKey, LevelRef};
        use std::sync::Arc;

        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            let (client, _server) = Pipe::pair();
            use horizon_world::dimension_type::DimensionFacts;
            let stale_key = horizon_core::LevelKey::parse("test:stale_guest_facts").unwrap();
            let mut stale = DimensionFacts::overworld().clone();
            stale.coordinate_scale = 19.0;
            DimensionFacts::set_levels(vec![(stale_key, stale)]);
            game.connect_local(client, "CustomLevelTest").unwrap();
            assert_eq!(game.dimension_facts_for(stale_key).coordinate_scale, 1.0);
            let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
            let json = serde_json::json!({"min_y": 0, "height": 256,
                "coordinate_scale": 4.0, "ambient_light": 0.3,
                "attributes": {"minecraft:gameplay/fast_lava": true}});
            registries.registries.insert(
                "minecraft:dimension_type".into(),
                vec![(
                    "test:type".into(),
                    Some(horizon_client::scoreboard::json_to_tag(&json)),
                )],
            );
            game.set_registries(Arc::new(registries));
            let spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "test:pocket".into(),
                seed: 0,
                game_type: 1,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            game.remote.as_mut().unwrap().logged_in = true;

            let level = game.level("test:pocket").unwrap();
            assert_ne!(level.dimension(), DimensionKey::OVERWORLD);
            assert_eq!(level.dimension().id(game).as_deref(), Some("test:pocket"));
            assert!(game.level("minecraft:overworld").is_none());
            assert!(level.height(game).is_some());
            let facts = level.dimension_type(game).unwrap();
            assert_eq!(facts.id, "test:type");
            assert_eq!(facts.coordinate_scale, 4.0);
            assert_eq!(facts.ambient_light, 0.3);
            assert!(facts.fast_lava);
            assert_eq!(game.dimension_type().coordinate_scale, 4.0);
            let overworld = LevelRef::new(level.epoch(), DimensionKey::OVERWORLD);
            assert_eq!(overworld.height(game), None);
            // An equal JSON body with a different type name must not hit the old cache.
            let mut renamed = (*game.remote.as_ref().unwrap().registries).clone();
            renamed
                .registries
                .get_mut("minecraft:dimension_type")
                .unwrap()[0]
                .0 = "test:renamed".into();
            game.set_registries(Arc::new(renamed));
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            assert_eq!(game.dimension_type().id, "test:renamed");
            // Reject invalid server numbers and enums without failing the join.
            for (field, value) in [
                ("coordinate_scale", serde_json::json!(0.0)),
                ("coordinate_scale", serde_json::json!("NaN")),
                ("coordinate_scale", serde_json::Value::Null),
                ("ambient_light", serde_json::json!("NaN")),
                ("ambient_light", serde_json::Value::Null),
                ("monster_spawn_block_light_limit", serde_json::json!(16)),
                ("monster_spawn_block_light_limit", serde_json::json!(-1)),
                ("skybox", serde_json::json!("invalid")),
            ] {
                let mut invalid = json.clone();
                invalid[field] = value;
                let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
                registries.registries.insert(
                    "minecraft:dimension_type".into(),
                    vec![(
                        "minecraft:the_nether".into(),
                        Some(horizon_client::scoreboard::json_to_tag(&invalid)),
                    )],
                );
                game.set_registries(Arc::new(registries));
                game.enter_remote_dimension(&spawn, false, Reason::Other);
                assert_eq!(game.dimension_type().coordinate_scale, 8.0, "{field}");
                assert!(game.dimension_type().fast_lava);
            }
            let min = f64::from(1.0E-5_f32);
            let max = 30_000_000.0_f64;
            for (value, valid) in [
                (min.next_down(), false),
                (min, true),
                (min.next_up(), true),
                (max.next_down(), true),
                (max, true),
                (max.next_up(), false),
                (f64::from_bits(1), false),
            ] {
                let mut boundary = json.clone();
                boundary["coordinate_scale"] = serde_json::json!(value);
                // Vanilla uses Codec.FLOAT, not a unit-interval ambient-light codec.
                boundary["ambient_light"] = serde_json::json!(1.1);
                let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
                registries.registries.insert(
                    "minecraft:dimension_type".into(),
                    vec![(
                        "minecraft:the_nether".into(),
                        Some(horizon_client::scoreboard::json_to_tag(&boundary)),
                    )],
                );
                game.set_registries(Arc::new(registries));
                game.enter_remote_dimension(&spawn, false, Reason::Other);
                assert_eq!(
                    game.dimension_type().coordinate_scale,
                    if valid { value } else { 8.0 }
                );
                if valid {
                    assert_eq!(game.dimension_type().ambient_light, 1.1);
                }
            }
            for ambient in [1.5_f32, -0.5_f32] {
                let mut legal = json.clone();
                legal["ambient_light"] = serde_json::json!(ambient);
                let block_limit = if ambient > 0.0 { 15 } else { 0 };
                legal["monster_spawn_block_light_limit"] = serde_json::json!(block_limit);
                let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
                registries.registries.insert(
                    "minecraft:dimension_type".into(),
                    vec![(
                        "test:type".into(),
                        Some(horizon_client::scoreboard::json_to_tag(&legal)),
                    )],
                );
                game.set_registries(Arc::new(registries));
                game.enter_remote_dimension(&spawn, false, Reason::Other);
                assert_eq!(game.dimension_type().coordinate_scale, 4.0);
                assert_eq!(game.dimension_type().ambient_light, ambient);
                assert_eq!(
                    game.dimension_type().monster_spawn_block_light_limit,
                    block_limit
                );
                let level = game.level("test:pocket").unwrap();
                assert_eq!(level.dimension_type(game).unwrap().ambient_light, ambient);
            }
            // Disconnect drops the connection-owned table; the server table is untouched.
            let lifetime =
                Arc::downgrade(game.remote.as_ref().unwrap().level_facts.as_ref().unwrap());
            game.leave_server(None).unwrap();
            assert!(
                lifetime.upgrade().is_none(),
                "connection facts must be released"
            );
            assert_eq!(DimensionFacts::of_level(stale_key).coordinate_scale, 19.0);
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "SecondFactsSession").unwrap();
            let key = horizon_core::LevelKey::parse("test:pocket").unwrap();
            assert_eq!(game.dimension_facts_for(key).coordinate_scale, 1.0);
            DimensionFacts::set_levels(Vec::new());
        });
    }

    /// The `both` handle reads answer on a client from its own copy: the local
    /// player's health, effects, equipment and last death, the blocks and time
    /// of day of the level it shows and the entities it tracks.
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_handle_reads_answer_from_the_clients_copy() {
        use super::super::{execution_role::ExecutionRole, native_entity_refs::tests::with_game};
        use horizon_client::{
            chunk::Chunk, first_person::Hand, level_loading::Reason, motion::PlayerMotion,
        };
        use horizon_core::geometry::Vec3;
        use horizon_protocol::{
            connection::Pipe,
            packets::play::{self, Clientbound as C, SpawnInfo},
        };
        use mod_api::{EntityRef, EquipmentSlot, GlobalPos};
        use std::sync::Arc;

        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "HandleReadTest").unwrap();
            let spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "minecraft:overworld".into(),
                seed: 0,
                game_type: 1,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: Some(play::GlobalPos {
                    dimension: "minecraft:the_nether".into(),
                    pos: [4, 70, -9],
                }),
                portal_cooldown: 0,
                sea_level: 63,
            };
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            let remote = game.remote.as_mut().unwrap();
            remote.logged_in = true;
            remote.entity_id = 7;
            remote.profile = 42;
            game.player = Some(PlayerMotion::new(game.motion, Vec3::new(1.5, 65.0, -2.5)).unwrap());
            game.yaw = 90.0;
            game.pitch = 10.0;

            // The local player and the level it shows.
            let player = game.local_player().unwrap();
            let me = player.entity();
            let living = game.living(me).unwrap();
            let level = game.level("minecraft:overworld").unwrap();
            assert!(game.level("minecraft:the_nether").is_none());
            assert_eq!(me.position(game), Some([1.5, 65.0, -2.5]));
            assert_eq!(me.rotation(game), Some([90.0, 10.0]));
            game.player.as_mut().unwrap().velocity = Vec3::new(0.0, -0.5, 0.25);
            assert_eq!(me.velocity(game), Some([0.0, -0.5, 0.25]));

            // Last death: the login/respawn packet's, as a GlobalPos.
            let died = player.last_death_location(game).unwrap();
            assert_eq!(died.dimension().to_string(), "minecraft:the_nether");
            let pos = died.pos();
            assert_eq!([pos.x, pos.y, pos.z], [4, 70, -9]);
            let _: GlobalPos = died;

            // Health, maximum health, effects and equipment come from the packets.
            game.handle_packet(C::SetHealth(play::SetHealth {
                health: 14.0,
                food: 20,
                saturation: 5.0,
            }))
            .unwrap();
            assert_eq!(living.health(game), Some(14.0));
            assert_eq!(living.max_health(game), Some(20.0));
            let speed = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:mob_effect")
                .unwrap()
                .id("minecraft:speed")
                .unwrap();
            assert_eq!(living.effects(game), Some(Vec::new()));
            assert!(
                living
                    .effect(
                        game,
                        mod_api::__private::part_tests::effect_key("minecraft:speed")
                    )
                    .is_none()
            );
            game.handle_packet(C::UpdateMobEffect(play::UpdateMobEffect {
                id: 7,
                effect: speed,
                amplifier: 1,
                duration: 600,
                flags: 2,
            }))
            .unwrap();
            let effect = living
                .effect(
                    game,
                    mod_api::__private::part_tests::effect_key("minecraft:speed"),
                )
                .unwrap();
            assert_eq!((effect.duration, effect.amplifier), (600, 1));
            assert_eq!(living.effects(game).unwrap().len(), 1);
            let stone = Some(horizon_protocol::item::Stack::new(
                game.item_id("minecraft:stone"),
                3,
            ));
            game.handle_packet(C::SetPlayerInventory(play::SetPlayerInventory {
                slot: 0,
                item: stone,
            }))
            .unwrap();
            let held = living.hand_item(game, Hand::Main).unwrap();
            assert_eq!(
                (
                    mod_api::__private::item_key_id(held.item()).unwrap(),
                    held.count()
                ),
                ("minecraft:stone", 3)
            );
            assert_eq!(living.equipment(game, EquipmentSlot::Head), None);
            assert_eq!(living.hand_item(game, Hand::Off), None);

            // Blocks of the chunks the client holds; nothing loads another.
            game.world
                .chunks
                .insert((0, 0), Arc::new(Chunk::empty(0, 0)));
            let chunks = game.world.chunks.len();
            let at = [3, 64, 5];
            assert!(level.is_loaded(game, at));
            assert!(!level.is_loaded(game, [160, 64, 0]));
            game.handle_packet(C::BlockUpdate(play::BlockUpdate { pos: at, state: 1 }))
                .unwrap();
            assert_eq!(
                level
                    .block(game, at)
                    .and_then(mod_api::__private::block_key_id),
                Some("minecraft:stone")
            );
            assert!(level.block_state(game, at).is_some());
            assert_eq!(level.block(game, [160, 64, 0]), None);
            assert_eq!(level.block_property(game, [160, 64, 0], "facing"), None);
            assert_eq!(game.world.chunks.len(), chunks);

            // The time of day, as the server's clock last set it.
            game.set_registries({
                let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
                registries.registries.insert(
                    "minecraft:world_clock".into(),
                    vec![("minecraft:overworld".into(), None)],
                );
                Arc::new(registries)
            });
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 100,
                clocks: vec![play::ClockState {
                    clock: 0,
                    total_ticks: 30_000,
                    partial_tick: 0.0,
                    rate: 1.0,
                }],
            }))
            .unwrap();
            assert_eq!(level.day_time(game), Some(30_000));

            // An entity the server told the client of; others read None.
            let pig = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:entity_type")
                .unwrap()
                .id("minecraft:pig")
                .unwrap();
            game.handle_packet(C::AddEntity(play::AddEntity {
                id: 31,
                uuid: 9,
                kind: pig,
                position: [10.0, 64.0, 12.0],
                velocity: [0.0; 3],
                pitch: 0,
                yaw: 64,
                head_yaw: 64,
                data: 0,
            }))
            .unwrap();
            let other = game.entity(31).unwrap();
            assert_eq!(other.position(game), Some([10.0, 64.0, 12.0]));
            assert!(other.velocity(game).is_some());
            assert!(other.rotation(game).is_some());
            assert!(game.living(other).is_none(), "only the local player");
            assert!(game.entity(99).is_none());
            let untracked = EntityRef::new(me.epoch(), 99);
            assert_eq!(untracked.position(game), None);
            assert_eq!(untracked.velocity(game), None);

            // A replaced level retires the handles.
            let mut other_level = spawn.clone();
            other_level.dimension = "test:other_overworld".into();
            game.enter_remote_dimension(&other_level, true, Reason::Other);
            assert_eq!(me.position(game), None);
            assert_eq!(living.health(game), None);
            assert_eq!(level.block(game, at), None);
            assert!(!level.is_loaded(game, at));
            assert_eq!(level.day_time(game), None);
            assert_eq!(player.last_death_location(game), None);

            // The server's role keeps its own reads: a client handle names nothing there.
            game.options.execution_role = ExecutionRole::IntegratedServer;
            assert_eq!(game.native_entity_target(untracked), None);
        });
    }

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_day_time_reads_overworld_clock_across_packets_and_levels() {
        use super::super::{execution_role::ExecutionRole, native_entity_refs::tests::with_game};
        use horizon_protocol::{
            connection::Pipe,
            packets::play::{self, Clientbound as C},
        };
        use mod_api::LevelRef;
        use std::sync::Arc;

        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "ClockReadTest").unwrap();
            let mut registries = (*game.remote.as_ref().unwrap().registries).clone();
            let dimensions = vec![
                ("minecraft:overworld".into(), None),
                ("minecraft:the_end".into(), None),
                ("minecraft:the_nether".into(), None),
            ];
            registries
                .registries
                .insert("minecraft:world_clock".into(), dimensions[..2].to_vec());
            registries
                .registries
                .insert("minecraft:dimension_type".into(), dimensions);
            let registries = Arc::new(registries);
            game.set_registries(registries.clone());
            let mut spawn = play::SpawnInfo {
                dimension_type: 1,
                dimension: "minecraft:the_end".into(),
                seed: 0,
                game_type: 1,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            let login = |spawn| {
                C::Login(play::Login {
                    player_id: 7,
                    hardcore: false,
                    levels: vec!["minecraft:overworld".into(), "minecraft:the_end".into()],
                    max_players: 1,
                    chunk_radius: 2,
                    simulation_distance: 2,
                    reduced_debug_info: false,
                    show_death_screen: true,
                    limited_crafting: false,
                    spawn,
                    online_mode: false,
                    enforces_secure_chat: false,
                })
            };
            let state = |clock, total_ticks, partial_tick, rate| play::ClockState {
                clock,
                total_ticks,
                partial_tick,
                rate,
            };
            game.handle_packet(login(spawn.clone())).unwrap();
            let end = game.level("minecraft:the_end").unwrap();
            let wrong_level = LevelRef::new(end.epoch(), mod_api::DimensionKey::OVERWORLD);
            assert_eq!(wrong_level.day_time(game), None);
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 100,
                clocks: vec![state(0, 30_000, 0.0, 1.0), state(1, 100, 0.0, 1.0)],
            }))
            .unwrap();
            assert_eq!(end.day_time(game), Some(30_000));
            assert_eq!(
                game.world.day_time, 100,
                "the sky follows the End default clock"
            );

            // An Overworld-only update must not select it as the End sky clock.
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 104,
                clocks: vec![state(0, 40_000, 0.75, 0.5)],
            }))
            .unwrap();
            assert_eq!(end.day_time(game), Some(40_000));
            assert_eq!(game.world.day_time, 104);
            game.tick_remote_level().unwrap();
            assert_eq!(end.day_time(game), Some(40_001));
            assert_eq!(game.world.day_time, 105);
            game.tick_remote_level().unwrap();
            assert_eq!(end.day_time(game), Some(40_001));
            assert_eq!(game.world.day_time, 106);

            // Empty time syncs advance every retained clock by the game-time delta.
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 110,
                clocks: vec![],
            }))
            .unwrap();
            assert_eq!(end.day_time(game), Some(40_003));
            assert_eq!(game.world.day_time, 110);
            // A paused clock stays put even while the client's game time runs.
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 110,
                clocks: vec![state(0, 50_000, 0.25, 0.0)],
            }))
            .unwrap();
            game.tick_remote_level().unwrap();
            assert_eq!(end.day_time(game), Some(50_000));
            assert_eq!(game.world.day_time, 111);

            // A frozen world ticks neither clock; packets still catch both up.
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 111,
                clocks: vec![state(0, 60_000, 0.0, 2.0)],
            }))
            .unwrap();
            game.handle_packet(C::TickingState(play::TickingState {
                rate: 20.0,
                frozen: true,
            }))
            .unwrap();
            game.tick_remote_level().unwrap();
            assert_eq!(game.world.game_time, 111);
            assert_eq!(end.day_time(game), Some(60_000));
            assert_eq!(game.world.day_time, 111);
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 114,
                clocks: vec![],
            }))
            .unwrap();
            assert_eq!(end.day_time(game), Some(60_006));
            assert_eq!(game.world.day_time, 114);
            // Backward corrections use the same signed delta and fractional state.
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 113,
                clocks: vec![],
            }))
            .unwrap();
            assert_eq!(end.day_time(game), Some(60_004));
            assert_eq!(game.world.day_time, 113);
            game.handle_packet(C::TickingState(play::TickingState {
                rate: 20.0,
                frozen: false,
            }))
            .unwrap();

            // Same-level respawn keeps the epoch and clocks, but replaces the
            // local player and its mood handler, with or without entity data.
            for keep in [0, 2] {
                game.ambience.moodiness = 0.375;
                game.handle_packet(C::Respawn(play::Respawn {
                    spawn: spawn.clone(),
                    keep,
                }))
                .unwrap();
                assert_eq!(game.ambience.moodiness, 0.0);
                assert_eq!(end.day_time(game), Some(60_004));
            }
            spawn.dimension_type = 0;
            spawn.dimension = "minecraft:overworld".into();
            game.handle_packet(C::Respawn(play::Respawn {
                spawn: spawn.clone(),
                keep: 0,
            }))
            .unwrap();
            assert_eq!(end.day_time(game), None);
            let overworld = game.level("minecraft:overworld").unwrap();
            assert_eq!(overworld.day_time(game), Some(60_004));
            assert_eq!(
                game.world.day_time, 60_004,
                "select the new default from retained clocks"
            );
            game.tick_remote_level().unwrap();
            assert_eq!(overworld.day_time(game), Some(60_006));
            assert_eq!(game.world.day_time, 60_006);

            // A dimension without a default clock must not select the prior one.
            spawn.dimension_type = 2;
            spawn.dimension = "minecraft:the_nether".into();
            game.handle_packet(C::Respawn(play::Respawn {
                spawn: spawn.clone(),
                keep: 0,
            }))
            .unwrap();
            let nether = game.level("minecraft:the_nether").unwrap();
            assert_eq!(overworld.day_time(game), None);
            assert_eq!(nether.day_time(game), Some(60_006));
            assert_eq!(game.world.day_time, 0);

            // Login starts a new listener epoch, even on the same connection.
            game.handle_packet(login(spawn.clone())).unwrap();
            assert_eq!(nether.day_time(game), None);
            let fresh = game.level("minecraft:the_nether").unwrap();
            assert_eq!(fresh.day_time(game), Some(0));
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 1,
                clocks: vec![],
            }))
            .unwrap();
            assert_eq!(
                fresh.day_time(game),
                Some(1),
                "the new clock anchor and rate are fresh too"
            );
            game.handle_packet(C::SetTime(play::SetTime {
                game_time: 1,
                clocks: vec![state(0, 7_000, 0.0, 1.0)],
            }))
            .unwrap();
            assert_eq!(fresh.day_time(game), Some(7_000));
            assert_eq!(game.world.day_time, 0);
            game.client_logging_out();
            assert_eq!(fresh.day_time(game), None);
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "ClockReadTest").unwrap();
            assert_eq!(fresh.day_time(game), None);
            game.set_registries(registries);
            game.handle_packet(login(spawn)).unwrap();
            let reconnected = game.level("minecraft:the_nether").unwrap();
            assert_eq!(fresh.day_time(game), None);
            assert_eq!(reconnected.day_time(game), Some(0));
        });
    }

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_camera_entity_returns_to_player_after_removal() {
        use super::super::{
            Game, native_entity_refs::tests::with_client_game, native_lifecycle::NativeLifecycle,
        };
        use horizon_client::{level_loading::Reason, motion::PlayerMotion, title::Title};
        use horizon_core::geometry::Vec3;
        use horizon_protocol::{
            connection::Pipe,
            packets::play::{self, Clientbound as C, SpawnInfo},
        };
        use mod_api::ClientHost;

        with_client_game(|game| {
            let camera = |game: &Game<'_>| (game as &dyn ClientHost).camera_entity();
            assert!(camera(game).is_none(), "the fixture has no player");
            game.native_lifecycle = NativeLifecycle::default();
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "CameraReadTest").unwrap();
            let mut spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "minecraft:overworld".into(),
                seed: 0,
                game_type: 3,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            let remote = game.remote.as_mut().unwrap();
            remote.logged_in = true;
            remote.entity_id = 7;
            assert!(camera(game).is_none(), "a level alone has no player");
            game.player = Some(PlayerMotion::new(game.motion, Vec3::ZERO).unwrap());
            let local = camera(game).unwrap();
            assert_eq!(
                local.id(),
                7,
                "use the remote identity, not PlayerMotion's default"
            );
            assert_eq!(local.position(game), Some([0.0; 3]));

            let creeper = game
                .remote
                .as_ref()
                .unwrap()
                .builtin
                .get("minecraft:entity_type")
                .unwrap()
                .id("minecraft:creeper")
                .unwrap();
            game.handle_packet(C::AddEntity(play::AddEntity {
                id: 31,
                uuid: 9,
                kind: creeper,
                position: [10.0, 64.0, 12.0],
                velocity: [0.0; 3],
                pitch: 0,
                yaw: 0,
                head_yaw: 0,
                data: 0,
            }))
            .unwrap();
            game.remote_set_camera(31);
            assert!(!game.controls_camera());
            assert!(game.hides_as_camera(31));
            assert!(game.spectated_post_effect.is_some());
            let attached = camera(game).unwrap();
            game.handle_packet(C::RemoveEntities(play::RemoveEntities { ids: vec![99] }))
                .unwrap();
            assert_eq!(camera(game), Some(attached));
            assert_eq!(attached.id(), 31);
            assert_eq!(attached.position(game), Some([10.0, 64.0, 12.0]));
            game.handle_packet(C::SetCamera(play::SetCamera { camera_id: 99 }))
                .unwrap();
            assert_eq!(
                camera(game),
                Some(attached),
                "unknown packet targets are ignored"
            );
            game.handle_packet(C::RemoveEntities(play::RemoveEntities { ids: vec![31] }))
                .unwrap();
            assert_eq!(camera(game), Some(local));
            assert!(
                game.controls_camera(),
                "movement input and position packets are enabled"
            );
            assert!(!game.hides_as_camera(31));
            assert!(game.spectated_post_effect.is_none());
            game.handle_packet(C::SetCamera(play::SetCamera { camera_id: 7 }))
                .unwrap();
            assert_eq!(camera(game), Some(local));

            game.title = Some(Title::new(game.save.root.clone(), &game.options.resources));
            assert!(
                camera(game).is_none(),
                "the title screen has no camera entity"
            );
            game.title = None;
            assert_eq!(camera(game), Some(local));

            spawn.dimension = "test:other_overworld".into();
            game.handle_packet(C::Respawn(play::Respawn { spawn, keep: 0 }))
                .unwrap();
            assert!(
                camera(game).is_none(),
                "replacement waits for its new player"
            );
            game.player = Some(PlayerMotion::new(game.motion, Vec3::ZERO).unwrap());
            let fresh = camera(game).unwrap();
            assert_eq!(fresh.id(), local.id());
            assert_ne!(fresh.epoch(), local.epoch());
            assert_eq!(local.position(game), None, "old level handles stay stale");
            assert_eq!(fresh.position(game), Some([0.0; 3]));
            game.client_logging_out();
            assert!(camera(game).is_none());
            assert_eq!(fresh.position(game), None);
        });
    }

    /// The local player's own state reads answer on a client from what the
    /// server sent it (`LocalPlayer`); another player reads None there.
    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_player_reads_answer_from_the_local_players_copy() {
        use super::super::{execution_role::ExecutionRole, native_entity_refs::tests::with_game};
        use horizon_client::{level_loading::Reason, motion::PlayerMotion};
        use horizon_core::geometry::Vec3;
        use horizon_protocol::{
            connection::Pipe,
            packets::play::{self, Clientbound as C, SpawnInfo},
        };
        use mod_api::{EntityRef, GameMode};

        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "PlayerReadTest").unwrap();
            let spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "minecraft:overworld".into(),
                seed: 0,
                game_type: 0,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            let remote = game.remote.as_mut().unwrap();
            remote.logged_in = true;
            remote.entity_id = 7;
            remote.profile = 42;
            game.player = Some(PlayerMotion::new(game.motion, Vec3::new(0.5, 65.0, 0.5)).unwrap());
            let player = game.local_player().unwrap();

            // Survival, as the login packet left it: building allowed, no flight.
            assert_eq!(player.game_mode(game), Some(GameMode::Survival));
            assert_eq!(player.may_build(game), Some(true));
            assert_eq!(player.abilities(game).map(|a| a.may_fly), Some(false));

            // CHANGE_GAME_MODE to adventure: no building; then the abilities packet.
            game.handle_packet(C::GameEvent(play::GameEvent {
                event: 3,
                param: 2.0,
            }))
            .unwrap();
            assert_eq!(player.game_mode(game), Some(GameMode::Adventure));
            assert_eq!(player.is_creative(game), Some(false));
            assert_eq!(player.is_spectator(game), Some(false));
            assert_eq!(player.may_build(game), Some(false));
            game.handle_packet(C::PlayerAbilities(play::PlayerAbilities {
                flags: 4 | 2,
                flying_speed: 0.1,
                walking_speed: 0.2,
            }))
            .unwrap();
            let abilities = player.abilities(game).unwrap();
            assert!(abilities.may_fly && abilities.flying && !abilities.invulnerable);
            assert!(!abilities.may_build);
            assert_eq!((abilities.fly_speed, abilities.walk_speed), (0.1, 0.2));

            // Food from SetHealth; the client keeps no exhaustion.
            game.handle_packet(C::SetHealth(play::SetHealth {
                health: 20.0,
                food: 5,
                saturation: 1.5,
            }))
            .unwrap();
            let food = player.food(game).unwrap();
            assert_eq!(
                (food.level, food.saturation, food.exhaustion),
                (5, 1.5, 0.0)
            );
            assert_eq!(player.needs_food(game), Some(true));
            assert_eq!(player.has_enough_food(game), Some(false));

            // The hotbar slot and the experience the server set.
            game.handle_packet(C::SetHeldSlot(play::SetHeldSlot { slot: 4 }))
                .unwrap();
            assert_eq!(player.selected_slot(game), Some(4));
            game.handle_packet(C::SetExperience(play::SetExperience {
                progress: 0.25,
                level: 3,
                total: 30,
            }))
            .unwrap();
            let xp = player.experience(game).unwrap();
            assert_eq!((xp.level, xp.progress, xp.total), (3, 0.25, 30));
            assert_eq!(player.xp_needed_for_next_level(game), Some(13));

            // The main inventory retains typed stacks and their damage, in slot order.
            let mut stack = horizon_client::items::ItemStack::new("minecraft:diamond_pickaxe", 1);
            stack.damage = 7;
            game.inventory.fill(None);
            game.inventory[3] = Some(stack.clone());
            game.inventory[8] = Some(horizon_client::items::ItemStack::new("minecraft:stone", 12));
            assert_eq!(
                player.inventory(game),
                Some(
                    vec![
                        stack,
                        horizon_client::items::ItemStack::new("minecraft:stone", 12)
                    ]
                    .try_into_api()
                    .unwrap()
                )
            );

            // Another player: the client isn't sent their state.
            let epoch = player.entity().epoch();
            let other = mod_api::__private::game::player_ref(EntityRef::new(epoch, 8), 43);
            assert_eq!(other.game_mode(game), None);
            assert_eq!(other.food(game), None);
            assert_eq!(other.selected_slot(game), None);
            assert_eq!(other.inventory(game), None);
        });
    }

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn client_epochs_follow_level_replacement_and_logout() {
        use super::super::{
            execution_role::ExecutionRole, native_entity_refs::tests::with_game,
            native_lifecycle::NativeLifecycle,
        };
        use horizon_client::{level_loading::Reason, mobs::Mob, motion::PlayerMotion};
        use horizon_core::geometry::Vec3;
        use horizon_protocol::{
            connection::Pipe,
            packets::play::{Clientbound, Respawn, SpawnInfo},
        };
        use mod_api::{CrosshairTarget, EntityRef};

        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            game.native_lifecycle = NativeLifecycle::default();
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "EpochTest").unwrap();
            // A transport alone has no ClientLevel yet.
            assert!(game.native_lifecycle.epoch().is_none());
            let mut spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "minecraft:overworld".into(),
                seed: 0,
                game_type: 1,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            let joined = game.native_lifecycle.epoch().unwrap();
            let old = EntityRef::new(joined, 12);
            let remote = game.remote.as_mut().unwrap();
            remote.menu.set(Some((7, "minecraft:generic_9x3".into())));
            let open_generation = remote.menu.generation;
            game.handle_packet(Clientbound::Respawn(Respawn {
                spawn: spawn.clone(),
                keep: 0,
            }))
            .unwrap();
            assert_eq!(game.native_lifecycle.epoch(), Some(joined));
            let remote = game.remote.as_mut().unwrap();
            assert!(remote.menu.as_ref().is_none());
            assert_ne!(remote.menu.generation, open_generation);
            let inventory_generation = remote.menu.generation;
            game.handle_packet(Clientbound::Respawn(Respawn {
                spawn: spawn.clone(),
                keep: 0,
            }))
            .unwrap();
            assert_eq!(game.native_lifecycle.epoch(), Some(joined));
            let remote = game.remote.as_mut().unwrap();
            assert!(remote.menu.as_ref().is_none());
            assert_ne!(remote.menu.generation, inventory_generation);
            remote.menu.set(Some((7, "minecraft:generic_9x3".into())));
            assert_ne!(remote.menu.generation, open_generation);
            assert_ne!(remote.menu.generation, inventory_generation);

            // A different level key replaces the world even with the same
            // dimension type; IDs can be reused but old handles cannot.
            spawn.dimension = "test:other_overworld".into();
            game.enter_remote_dimension(&spawn, true, Reason::Other);
            let replaced = game.native_lifecycle.epoch().unwrap();
            assert_ne!(replaced, joined);
            let current = EntityRef::new(replaced, 12);
            let kind = game.res.mob_types().get("minecraft:pig").unwrap().clone();
            game.mobs
                .push(Mob::new(12, kind, Vec3::new(0.0, 1.0, 2.0), 0.0));
            game.player = Some(PlayerMotion::new(game.motion, Vec3::ZERO).unwrap());
            game.yaw = 0.0;
            game.pitch = 0.0;
            let Some(CrosshairTarget::Entity { entity, .. }) = game.crosshair_target() else {
                panic!("the crosshair hits the pig in front of the player");
            };
            assert_eq!(entity, current);
            assert!(
                game.native_entity_target(current).is_none(),
                "still server-only"
            );
            // Isolate the existing epoch check from the logical-server guard.
            game.options.execution_role = ExecutionRole::IntegratedServer;
            assert!(game.native_entity_target(current).is_some());
            assert!(game.native_entity_target(old).is_none());
            game.options.execution_role = ExecutionRole::Client;

            game.client_logging_out();
            assert!(game.native_lifecycle.epoch().is_none());
            assert!(game.crosshair_target().is_none());
            game.client_logging_out();
            assert!(game.native_lifecycle.epoch().is_none());
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "EpochTest").unwrap();
            assert!(game.native_lifecycle.epoch().is_none());
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            assert_ne!(game.native_lifecycle.epoch(), Some(replaced));
            game.client_logging_out();
            assert!(game.native_lifecycle.epoch().is_none());
        });
    }

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn connection_and_dimension_replacement_clear_every_particle_group() {
        use super::super::{
            execution_role::ExecutionRole, native_entity_refs::tests::with_game,
            native_hooks::particle_count_tests::fill,
        };
        use horizon_client::level_loading::Reason;
        use horizon_protocol::{connection::Pipe, packets::play::SpawnInfo};
        with_game(|game| {
            game.options.execution_role = ExecutionRole::Client;
            fill(game);
            // Include pending particles as well as live groups before replacing the connection.
            game.particles.elder_guardian();
            let (client, _server) = Pipe::pair();
            game.connect_local(client, "ParticleReset").unwrap();
            assert_eq!(game.client_particle_count(), 0);
            assert!(game.pickups.live.is_empty());
            game.tick_particles();
            assert_eq!(game.client_particle_count(), 0);
            let spawn = SpawnInfo {
                dimension_type: 0,
                dimension: "minecraft:overworld".into(),
                seed: 0,
                game_type: 1,
                previous_game_type: None,
                debug: false,
                flat: true,
                last_death: None,
                portal_cooldown: 0,
                sea_level: 63,
            };
            fill(game);
            game.particles.elder_guardian();
            game.enter_remote_dimension(&spawn, false, Reason::Other);
            assert_eq!(game.client_particle_count(), 0);
            assert!(game.pickups.live.is_empty());
            game.tick_particles();
            assert_eq!(
                game.client_particle_count(),
                0,
                "pending groups cannot return in the new level"
            );
        });
    }

    #[test]
    #[ignore = "needs prepared resources in HORIZON_TEST_RESOURCES"]
    fn forgetting_and_closing_remote_columns_clear_their_data() {
        use super::super::native_entity_refs::tests::with_game;
        use horizon_client::{chunk::Chunk, containers::BlockEntity};

        with_game(|game| {
            for x in 0..2 {
                game.streaming
                    .insert_remote(&mut game.world, Chunk::empty(x, 0));
                game.world
                    .block_entities
                    .insert((x * 16, 64, 0), BlockEntity::Chest(vec![]));
            }
            // A same-position resend replaces the data without removing the column.
            game.streaming
                .replace_remote(&mut gameXÚYHØ[Y\˜JØ[YJK[Ü˜\

NÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”™[[Ý™Q[]Y\Ê^NŽ”™[[Ý™Q[]Y\ÈÈYÎˆ™XÈVÎNWHJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[Y\˜JØ[YJKÛÛYJ]XÚY
JNÂˆ\ÜÙ\Ù\HJ]XÚYšY

KÌJNÂˆ\ÜÙ\Ù\HJ]XÚYœÜÚ][ÛŠØ[YJKÛÛYJÌLŒŒL‹ŒJJNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù]Ø[Y\˜J^NŽ”Ù]Ø[Y\˜HÈØ[Y\˜WÚYˆNHJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJˆØ[Y\˜JØ[YJKˆÛÛYJ]XÚY
Kˆ[šÛ›ÝÛˆXÚÙ]\™Ù]È\™HYÛ›Ü™Y‚ˆ
NÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”™[[Ý™Q[]Y\Ê^NŽ”™[[Ý™Q[]Y\ÈÈYÎˆ™XÈVÌÌWHJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[Y\˜JØ[YJKÛÛYJØØ[
JNÂˆ\ÜÙ\JˆØ[YK˜ÛÛ›Û×ØØ[Y\˜J
Kˆ›[Ý™[Y[[œ][™ÜÚ][ÛˆXÚÙ]È\™H[˜X›Y‚ˆ
NÂˆ\ÜÙ\JYØ[YKšY\×Ø\×ØØ[Y\˜JÌJJNÂˆ\ÜÙ\JØ[YKœÜXÝ]YÜÜÝÙY™™XÝš\×Û›Û™J
JNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù]Ø[Y\˜J^NŽ”Ù]Ø[Y\˜HÈØ[Y\˜WÚYˆÈJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[Y\˜JØ[YJKÛÛYJØØ[
JNÂ‚ˆØ[YK]HHÛÛYJ]NŽ›™]ÊØ[YKœØ]™Kœ›ÛÝ˜ÛÛ™J
K	™Ø[YK›Ü[ÛœËœ™\ÛÝ\˜Ù\ÊJNÂˆ\ÜÙ\JˆØ[Y\˜JØ[YJKš\×Û›Û™J
KˆH]HØÜ™Y[ˆ\È›ÈØ[Y\˜H[]H‚ˆ
NÂˆØ[YK]HH›Û™NÂˆ\ÜÙ\Ù\HJØ[Y\˜JØ[YJKÛÛYJØØ[
JNÂ‚ˆÜ]Û‹™[Y[œÚ[ÛˆH\Ý›Ý\—ÛÝ™\ÛÜ›‹š[Ê
NÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”™\Ü]ÛŠ^NŽ”™\Ü]ÛˆÈÜ]Û‹ÙY\ˆJJBˆ[Ü˜\

NÂˆ\ÜÙ\JˆØ[Y\˜JØ[YJKš\×Û›Û™J
Kˆœ™\XÙ[Y[ØZ]È›Üˆ]È™]È^Y\ˆ‚ˆ
NÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ–‘T“ÊK[Ü˜\

JNÂˆ]œ™\ÚHØ[Y\˜JØ[YJK[Ü˜\

NÂˆ\ÜÙ\Ù\HJœ™\ÚšY

KØØ[šY

JNÂˆ\ÜÙ\Û™HJœ™\Ú™\ØÚ

KØØ[™\ØÚ

JNÂˆ\ÜÙ\Ù\HJØØ[œÜÚ][ÛŠØ[YJK›Û™K›Û]™[[™\ÈÝ^HÝ[HŠNÂˆ\ÜÙ\Ù\HJœ™\ÚœÜÚ][ÛŠØ[YJKÛÛYJÌŒÈ×JJNÂˆØ[YK˜ÛY[ÛÙÙÚ[™×ÛÝ]

NÂˆ\ÜÙ\JØ[Y\˜JØ[YJKš\×Û›Û™J
JNÂˆ\ÜÙ\Ù\HJœ™\ÚœÜÚ][ÛŠØ[YJK›Û™JNÂˆJNÂˆB‚ˆËËÈHØØ[^Y\‰ÜÈÝÛˆÝ]H™XYÈ[œÝÙ\ˆÛˆHÛY[œ›ÛHÚ]BˆËËÈÙ\™\ˆÙ[]
ØØ[^Y\˜
NÈ[›Ý\ˆ^Y\ˆ™XYÈ›Û™H\™K‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆÛY[Ü^Y\—Ü™XY×Ø[œÝÙ\—Ùœ›ÛWÝWÛØØ[Ü^Y\œ×ØÛÜJ
HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽžÙ^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛK˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[Y_NÂˆ\ÙHÜš^›Û—ØÛY[ŽžÛ]™[ÛØY[™ÎŽ”™X\ÛÛ‹[Ý[ÛŽŽ”^Y\“[Ý[ÛŸNÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽžÂˆÛÛ›™XÝ[ÛŽŽ”\KˆXÚÙ]ÎŽœ^NŽžÜÙ[‹ÛY[›Ý[™\ÈËÜ]Û’[™›ßKˆNÂˆ\ÙH[ÙØ\NŽžÑ[]T™Y‹Ø[YS[Ù_NÂ‚ˆÚ]ÙØ[YJØ[Y_ÂˆØ[YK›Ü[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆ]
ÛY[ÜÙ\™\ŠHH\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[”^Y\”™XY\ÝŠK[Ü˜\

NÂˆ]Ü]ÛˆHÜ]Û’[™›ÈÂˆ[Y[œÚ[Û—Ý\Nˆˆ[Y[œÚ[ÛŽˆ›Z[™XÜ˜Y›Ý™\ÛÜ›‹š[Ê
KˆÙYYˆˆØ[YWÝ\Nˆˆ™]š[Ý\×ÙØ[YWÝ\Nˆ›Û™KˆXYÎˆ˜[ÙKˆ›]ˆYKˆ\ÝÙX]ˆ›Û™KˆÜ[ØÛÛÛÝÛŽˆˆÙXWÛ]™[ˆŒËˆNÂˆØ[YK™[\—Ü™[[ÝWÙ[Y[œÚ[ÛŠ	œÜ]Û‹˜[ÙK™X\ÛÛŽŽ“Ý\ŠNÂˆ]™[[ÝHHØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

NÂˆ™[[ÝK›ÙÙÙYÚ[ˆHYNÂˆ™[[ÝK™[]WÚYHÎÂˆ™[[ÝKœ›Ùš[HHŽÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ›™]ÊKKŒJJK[Ü˜\

JNÂˆ]^Y\ˆHØ[YK›ØØ[Ü^Y\Š
K[Ü˜\

NÂ‚ˆËÈÝ\š]˜[\ÈHÙÚ[ˆXÚÙ]Y]ˆZ[[™È[ÝÙY›È›YÚ‚ˆ\ÜÙ\Ù\HJ^Y\‹™Ø[YWÛ[ÙJØ[YJKÛÛYJØ[YS[ÙNŽ”Ý\š]˜[
JNÂˆ\ÜÙ\Ù\HJ^Y\‹›X^WØZ[
Ø[YJKÛÛYJYJJNÂˆ\ÜÙ\Ù\HJ^Y\‹˜Xš[]Y\ÊØ[YJK›X\
_K›X^WÙ›JKÛÛYJ˜[ÙJJNÂ‚ˆËÈÒS‘ÑWÑÐSQWÓSÑHÈY™[\™Nˆ›ÈZ[[™ÎÈ[ˆHXš[]Y\ÈXÚÙ]‚ˆØ[YKš[™WÜXÚÙ]
ÎŽ‘Ø[YQ]™[
^NŽ‘Ø[YQ]™[Âˆ]™[ˆËˆ\˜[Nˆ‹ŒˆJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJ^Y\‹™Ø[YWÛ[ÙJØ[YJKÛÛYJØ[YS[ÙNŽY™[\™JJNÂˆ\ÜÙ\Ù\HJ^Y\‹š\×ØÜ™X]]™JØ[YJKÛÛYJ˜[ÙJJNÂˆ\ÜÙ\Ù\HJ^Y\‹š\×ÜÜXÝ]ÜŠØ[YJKÛÛYJ˜[ÙJJNÂˆ\ÜÙ\Ù\HJ^Y\‹›X^WØZ[
Ø[YJKÛÛYJ˜[ÙJJNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”^Y\Xš[]Y\Ê^NŽ”^Y\Xš[]Y\ÈÂˆ›YÜÎˆ‹ˆ›Z[™×ÜÜYYˆŒKˆØ[Ú[™×ÜÜYYˆŒ‹ˆJJBˆ[Ü˜\

NÂˆ]Xš[]Y\ÈH^Y\‹˜Xš[]Y\ÊØ[YJK[Ü˜\

NÂˆ\ÜÙ\JXš[]Y\Ë›X^WÙ›H	‰ˆXš[]Y\Ë™›Z[™È	‰ˆXXš[]Y\Ëš[[™\˜X›JNÂˆ\ÜÙ\JXXš[]Y\Ë›X^WØZ[
NÂˆ\ÜÙ\Ù\HJ
Xš[]Y\Ë™›WÜÜYYXš[]Y\ËØ[×ÜÜYY
K
ŒKŒŠJNÂ‚ˆËÈ›ÛÙœ›ÛHÙ]X[ÈHÛY[ÙY\È›È^]\Ý[Û‹‚ˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù]X[
^NŽ”Ù]X[ÂˆX[ˆŒŒˆ›ÛÙˆKˆØ]\˜][ÛŽˆKKˆJJBˆ[Ü˜\

NÂˆ]›ÛÙH^Y\‹™›ÛÙ
Ø[YJK[Ü˜\

NÂˆ\ÜÙ\Ù\HJˆ
›ÛÙ›]™[›ÛÙœØ]\˜][Û‹›ÛÙ™^]\Ý[ÛŠKˆ
KKKŒ
Bˆ
NÂˆ\ÜÙ\Ù\HJ^Y\‹›™YY×Ù›ÛÙ
Ø[YJKÛÛYJYJJNÂˆ\ÜÙ\Ù\HJ^Y\‹š\×Ù[›ÝYÚÙ›ÛÙ
Ø[YJKÛÛYJ˜[ÙJJNÂ‚ˆËÈHÝ˜\ˆÛÝ[™H^\šY[˜ÙHHÙ\™\ˆÙ]‚ˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù][ÛÝ
^NŽ”Ù][ÛÝÈÛÝˆJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJ^Y\‹œÙ[XÝYÜÛÝ
Ø[YJKÛÛYJ
JNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù]^\šY[˜ÙJ^NŽ”Ù]^\šY[˜ÙHÂˆ›ÙÜ™\ÜÎˆŒKˆ]™[ˆËˆÝ[ˆÌˆJJBˆ[Ü˜\

NÂˆ]H^Y\‹™^\šY[˜ÙJØ[YJK[Ü˜\

NÂˆ\ÜÙ\Ù\HJ
›]™[œ›ÙÜ™\ÜËÝ[
K
ËŒKÌ
JNÂˆ\ÜÙ\Ù\HJ^Y\‹žÛ™YYYÙ›Ü—Û™^Û]™[
Ø[YJKÛÛYJLÊJNÂ‚ˆËÈHXZ[ˆ[™[ÜžH™]Z[œÈ\YÝXÚÜÈ[™Z\ˆ[XYÙK[ˆÛÝÜ™\‹‚ˆ]]]ÝXÚÈHÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎŽ›™]Ê›Z[™XÜ˜Y™X[[Û™ÜXÚØ^H‹JNÂˆÝXÚË™[XYÙHHÎÂˆØ[YKš[™[ÜžK™š[
›Û™JNÂˆØ[YKš[™[ÜžVÌ×HHÛÛYJÝXÚË˜ÛÛ™J
JNÂˆØ[YKš[™[ÜžVÎHHÛÛYJÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎŽ›™]Ê›Z[™XÜ˜YœÝÛ™H‹LŠJNÂˆ\ÜÙ\Ù\HJˆ^Y\‹š[™[ÜžJØ[YJKˆÛÛYJˆ™XÈVÂˆÝXÚËˆÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎŽ›™]Ê›Z[™XÜ˜YœÝÛ™H‹LŠBˆBˆžWÚ[×Ø\J
Bˆ[Ü˜\

Bˆ
Bˆ
NÂ‚ˆËÈ[›Ý\ˆ^Y\ŽˆHÛY[\Û‰ÝÙ[Z\ˆÝ]K‚ˆ]\ØÚH^Y\‹™[]J
K™\ØÚ

NÂˆ]Ý\ˆH[ÙØ\NŽ—×Üš]˜]NŽ™Ø[YNŽœ^Y\—Ü™YŠ[]T™YŽŽ›™]Ê\ØÚ
KÊNÂˆ\ÜÙ\Ù\HJÝ\‹™Ø[YWÛ[ÙJØ[YJK›Û™JNÂˆ\ÜÙ\Ù\HJÝ\‹™›ÛÙ
Ø[YJK›Û™JNÂˆ\ÜÙ\Ù\HJÝ\‹œÙ[XÝYÜÛÝ
Ø[YJK›Û™JNÂˆ\ÜÙ\Ù\HJÝ\‹š[™[ÜžJØ[YJK›Û™JNÂˆJNÂˆB‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆÛY[Ù\ØÚ×Ù›ÛÝ×Û]™[Ü™\XÙ[Y[Ø[™ÛÙÛÝ]

HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽžÂˆ^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛK˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[YKˆ˜]]™WÛY™XÞXÛNŽ“˜]]™SY™XÞXÛKˆNÂˆ\ÙHÜš^›Û—ØÛY[ŽžÛ]™[ÛØY[™ÎŽ”™X\ÛÛ‹[ØœÎŽ“[Ø‹[Ý[ÛŽŽ”^Y\“[Ý[ÛŸNÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽžÂˆÛÛ›™XÝ[ÛŽŽ”\KˆXÚÙ]ÎŽœ^NŽžÐÛY[›Ý[™™\Ü]Û‹Ü]Û’[™›ßKˆNÂˆ\ÙH[ÙØ\NŽžÐÜ›ÜÜÚZ\•\™Ù][]T™YŸNÂ‚ˆÚ]ÙØ[YJØ[Y_ÂˆØ[YK›Ü[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[ÂˆØ[YK›˜]]™WÛY™XÞXÛHH˜]]™SY™XÞXÛNŽ™Y˜][

NÂˆ]
ÛY[ÜÙ\™\ŠHH\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[‘\ØÚ\ÝŠK[Ü˜\

NÂˆËÈH˜[œÜÜ[Û™H\È›ÈÛY[]™[Y]‚ˆ\ÜÙ\JØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

Kš\×Û›Û™J
JNÂˆ]]]Ü]ÛˆHÜ]Û’[™›ÈÂˆ[Y[œÚ[Û—Ý\Nˆˆ[Y[œÚ[ÛŽˆ›Z[™XÜ˜Y›Ý™\ÛÜ›‹š[Ê
KˆÙYYˆˆØ[YWÝ\NˆKˆ™]š[Ý\×ÙØ[YWÝ\Nˆ›Û™KˆXYÎˆ˜[ÙKˆ›]ˆYKˆ\ÝÙX]ˆ›Û™KˆÜ[ØÛÛÛÝÛŽˆˆÙXWÛ]™[ˆŒËˆNÂˆØ[YK™[\—Ü™[[ÝWÙ[Y[œÚ[ÛŠ	œÜ]Û‹˜[ÙK™X\ÛÛŽŽ“Ý\ŠNÂˆ]›Ú[™YHØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

K[Ü˜\

NÂˆ]ÛH[]T™YŽŽ›™]Ê›Ú[™YLŠNÂˆ]™[[ÝHHØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

NÂˆ™[[ÝK›Y[KœÙ]
ÛÛYJ
Ë›Z[™XÜ˜Y™Ù[™\šX×Î^È‹š[Ê
JJJNÂˆ]Ü[—ÙÙ[™\˜][ÛˆH™[[ÝK›Y[K™Ù[™\˜][ÛŽÂˆØ[YKš[™WÜXÚÙ]
ÛY[›Ý[™Ž”™\Ü]ÛŠ™\Ü]ÛˆÂˆÜ]ÛŽˆÜ]Û‹˜ÛÛ™J
KˆÙY\ˆˆJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

KÛÛYJ›Ú[™Y
JNÂˆ]™[[ÝHHØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

NÂˆ\ÜÙ\J™[[ÝK›Y[K˜\×Ü™YŠ
Kš\×Û›Û™J
JNÂˆ\ÜÙ\Û™HJ™[[ÝK›Y[K™Ù[™\˜][Û‹Ü[—ÙÙ[™\˜][ÛŠNÂˆ][™[ÜžWÙÙ[™\˜][ÛˆH™[[ÝK›Y[K™Ù[™\˜][ÛŽÂˆØ[YKš[™WÜXÚÙ]
ÛY[›Ý[™Ž”™\Ü]ÛŠ™\Ü]ÛˆÂˆÜ]ÛŽˆÜ]Û‹˜ÛÛ™J
KˆÙY\ˆˆJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

KÛÛYJ›Ú[™Y
JNÂˆ]™[[ÝHHØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

NÂˆ\ÜÙ\J™[[ÝK›Y[K˜\×Ü™YŠ
Kš\×Û›Û™J
JNÂˆ\ÜÙ\Û™HJ™[[ÝK›Y[K™Ù[™\˜][Û‹[™[ÜžWÙÙ[™\˜][ÛŠNÂˆ™[[ÝK›Y[KœÙ]
ÛÛYJ
Ë›Z[™XÜ˜Y™Ù[™\šX×Î^È‹š[Ê
JJJNÂˆ\ÜÙ\Û™HJ™[[ÝK›Y[K™Ù[™\˜][Û‹Ü[—ÙÙ[™\˜][ÛŠNÂˆ\ÜÙ\Û™HJ™[[ÝK›Y[K™Ù[™\˜][Û‹[™[ÜžWÙÙ[™\˜][ÛŠNÂ‚ˆËÈHY™™\™[]™[Ù^H™\XÙ\ÈHÛÜ›]™[ˆÚ]HØ[YBˆËÈ[Y[œÚ[Ûˆ\NÈQÈØ[ˆ™H™]\ÙY]Û[™\ÈØ[››Ý‚ˆÜ]Û‹™[Y[œÚ[ÛˆH\Ý›Ý\—ÛÝ™\ÛÜ›‹š[Ê
NÂˆØ[YK™[\—Ü™[[ÝWÙ[Y[œÚ[ÛŠ	œÜ]Û‹YK™X\ÛÛŽŽ“Ý\ŠNÂˆ]™\XÙYHØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

K[Ü˜\

NÂˆ\ÜÙ\Û™HJ™\XÙY›Ú[™Y
NÂˆ]Ý\œ™[H[]T™YŽŽ›™]Ê™\XÙYLŠNÂˆ]Ú[™HØ[YKœ™\Ë›[Ø—Ý\\Ê
K™Ù]
›Z[™XÜ˜YœYÈŠK[Ü˜\

K˜ÛÛ™J
NÂˆØ[YK›[ØœÂˆœ\Ú
[ØŽŽ›™]ÊL‹Ú[™™XÌÎŽ›™]ÊŒKŒ‹Œ
KŒ
JNÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ–‘T“ÊK[Ü˜\

JNÂˆØ[YKžX]ÈHŒÂˆØ[YKœ]ÚHŒÂˆ]ÛÛYJÜ›ÜÜÚZ\•\™Ù]Ž‘[]HÈ[]K‹ˆJHHØ[YK˜Ü›ÜÜÚZ\—Ý\™Ù]

H[ÙHÂˆ[šXÈJHÜ›ÜÜÚZ\ˆ]ÈHYÈ[ˆœ›ÛÙˆH^Y\ˆŠNÂˆNÂˆ\ÜÙ\Ù\HJ[]KÝ\œ™[
NÂˆ\ÜÙ\JˆØ[YK›˜]]™WÙ[]WÝ\™Ù]
Ý\œ™[
Kš\×Û›Û™J
KˆœÝ[Ù\™\‹[Û›H‚ˆ
NÂˆËÈ\ÛÛ]HH^\Ý[™È\ØÚÚXÚÈœ›ÛHHÙÚXØ[\Ù\™\ˆÝX\™‚ˆØ[YK›Ü[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\ŽÂˆ\ÜÙ\JØ[YK›˜]]™WÙ[]WÝ\™Ù]
Ý\œ™[
Kš\×ÜÛÛYJ
JNÂˆ\ÜÙ\JØ[YK›˜]]™WÙ[]WÝ\™Ù]
Û
Kš\×Û›Û™J
JNÂˆØ[YK›Ü[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Â‚ˆØ[YK˜ÛY[ÛÙÙÚ[™×ÛÝ]

NÂˆ\ÜÙ\JØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

Kš\×Û›Û™J
JNÂˆ\ÜÙ\JØ[YK˜Ü›ÜÜÚZ\—Ý\™Ù]

Kš\×Û›Û™J
JNÂˆØ[YK˜ÛY[ÛÙÙÚ[™×ÛÝ]

NÂˆ\ÜÙ\JØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

Kš\×Û›Û™J
JNÂˆ]
ÛY[ÜÙ\™\ŠHH\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[‘\ØÚ\ÝŠK[Ü˜\

NÂˆ\ÜÙ\JØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

Kš\×Û›Û™J
JNÂˆØ[YK™[\—Ü™[[ÝWÙ[Y[œÚ[ÛŠ	œÜ]Û‹˜[ÙK™X\ÛÛŽŽ“Ý\ŠNÂˆ\ÜÙ\Û™HJØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

KÛÛYJ™\XÙY
JNÂˆØ[YK˜ÛY[ÛÙÙÚ[™×ÛÝ]

NÂˆ\ÜÙ\JØ[YK›˜]]™WÛY™XÞXÛK™\ØÚ

Kš\×Û›Û™J
JNÂˆJNÂˆB‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆÛÛ›™XÝ[Û—Ø[™Ù[Y[œÚ[Û—Ü™\XÙ[Y[ØÛX\—Ù]™\žWÜ\XÛWÙÜ›Ý\

HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽžÂˆ^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛK˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[YKˆ˜]]™WÚÛÚÜÎŽœ\XÛWØÛÝ[Ý\ÝÎŽ™š[ˆNÂˆ\ÙHÜš^›Û—ØÛY[Ž›]™[ÛØY[™ÎŽ”™X\ÛÛŽÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽžØÛÛ›™XÝ[ÛŽŽ”\KXÚÙ]ÎŽœ^NŽ”Ü]Û’[™›ßNÂˆÚ]ÙØ[YJØ[Y_ÂˆØ[YK›Ü[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆš[
Ø[YJNÂˆËÈ[˜ÛYH[™[™È\XÛ\È\ÈÙ[\È]™HÜ›Ý\È™Y›Ü™H™\XÚ[™ÈHÛÛ›™XÝ[Û‹‚ˆØ[YKœ\XÛ\Ë™[\—ÙÝX\™X[Š
NÂˆ]
ÛY[ÜÙ\™\ŠHH\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[”\XÛT™\Ù]ŠK[Ü˜\

NÂˆ\ÜÙ\Ù\HJØ[YK˜ÛY[Ü\XÛWØÛÝ[

K
NÂˆ\ÜÙ\JØ[YKœXÚÝ\Ë›]™Kš\×Ù[\J
JNÂˆØ[YKXÚ×Ü\XÛ\Ê
NÂˆ\ÜÙ\Ù\HJØ[YK˜ÛY[Ü\XÛWØÛÝ[

K
NÂˆ]Ü]ÛˆHÜ]Û’[™›ÈÂˆ[Y[œÚ[Û—Ý\Nˆˆ[Y[œÚ[ÛŽˆ›Z[™XÜ˜Y›Ý™\ÛÜ›‹š[Ê
KˆÙYYˆˆØ[YWÝ\NˆKˆ™]š[Ý\×ÙØ[YWÝ\Nˆ›Û™KˆXYÎˆ˜[ÙKˆ›]ˆYKˆ\ÝÙX]ˆ›Û™KˆÜ[ØÛÛÛÝÛŽˆˆÙXWÛ]™[ˆŒËˆNÂˆš[
Ø[YJNÂˆØ[YKœ\XÛ\Ë™[\—ÙÝX\™X[Š
NÂˆØ[YK™[\—Ü™[[ÝWÙ[Y[œÚ[ÛŠ	œÜ]Û‹˜[ÙK™X\ÛÛŽŽ“Ý\ŠNÂˆ\ÜÙ\Ù\HJØ[YK˜ÛY[Ü\XÛWØÛÝ[

K
NÂˆ\ÜÙ\JØ[YKœXÚÝ\Ë›]™Kš\×Ù[\J
JNÂˆØ[YKXÚ×Ü\XÛ\Ê
NÂˆ\ÜÙ\Ù\HJˆØ[YK˜ÛY[Ü\XÛWØÛÝ[

Kˆˆœ[™[™ÈÜ›Ý\ÈØ[››Ý™]\›ˆ[ˆH™]È]™[‚ˆ
NÂˆJNÂˆB‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆ›Ü™Ù][™×Ø[™ØÛÜÚ[™×Ü™[[ÝWØÛÛ[[œ×ØÛX\—ÝZ\—Ù]J
HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽ›˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[YNÂˆ\ÙHÜš^›Û—ØÛY[ŽžØÚ[šÎŽÚ[šËÛÛZ[™\œÎŽ›ØÚÑ[]_NÂ‚ˆÚ]ÙØ[YJØ[Y_Âˆ›Üˆ[ˆ‹ŒˆÂˆØ[YKœÝ™X[Z[™Âˆš[œÙ\Ü™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆØ[YKÛÜ›ˆ˜›ØÚ×Ù[]Y\Âˆš[œÙ\


ˆM‹
K›ØÚÑ[]NŽÚ\Ý
™XÈV×JJNÂˆBˆËÈHØ[YK\ÜÚ][Ûˆ™\Ù[™™\XÙ\ÈH]HÚ]Ý]™[[Ýš[™ÈHÛÛ[[‹‚ˆØ[YKœÝ™X[Z[™Âˆœ™\XÙWÜ™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆ\ÜÙ\Ù\HJØ[YKÛÜ›˜Ú[šÜË›[Š
KŠNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	Š
JJNÂˆ\ÜÙ\JØ[YKœÝ™X[Z[™Ëš\×Û]

JNÂˆ]]]›Ü™ÛÝ[ˆHÊ
WKš[×Ú]\Š
K˜ÛÛXÝ

NÂˆØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]›Ü™ÛÝ[‹YJNÂˆ\ÜÙ\JYØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	Š
JJNÂˆ\ÜÙ\JYØ[YKÛÜ›˜›ØÚ×Ù[]Y\Ë˜ÛÛZ[œ×ÚÙ^J	Š
JJNÂˆ\ÜÙ\JYØ[YKœÝ™X[Z[™Ëš\×Û]

JNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠK
JJNÂˆØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]Ê
WKš[×Ú]\Š
K˜ÛÛXÝ

KYJNÂˆ›Üˆ[ˆÍËL×HÂˆØ[YKœÝ™X[Z[™Âˆš[œÙ\Ü™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆØ[YKÛÜ›ˆ˜›ØÚ×Ù[]Y\Âˆš[œÙ\


ˆM‹
K›ØÚÑ[]NŽÚ\Ý
™XÈV×JJNÂˆBˆËÈØXÚKXÙ[™HXÚÙ]ˆ[Ý™HÈ
MK
K]˜Y]\ÈL‚ˆØ[YK™›ÜÜ™[[ÝWØÚ[šÜ×ÛÝ]ÛÙ—Ü˜[™ÙJ
MK
KL
NÂˆ\ÜÙ\JYØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠK
JJNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠË
JJNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠLË
JJNÂˆËÈØXÚK\˜Y]\ÈXÚÙ]ˆÚš[šÚ[™ÈÈˆÙY\È˜Y]\ÈH\›Ý[™
MK
K‚ˆØ[YK™›ÜÜ™[[ÝWØÚ[šÜ×ÛÝ]ÛÙ—Ü˜[™ÙJ
MK
KŠNÂˆ\ÜÙ\JYØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠË
JJNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	ŠLË
JJNÂˆ\ÜÙ\JYØ[YKÛÜ›˜›ØÚ×Ù[]Y\Ë˜ÛÛZ[œ×ÚÙ^J	ŠÈ
ˆM‹
JJNÂˆËÈH]\ˆ›Ü™Ù]›Üˆ[ˆ]šXÝYÛÛ[[ˆ\È\›[\ÜË‚ˆØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]ÊË
WKš[×Ú]\Š
K˜ÛÛXÝ

KYJNÂˆØ[YK™›Ü™Ù]Ø[Ü™[[ÝWØÛÛ[[œÊ
NÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜËš\×Ù[\J
JNÂˆ\ÜÙ\JØ[YKÛÜ›˜›ØÚ×Ù[]Y\Ëš\×Ù[\J
JNÂˆJNÂˆB‚ˆËËÈHÛY[Ú[šÈÙX[H
Ø[YKÛ™]ÛÜšËÜ™[[ÝWØÚ[šÜËœœÊHXÜ›ÜÜÈ]™\žHœ™YBˆËËÈ›Ý]K\ÈHœ™YYÙ]XXÚ›Ý]H™]\›œÎˆH›Ü™Ù]XÚÙ]BˆËËÈXYÙ\ˆÙ[™K[[Ý™H›Ü
HØXÚKXÙ[™HXÚÙ]
KH˜Y]\Ë\Úš[šÂˆËËÈ›Ü
HØXÚK\˜Y]\ÈXÚÙ]
H[™HÛY[]™[ÛÚ[™ËˆÛ™Hœ™YYˆËËÈ[žH\ˆÛÛ[[‹[ˆHÜ™\ˆH›Ý]\Èœ™YH[K[™H™\X]ˆËËÈ›Ü™Ù]›Üˆ[ˆ[™XYKYœ™YYÛÛ[[ˆœ™Y\È›Ý[™ÎˆYÜ[™ÈHÛÛ[[‚ˆËËÈ\ÈHÚ[šÈXÚÙ]	ÜÈÚ[™È
™[[ÝWØÚ[šØ
K›ÝHÝ™X[Z[™ÂˆËËÈ]™[	ÜË‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆ]™\žWÙœ™YWÜ›Ý]WÜ™]\›œ×ÝWØÛÛ[[œ×Ú]Ùœ™YYÛÛ˜ÙJ
HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽ›˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[YNÂˆ\ÙHÜš^›Û—ØÛY[Ž˜Ú[šÎŽÚ[šÎÂ‚ˆÚ]ÙØ[YJØ[Y_Âˆ›Üˆ[ˆÌWHÂˆØ[YKœÝ™X[Z[™Âˆš[œÙ\Ü™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆBˆËÈH›Ü™Ù]XÚÙ]œ™Y\È]ÈÛÛ[[ˆÛ˜ÙNÈHØ[YH›Ü™Ù]YØZ[‹ˆËÈ›ÜˆHÛÛ[[ˆ[™XYHœ™YYœ™Y\È›Ý[™Ë‚ˆ]]]ÛÛ™HHÊ
WKš[×Ú]\Š
K˜ÛÛXÝ

NÂˆ\ÜÙ\Ù\HJØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]ÛÛ™KYJKÊ
WJNÂˆ]]]ÛÛ™HHÊ
WKš[×Ú]\Š
K˜ÛÛXÝ

NÂˆ\ÜÙ\JØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]ÛÛ™KYJKš\×Ù[\J
JNÂˆ›Üˆ[ˆÍËL×HÂˆØ[YKœÝ™X[Z[™Âˆš[œÙ\Ü™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆBˆËÈHÙ[™H[Ý™HÈ
MK
H]˜Y]\ÈLœ™Y\È
K
HÛ›K‚ˆ\ÜÙ\Ù\HJØ[YK™›ÜÜ™[[ÝWØÚ[šÜ×ÛÝ]ÛÙ—Ü˜[™ÙJ
MK
KL
KÊK
WJNÂˆËÈÚš[šÚ[™ÈH˜Y]\ÈÈˆœ™Y\È
Ë
HÛ›K‚ˆ\ÜÙ\Ù\HJØ[YK™›ÜÜ™[[ÝWØÚ[šÜ×ÛÝ]ÛÙ—Ü˜[™ÙJ
MK
KŠKÊË
WJNÂˆËÈHÛY[]™[ÛÚ[™Èœ™Y\ÈÚ]\ÈY‚ˆ\ÜÙ\Ù\HJØ[YK™›Ü™Ù]Ø[Ü™[[ÝWØÛÛ[[œÊ
KÊLË
WJNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜËš\×Ù[\J
JNÂˆ\ÜÙ\JØ[YKÛÜ›˜›ØÚ×Ù[]Y\Ëš\×Ù[\J
JNÂˆJNÂˆB‚ˆËËÈHÙX[HÚ]È]HXÚÙ]]™[ˆÝ™X[Z[™ÈYÜ[™Ë™\XÚ[™ÈÜ‚ˆËËÈ™K\Ù[™[™ÈHÛÛ[[ˆÙY\È]™\ÚY[8 %H™K\Ù[ÜÚ][ÛˆØYÂˆËËÈYØZ[ˆ
™[[ÝWØÚ[šØ
K™]™\ˆ[ˆ[›ØY[™HØY8 %ÛÈH›Ü™Ù]ˆËËÈ™]\›œÈ^XÝH]‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆÝ™X[Z[™×ØYÜ[Û—Ø[™Ü™\Ù[™ÜÝ^WÜ™\ÚY[Ý[[ÝWÙ›Ü™Ù]

HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽ›˜]]™WÙ[]WÜ™YœÎŽ\ÝÎŽÚ]ÙØ[YNÂˆ\ÙHÜš^›Û—ØÛY[Ž˜Ú[šÎŽÚ[šÎÂ‚ˆÚ]ÙØ[YJØ[Y_ÂˆØ[YKœÝ™X[Z[™Âˆš[œÙ\Ü™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆËÈH™K\Ù[™ÙˆH™\ÚY[]ÛÛ[[ˆZÙ\È™\XÙWÜ™[[ÝK‚ˆØ[YKœÝ™X[Z[™Âˆœ™\XÙWÜ™[[ÝJ	›]]Ø[YKÛÜ›Ú[šÎŽ™[\J
JNÂˆ\ÜÙ\JØ[YKÛÜ›˜Ú[šÜË˜ÛÛZ[œ×ÚÙ^J	Š
JJNÂˆ]]]ÛÛ™HHÊ
WKš[×Ú]\Š
K˜ÛÛXÝ

NÂˆ\ÜÙ\Ù\HJØ[YK™›Ü™Ù]Ü™[[ÝWØÛÛ[[œÊ	›]]ÛÛ™KYJKÊ
WJNÂˆJNÂˆB‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆ›Ú[š[™×ØWÜÙ\™\—Ù[\Y\×ÝWÛÙ™—Ú[™ÝÛÊ
HÂˆËÈÛY[XÚÙ]\Ý[™\‹š[™SÙÚ[ˆXZÙ\ÈH™]ÈØØ[^Y\‹ÚÜÙHÚÛH[™[ÜžH\ÂˆËÈ[\NˆHÙ\™\ˆ]™]™\ˆÙ[™ÈÛÝH]\Ý›ÝX]™HH\ÝÙ\™\‰ÜÈ][K‚ˆ\ÙHÜ˜]NŽ™Ø[YNŽžÂˆ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜË^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛKØYÜ™\ÛÝ\˜Ù\ËˆNÂˆ\ÙHÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎÂˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK[Ü˜\

JNÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\‹ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆÜ[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆ]
š\œÝÙš\œÝÜÙ\™\ŠHHÜš^›Û—Ü›ÝØÛÛŽ˜ÛÛ›™XÝ[ÛŽŽ”\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
š\œÝ’Ü\ÝŠK[Ü˜\

NÂˆ]ÝXÚÈHÛÛYJ][TÝXÚÎŽ›™]Ê›Z[™XÜ˜YœÚY[‹JJNÂˆØ[YKš[™[ÜžVÌHHÝXÚÊ
NÂˆØ[YK˜\›[Ü–ÌHHÝXÚÊ
NÂˆØ[YK›Ù™š[™HÝXÚÊ
NÂˆ]
ÙXÛÛ™ÜÙXÛÛ™ÜÙ\™\ŠHHÜš^›Û—Ü›ÝØÛÛŽ˜ÛÛ›™XÝ[ÛŽŽ”\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÙXÛÛ™’Ü\ÝŠK[Ü˜\

NÂˆ\ÜÙ\JØ[YKš[™[ÜžVÌKš\×Û›Û™J
H	‰ˆØ[YK˜\›[Ü–ÌKš\×Û›Û™J
JNÂˆ\ÜÙ\JØ[YK›Ù™š[™š\×Û›Û™J
JNÂˆB‚ˆÖÝ\ÝBˆ›ˆWØÚ[š×ÜÝÜ˜YÙWÝZÙ\×Ø[žWÜ˜Y]\×Ø[™ØÙ[™WØWÜÙ\™\—ÜÙ[™Ê
HÂˆ\ÙHÝ\\ŽŽš[—ÜÝÜ˜YÙWÜ˜[™ÙNÂˆËÈ˜[š[IÜÈÝÜ˜YÙNˆ˜Y]\È]X\Ý‹\ÈËXXÚØ^Hœ›ÛHHÙ[™K‚ˆ\ÜÙ\J[—ÜÝÜ˜YÙWÜ˜[™ÙJL

KLËLLÊJNÂˆ\ÜÙ\JZ[—ÜÝÜ˜YÙWÜ˜[™ÙJL

KM
JNÂˆ\ÜÙ\J[—ÜÝÜ˜YÙWÜ˜[™ÙJMK
LL
KLKMJJNÂˆ\ÜÙ\JZ[—ÜÝÜ˜YÙWÜ˜[™ÙJ
LL
KL‹L
JNÂˆËÈÝ]ÙˆLÌ‰ÜÈ˜[™ÙK\ÈHÙ\™\ˆX^HÙ[™ˆ›ÈÝ™\™›ÝË[™HÝ[\ÈHÜ˜\ÛÝÜ›Û™Ë‚ˆ\ÜÙ\J[—ÜÝÜ˜YÙWÜ˜[™ÙJLÌŽŽ“PV

KLÌŽŽ“PVLÌŽŽ“RSˆ
ÈJJNÂˆ\ÜÙ\JZ[—ÜÝÜ˜YÙWÜ˜[™ÙJL
LÌŽŽ“RS‹
KLÌŽŽ“PV
JNÂˆ\ÜÙ\J[—ÜÝÜ˜YÙWÜ˜[™ÙJˆLˆ
LÌŽŽ“RS‹LÌŽŽ“PV
KˆLÌŽŽ“RSˆ
ÈLËˆLÌŽŽ“PVHLÂˆ
JNÂˆB‚ˆ\ÙHÝ\\ŽŽžÂˆ›Ø]Ú[\˜XÝÜ\ÜÙ\ËÚ]ØY\—Ù]™[ÛY[Ú\×Û]™[ÛÜÙ\™\—Ùš\™Kˆ][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\ËÙY\ØÛY[ÜÝ]KÜ[—Ú\ÚÛÜ™\‹™[[ÝWÛØš™XÝÜÝÚ[™ÜËˆ™[[ÝWÝ\ÙWÜÝÚ[™ÜËÚ\WÛYY]ËÛ˜\Ý\™Ù]Þ[˜ÙYÛØØ[Ý\ÙKˆNÂ‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ[™ØÜ˜]ÚÛY\ËØÝÙ—Bˆ›ˆ˜XÚÚ[™×Ù[Z]\œ×Ù›ÛÝ×ÝWÜXÚÙ]Ý\™Ù]

HÂˆ\ÙHÜ˜]NŽ™Ø[YNŽžÂˆ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜË^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛKˆØYÜ™\ÛÝ\˜Ù\ËÝ[NŽ•ÕSWÑU‘S•ˆNÂˆ\ÙHÜš^›Û—ØÛY[ŽžÛ[Ý[ÛŽŽ”^Y\“[Ý[Û‹\XÛ\ÎŽ”\XÛU\KÜÙNŽ”ÜÙ_NÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽžÂˆ[]WÙ]NŽžÕ˜[YKÙ\šX[^™\ŸKˆXÚÙ]ÎŽœ^NŽžÜÙ[‹ÛY[›Ý[™\ÈßKˆNÂ‚ˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK[Ü˜\

JNÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\‹ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆÜ[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆ]
ÛY[ÜÙ\™\ŠHHÜš^›Û—Ü›ÝØÛÛŽ˜ÛÛ›™XÝ[ÛŽŽ”\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[‘[Z]\•\ÝŠK[Ü˜\

NÂˆØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

K™[]WÚYHÎÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ›™]ÊŒŒŒ
JK[Ü˜\

JNÂˆØ[YKœ\XÛ\ËœÙYYÜ˜[™Û\Ê‹Ë
NÂ‚ˆ]ÚXÚÈHØ[YNˆ	‘Ø[YO	×Ï‹K™Y]ˆÙÈ×KZYÚˆŒÌŸÂˆ]]]ÙY[ˆH˜[ÙNÂˆ›Üˆ
Ü[ÛœË]™[ØÚ]JH[ˆØ[YKœ\XÛ\Ëœ™\]Y\ÝY˜\×Ü™YŠ
K[Ü˜\

HÂˆYˆÜ[ÛœËHOHHÂˆÛÛ[YNÂˆBˆÙY[ˆHYNÂˆ]HH™Y]ÌWH
ÈŽ™œ›ÛJZYÚ
H
ˆ
H
È
™[ØÚ]VÌWHHŒŠHÈŒ
NÂˆ\ÜÙ\J
]ÌHH™Y]ÌJK˜XœÊ
HHŽ™œ›ÛJ—ÙŒÌŠHÈŒ
NÂˆ\ÜÙ\Jˆ
]ÌWHHJK˜XœÊ
HYKLLˆžÝNßH]Ø]ßK^XÝYHÞ_H‚ˆ
NÂˆ\ÜÙ\J
]Ì—HH™Y]Ì—JK˜XœÊ
HHŽ™œ›ÛJ—ÙŒÌŠHÈŒ
NÂˆBˆ\ÜÙ\JÙY[‹››ÈÝNßH[Z\ÜÚ[ÛœÈŠNÂˆNÂˆ]\\ÈHÂˆ\XÛU\NŽÜš]ˆ\XÛU\NŽ‘[˜Ú[Y]ˆ\XÛU\NŽ•Ý[SÙ•[™Z[™ËˆNÂˆ]\™Ù]HÂˆ]Ý\HÍKŒŒËŒNÂˆØ[YKœÜ]Û—Ù[]J\™Ù]›Z[™XÜ˜Yœ^Y\ˆ‹Ý\ÌŒÈ×JNÂˆØ[YKœ\XÛ\Ëœ™\]Y\ÝYHÛÛYJ™XÎŽ›™]Ê
JNÂˆ›ÜˆXÝ[Ûˆ[ˆÌK—HÂˆØ[YKš[™WÜXÚÙ]
ÎŽ[š[X]J^NŽ[š[X]HÈYˆ\™Ù]XÝ[ÛˆJJBˆ[Ü˜\

NÂˆBˆØ[YKš[™WÜXÚÙ]
ÎŽ‘[]Q]™[
^NŽ‘[]Q]™[ÂˆYˆ\™Ù]ˆ]™[ˆÕSWÑU‘S•ˆJJBˆ[Ü˜\

NÂˆ›ÜˆH[ˆ\\ÈÂˆÚXÚÊ	™Ø[YKKÝ\KŽ
NÂˆB‚ˆ][Ý™YHÌMKŒKŒL‹ŒNÂˆØ[YKœ\XÛ\Ëœ™\]Y\ÝYHÛÛYJ™XÎŽ›™]Ê
JNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”Ù][]Q]J^NŽ”Ù][]Q]HÂˆYˆ\™Ù]ˆ˜[Y\Îˆ™XÈVÊ‹Ù\šX[^™\ŽŽ”ÔÑK˜[YNŽ’Y
ÜÙNŽÜ›ÝXÚ[™ËšY

JJWKˆJJBˆ[Ü˜\

NÂˆØ[YKš[™WÜXÚÙ]
ÎŽ‘[]TÜÚ][Û”Þ[˜Ê^NŽ‘[]TÜÚ][Û”Þ[˜ÈÂˆYˆ\™Ù]ˆ]ˆ^NŽ”ÜÚ][Û”]Ž“[™X\Š[Ý™Y
KˆX]ÎˆŒˆ]ÚˆŒˆÛ—ÙÜ›Ý[™ˆYKˆJJBˆ[Ü˜\

NÂˆ›ÜˆÈ[ˆ‹ÂˆØ[YKXÚ×Ü™[[ÝWÙ[]Y\ÊYJNÂˆBˆØ[YKXÚ×Ü\XÛ\Ê
NÂˆ›ÜˆH[ˆ\\ÈÂˆÚXÚÊ	™Ø[YKK[Ý™YKJNÂˆB‚ˆËÈHÛ[]H[Ý™YYØZ[ˆ™Y›Ü™H[œZ\š[™Ëˆ™K\Z\š[™È]ÈY™Y›Ü™BˆËÈH™^\XÛHXÚÈÙY\È[Z][™È]HÛ[]IÜÈš[˜[›ÙK‚ˆØ[YKœ\XÛ\Ëœ™\]Y\ÝYHÛÛYJ™XÎŽ›™]Ê
JNÂˆ]\ÝHÌNŒKŒL‹ŒNÂˆØ[YK›[Ø—Ú[™^ˆ™Ù]Û]]
	›]]Ø[YK›[ØœË\™Ù]\ÈLÌŠBˆ[Ü˜\

BˆœÜÚ][ÛˆH™XÌÎŽ›™]Ê\ÝÌK\ÝÌWK\ÝÌ—JNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ”™[[Ý™Q[]Y\Ê^NŽ”™[[Ý™Q[]Y\ÈÂˆYÎˆ™XÈVÝ\™Ù]KˆJJBˆ[Ü˜\

NÂˆØ[YKœÜ]Û—Ù[]J\™Ù]›Z[™XÜ˜Yœ^Y\ˆ‹ÌŽŒËŒKŒKÌŒÈ×JNÂˆØ[YKXÚ×Ü\XÛ\Ê
NÂˆ›ÜˆH[ˆ\\ÈÂˆÚXÚÊ	™Ø[YKK\ÝKJNÂˆB‚ˆØ[YKœ\XÛ\Ë˜ÛX\Š
NÂˆØ[YKœ\XÛ\Ëœ™\]Y\ÝYHÛÛYJ™XÎŽ›™]Ê
JNÂˆØ[YKš[™WÜXÚÙ]
ÎŽ‘[]Q]™[
^NŽ‘[]Q]™[ÂˆYˆËˆ]™[ˆÕSWÑU‘S•ˆJJBˆ[Ü˜\

NÂˆÚXÚÊ	™Ø[YK\XÛU\NŽ•Ý[SÙ•[™Z[™ËÌŒŒŒKKŽ
NÂˆØ[YKœ\XÛ\Ëœ™\]Y\ÝYHÛÛYJ™XÎŽ›™]Ê
JNÂˆ]^Y\ˆHØ[YKœ^Y\‹˜\×Û]]

K[Ü˜\

NÂˆ^Y\‹œÜÚ][ÛˆH™XÌÎŽ›™]ÊMŒŽŒŒ
NÂˆ^Y\‹œÙ]ÜÜÙJÜÙNŽ”ÝÚ[[Z[™ÊK[Ü˜\

NÂˆØ[YKXÚ×Ü\XÛ\Ê
NÂˆÚXÚÊ	™Ø[YK\XÛU\NŽ•Ý[SÙ•[™Z[™ËËMŒŽŒŒKŠNÂˆB‚ˆËËÈH˜[š[H‹ŒÈÛY[	ÜÈÙ\™\˜›Ý[™ÛÛZ[™\ÛXÚÔXÚÙ]›ÜˆHÚYXÛXÚÈœ›ÛHÝ˜\ˆÛÝÍÂˆËËÈ[ÈÛÝL\ÝYÍÈš\œÝ
Ú\™HÝ™X[H\›™\ÜË[™[ÜžX
KˆHÙ^H[Ø^\ÈXYË‚ˆÖÝ\ÝBˆ›ˆÚ[™ÙYÜÛÝ×ÙÛ×ÛÝ]Ú[—ÝWÛÜ™\—Ý˜[š[WÜ×Ú\ÚÛX\Ú]\˜]\Ê
HÂˆ\ÜÙ\Ù\HJÜ[—Ú\ÚÛÜ™\Š	–ÌLÍ×JKÌÍËLJNÂˆ\ÜÙ\Ù\HJÜ[—Ú\ÚÛÜ™\Š	–ÌK‹Í‹JVÌK
NÂˆËÈ\ÝÙ^\ÈHX›HÜ›ÝÜÎÈ]™\žHÙ^H\ÈÝ[\ÝYÛ˜ÙK‚ˆ][ˆ™XÏLMˆH
‹ŠK˜ÛÛXÝ

NÂˆ]]]Ý]HÜ[—Ú\ÚÛÜ™\Š	˜[
NÂˆÝ]œÛÜÝ[œÝX›J
NÂˆ\ÜÙ\Ù\HJÝ][
NÂˆB‚ˆÖÝ\ÝBˆ›ˆ\YØÚ]Ú\×ÜÙ[Ø\×ÝWØÛY[ØÚ]Ù]™[ÛX]™\×Ú]

HÂˆ\ÜÙ\Ù\HJÚ]ØY\—Ù]™[
š[È‹ËßßJKÛÛYJš[È‹š[Ê
JJNÂˆ\ÜÙ\Ù\HJˆÚ]ØY\—Ù]™[
š[È‹KßKœ\ÚÜÝŠˆÛÜ›ŠJKˆÛÛYJš[ÈÛÜ›‹š[Ê
JBˆ
NÂˆ\ÜÙ\Ù\HJÚ]ØY\—Ù]™[
š[È‹Ëß
˜ÈHYJK›Û™JNÂˆ\ÜÙ\Ù\HJÚ]ØY\—Ù]™[
š[È‹KßK˜ÛX\Š
JK›Û™JNÂˆB‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆ›ØÚ×Ø[X\×ÜÝ\š]˜[ØÛÛœÝ[\[Û—Ø[™ØÛY[Ü™YXÝ[ÛŠ
HÂˆ\ÙHÜ˜]NŽ™Ø[YNŽžÐ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜËÛÜ›ØYÜ™\ÛÝ\˜Ù\ßNÂˆ\ÙHÜš^›Û—ØÛY[ŽžÂˆÚ[šÎŽÚ[šËˆ][\ÎŽ’][TÝXÚËˆ[ÙØÛÛ[ŽžÐÛÛ[[ÙZ[KˆNÂˆ\ÙHÜš^›Û—Û[ÙÜXÚØYÙNŽ˜ÛÛ[ŽžÐ›ØÚËÛÛ[][_NÂˆ\ÙHÝŽžÜ]Ž”]Y‹Þ[˜ÎŽ\˜ßNÂˆ]˜\ÙHH]YŽŽ™œ›ÛJˆÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK™^XÝ
œ™\\™Y™\ÛÝ\˜Ù\È™\]Z\™YŠKˆ
NÂˆ]›ÛÝHÝŽ™[ŽŽ[\Ù\Š
Kš›Ú[Š›Ü›X]JšÜš^›Û‹X[X\ËYØ[YK^ßH‹ÝŽœ›ØÙ\ÜÎŽšY

JJNÂˆ]]]ÛÛ[HÛÛ[Ž™XÛ\™Y
œXžHŠNÂˆÛÛ[ˆ™XÛ\™Jˆ	–Ð›ØÚÈÂˆYˆœXžN™œ˜[YH‹š[Ê
Kˆ[\]Nˆ›Z[™XÜ˜YœÝÛ™H‹š[Ê
Kˆ][Nˆ˜[ÙKˆ‹‘Y˜][Ž™Y˜][

BˆWKˆ	–È™[\H‹™š[Y—K›X\
˜[Y_][HÂˆYˆ›Ü›X]JœXžNžÛ˜[Y_HŠKˆ[\]Nˆ›Z[™XÜ˜YœÝXÚÈ‹š[Ê
Kˆ›ØÚÎˆÛÛYJœXžN™œ˜[YH‹š[Ê
JKˆ‹‘Y˜][Ž™Y˜][

BˆJKˆ
Bˆ[Ü˜\

NÂˆ][ÙÈHÐÛÛ[[ÙÂˆYˆœXžH‹š[Ê
KˆÛÛ[ˆ˜\œÎˆ™XÎŽ›™]Ê
KˆWNÂˆ][™[ÜžHHÜš^›Û—Û[ÙÜXÚØYÙNŽ™[˜X›YŽ‘[˜X›Y[ÙÎŽ›™]ÊÊˆœXžH‹×ÛÝÛ™Y

KˆÜš^›Û—Û[ÙÜXÚØYÙNŽ”ÚYNŽ›Ýˆ
WJNÂˆZ[
	˜˜\ÙKÝŽœ]Ž”]Ž›™]ÊˆŠK	›[ÙË	œ›ÛÝ	š[™[ÜžJK[Ü˜\

NÂˆ]
™YÚ\ÝžKÊHHÜš^›Û—ÝÛÜ›Ž›]™[Ž™]NŽ›ØYÜ™YÚ\ÝžWÝÚ]Ù›YÜÊ	œ›ÛÝ
K[Ü˜\

NÂˆÜ˜]NŽ™Ø[YNŽ˜\WÝ˜[Y\ÎŽœ™YÚ\Ý\—ÙØ[YWÚÙ^\Ê
‹œ™YÚ\ÝžK›[Š
JK›X\
YÂˆ™YÚ\ÝžBˆœÝ]JÜš^›Û—ÝÛÜ›Žœ™YÚ\ÝžNŽ›ØÚÔÝ]RY
Y\ÈLÌŠJBˆ[Ü˜\

Bˆ›˜[YBˆ˜ÛÛ™J
BˆJJNÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\‹ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆÜ[ÛœË™^XÝ][Û—Ü›ÛHHÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\ŽÂˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆØ[YKÛÜ›HÛÜ›Ž›™]ÊØ[YKœ™\Ë™˜XÝÊ
K˜Z\‹JNÂˆ›Üˆ[ˆLK‹LHÂˆ›Üˆˆ[ˆLK‹LHÂˆØ[YKÛÜ›ˆ˜Ú[šÜÂˆš[œÙ\

ŠK\˜ÎŽ›™]ÊÚ[šÎŽ™[\JŠJJNÂˆBˆBˆØ[YKœÞ[˜×ÝXÚ×ØÚ[šÜÊ
NÂˆØ[YKœÙ]Û[ÙWÚY

NÂˆØ[YKœ^Y\ˆH›Û™NÂˆØ[YKœÙ[XÝYHÂˆ]\™Ù]HØ[YBˆœXÚÂˆ˜\ÜÙ]Âˆœ™YÚ\ÝžBˆ™Y˜][ÜÝ]JœXžN™œ˜[YHŠBˆ[Ü˜\

NÂˆ›Üˆ
K˜[YJH[ˆÈœXžN™[\H‹œXžN™š[Y—Kš]\Š
K™[[Y\˜]J
HÂˆ]ÜÈHÚH\ÈLÌˆ
ˆ‹ÌNÂˆ]]HÜ˜]NŽ™Ø[YNŽ’]ÂˆÜËˆ˜XÙNˆÌKKˆ[œÚYNˆ˜[ÙKˆÚ[ˆÙŽ™œ›ÛJÜÖÌJH
ÈKÌŒWKˆNÂˆ]ÝXÚÈH][TÝXÚÎŽ›™]Ê˜[YKŠNÂˆ\ÜÙ\Ù\HJØ[YKš][\Ê
K™Ù]
˜[YJK[Ü˜\

K˜›ØÚËÛÛYJ\™Ù]
JNÂˆ]™]šY]ÈHØ[YBˆœ™[[ÝWØØ[—ÜXÙJ	š]	œÝXÚÊBˆ™^XÝ
˜ÛY[™YXÝÈH[X\ÈŠNÂˆ\ÜÙ\Ù\HJÜ™]šY]ËŒž™]šY]ËŒžK™]šY]ËŒž—KÜÊNÂˆ\ÜÙ\Ù\HJˆØ[YKÛÜ›˜›ØÚÊÜÖÌKÜÖÌWKÜÖÌ—JKˆØ[YKœ™\Ë™˜XÝÊ
K˜Z\‹ˆœ™YXÝ[Ûˆ\ÈžH‚ˆ
NÂˆ
™Ø[YKš[Û]]

HHÛÛYJÝXÚÊNÂˆ\ÜÙ\JØ[YK\ÙWÚ[™
ÛÛYJ	š]
JK[Ü˜\

JNÂˆ\ÜÙ\Ù\HJØ[YKÛÜ›˜›ØÚÊÜÖÌKÜÖÌWKÜÖÌ—JK\™Ù]
NÂˆ][HØ[YKš[

K[Ü˜\

NÂˆ\ÜÙ\Ù\HJ
	Šš[š][K[˜ÛÝ[
K

›˜[YKJJNÂˆBˆ]›ØÚÙYHÜ˜]NŽ™Ø[YNŽ’]ÂˆÜÎˆÌÌKˆ˜XÙNˆÌKKˆ[œÚYNˆ˜[ÙKˆÚ[ˆÌKÌKŒWKˆNÂˆØ[YKœÙ]Û[ÙWÚY
ŠNÂˆ]™Y›Ü™HHØ[YKš[

K˜ÛÛ™Y

NÂˆØ[YK\ÙWÚ[™
ÛÛYJ	˜›ØÚÙY
JK[Ü˜\

NÂˆ\ÜÙ\Ù\HJˆØ[YKš[

Kˆ™Y›Ü™K˜\×Ü™YŠ
Kˆ˜Y™[\™H™Y\Ø[ÙY\ÈH[X\ÈÝXÚÈ‚ˆ
NÂˆ›Ü
Ø[YJNÂˆÝŽ™œÎŽœ™[[Ý™WÙ\—Ø[
›ÛÝ
K[Ü˜\

NÂˆB‚ˆ\ÙHÜš^›Û—ØÛY[ŽžÂˆÛÛZ[™\œÎŽžÐ™XXÛÛ‹›ØÚÑ[]_Kˆ][\ÎŽ’][TÝXÚËˆNÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ›˜Ž•YÎÂ‚ˆËËÈÌLŒˆHÙ\™\‰ÜÈÛY[[Ý™\È]HÜYYHÙ\™\‰ÜÈ]šX]\ÈÚ]™H]™]™\ˆ]ÂˆËËÈY™™XÝÉÈÝÛ‹ˆ˜[š[IÜÈÙ\™\ˆ™[[Ý™\ÈHÜYY][œÈÝ][ˆÙ\™\”^Y\‹™ÕXÚËY\‚ˆËËÈHXÚÉÜÈÙ[™\Q[]Q]NˆÛY[›Ý[™™[[Ý™S[Ø‘Y™™XÝXÚÙ]ÛÙ\ÈÝ]]Û˜ÙH[™BˆËËÈ[Ý™[Y[ÜYYÚ]Ý]HY™™XÝ	ÜÈ[ÙYšY\ˆHXÚÈ]\‹ÛÈ›Üˆ]XÚÈHÛY[Ý[ˆËËÈ[œÈ˜\Ý
HÝ[Ûˆ[šÈÝ\ÈHØ[YHØ^JKˆHÛY[HXÚÈÙ™ˆ\ÈÚ]Üš[PPÉÜÂˆËËÈÚ[][][Ûˆ›YÙÙY‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆWÜÙ\™\œ×ØÛY[ÝZÙ\×Ú]×ÜÜYYÙœ›ÛWÝWØ]šX]\×Û›ÝÝWÙY™™XÝ

HÂˆ\ÙHÜ˜]NŽ™Ø[YNŽžÐ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜËØYÜ™\ÛÝ\˜Ù\ßNÂˆ\ÙHÜš^›Û—ØÛY[Ž›[Ý[ÛŽŽ”^Y\“[Ý[ÛŽÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽœ^NŽžÂˆ]šX]S[ÙYšY\‹]šX]TÛ˜\ÚÝÛY[›Ý[™\ÈË™[[Ý™S[Ø‘Y™™XÝˆ\]P]šX]\Ë\]S[Ø‘Y™™XÝˆNÂˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJˆÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK™^XÝ
œ™\\™Y™\ÛÝ\˜Ù\È™\]Z\™YŠKˆ
NÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽÛY[ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆØ[YKš[š]Ü™[[ÝWÙ›Ü—Ü™\^J
K[Ü˜\

NÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ›™]ÊKŒJJK[Ü˜\

JNÂˆ]YHHØ[YKœ™[[ÝK˜\×Ü™YŠ
K[Ü˜\

K™[]WÚYÂˆ]ÜYYHØ[YKœ™[[ÝWÙY™™XÝÛ™]ÛÜš×ÚY
œÜYYŠK[Ü˜\

NÂˆ]]šX]HHÂˆ]HØ[YKœ^Y\‹˜\×Ü™YŠ
K[Ü˜\

NÂˆ]Ù^HH˜]šX]\ËšÙ^J›Z[™XÜ˜Y›[Ý™[Y[ÜÜYYŠK[Ü˜\

NÂˆÙ^K˜]šX]J
Kœ™YÚ\ÝžWÚY

BˆNÂˆËÈ^Y\‹˜Ü™X]P]šX]\ÎˆŒQ‹‚ˆ]Ø[Ú[™ÈHŽ™œ›ÛJŒYŒÌŠNÂˆËÈÜYYRNˆŒˆ
H›Ø]
HH]™[][\Z[™ÈHÝ[‚ˆ][[Ý[HŽ™œ›ÛJŒ™ŒÌŠH
ˆ‹ŒÂˆ]]šX]\ÈHÜYˆ›ÛÛÂˆÎŽ•\]P]šX]\Ê\]P]šX]\ÈÂˆYˆYKˆ˜[Y\Îˆ™XÈVÐ]šX]TÛ˜\ÚÝÂˆ]šX]Kˆ˜\ÙNˆØ[Ú[™Ëˆ[ÙYšY\œÎˆÜYˆ[Š]šX]S[ÙYšY\ˆÂˆYˆ›Z[™XÜ˜Y™Y™™XÝœÜYY‹š[Ê
Kˆ[[Ý[ˆÜ\˜][ÛŽˆ‹ˆJBˆš[×Ú]\Š
Bˆ˜ÛÛXÝ

KˆWKˆJBˆNÂˆËÈXXÚÝ\ˆHXÚÙ]ÈÙˆÛ™HÙ\™\ˆXÚË[ˆHÛY[XÚË‚ˆ]]]Ý\HXÚÙ]Îˆ™XÏÏŸÂˆ›ÜˆXÚÙ][ˆXÚÙ]ÈÂˆØ[YKš[™WÜXÚÙ]
XÚÙ]
K[Ü˜\

NÂˆBˆØ[YKXÚ×ØÛY[ÙY™™XÝÊ
NÂˆØ[YKœ^Y\‚ˆ˜\×Û]]

Bˆ[Ü˜\

Bˆ˜]šX]J›Z[™XÜ˜Y›[Ý™[Y[ÜÜYYŠBˆ[Ü˜\

BˆNÂˆ]Y™™XÝHÎŽ•\]S[Ø‘Y™™XÝ
\]S[Ø‘Y™™XÝÂˆYˆYKˆY™™XÝˆÜYYˆ[\YšY\ŽˆKˆ\˜][ÛŽˆŒˆ›YÜÎˆˆˆJNÂˆ]˜\ÝHØ[Ú[™È
ˆ
KŒ
È[[Ý[
NÂˆ]ÛÜÙHHNˆŽˆ
HHŠK˜XœÊ
HYKNNÂˆ]ÙY[ˆHÂˆÝ\
™XÈVÙY™™XÝJKˆÝ\
™XÈVØ]šX]\ÊYJWJKˆÝ\
™XÎŽ›™]Ê
JKˆÝ\
™XÈVÐÎŽ”™[[Ý™S[Ø‘Y™™XÝ
™[[Ý™S[Ø‘Y™™XÝÂˆYˆYKˆY™™XÝˆÜYYˆJWJKˆÝ\
™XÈVØ]šX]\Ê˜[ÙJWJKˆNÂˆ]^XÝYHÝØ[Ú[™Ë˜\Ý˜\Ý˜\ÝØ[Ú[™×NÂˆ\ÜÙ\JˆÙY[‹š]\Š
Kžš\
^XÝY
K˜[

	œËJ_ÛÜÙJËJJKˆœÜYYÛˆXXÚXÚÎˆÜÙY[ŽßK˜[š[IÜÈÛY[ˆÙ^XÝYßH‚ˆ
NÂˆB‚ˆËËÈ][Qœ˜[YKš[\˜XÝXœÝ˜XÝ›Ø]š[\˜XÝ[™Z[™XØ\š[\˜XÝ‚ˆËËÈ[\˜XÝ[ÛœÈ]\ÜÈX]™HHÛXÚÈÈH[][HÚ]Ý]ÝÚ[™Ú[™ÎÂˆËËÈ™ZXÛH›Ø\™[™È[œÝÙ\œÈÓÓ”ÕSQH[™[™ÈHÛXÚÈÚ]Ý]H™YXÝYÝÚ[™È
ÎLLŒ
K‚ˆÖÝ\ÝBˆ›ˆØš™XÝÚ[\˜XÝ[Ûœ×Ü\Ü×Ø[™ÜÝÚ[™×Ý˜[š[J
HÂˆËÈ][Qœ˜[YNˆš^Yœ˜[YHÜˆ[\Hœ˜[YH
È[\H[™\ÜÙ\Ë‚ˆ\ÜÙ\J][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\ÊYKYKYJJNÂˆ\ÜÙ\J][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\ÊYK˜[ÙK˜[ÙJJNÂˆ\ÜÙ\J][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\Ê˜[ÙK˜[ÙK˜[ÙJJNÂˆËÈ][Qœ˜[YNˆ[œÙ\[™ÈÜˆ›Ý][™È][HÙ\È›Ý\ÜË‚ˆ\ÜÙ\JZ][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\Ê˜[ÙKYK˜[ÙJJNÈËÈ›Ý]\Âˆ\ÜÙ\JZ][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\Ê˜[ÙK˜[ÙKYJJNÈËÈ[œÙ\Âˆ\ÜÙ\JZ][WÙœ˜[YWÚ[\˜XÝÜ\ÜÙ\Ê˜[ÙKYKYJJNÈËÈ›Ý]\Â‚ˆËÈ›Ø]ˆÛ™XZÚ[™ÈÜˆÝ]ÙˆÛÛ›Û\ÜÙ\Ë‚ˆ\ÜÙ\J›Ø]Ú[\˜XÝÜ\ÜÙ\Ê˜[ÙKYK˜[ÙKYJJNÈËÈÛ™XZÚ[™Âˆ\ÜÙ\J›Ø]Ú[\˜XÝÜ\ÜÙ\Ê˜[ÙK˜[ÙKYKYJJNÈËÈÝ]ÙˆÛÛ›ÛˆËÈ›Ø]ˆHÛY[[œÝÙ\œÈÕPÐÑTÔË›ÛÛHÜˆ›Ý
Ý\šY[™È\ÈHÙ\™\‰ÜÊK‚ˆ\ÜÙ\JX›Ø]Ú[\˜XÝÜ\ÜÙ\Ê˜[ÙK˜[ÙK˜[ÙKYJJNÂˆ\ÜÙ\JX›Ø]Ú[\˜XÝÜ\ÜÙ\Ê˜[ÙK˜[ÙK˜[ÙK˜[ÙJJNÂ‚ˆËÈÚ\Ý›Ø]ˆÛ™XZÚ[™ÈÜ[œÈHÚ\ÝÈ›Ø\™[™È[œÝÙ\œÈÕPÐÑTÔË‚ˆ\ÜÙ\JX›Ø]Ú[\˜XÝÜ\ÜÙ\ÊYKYK˜[ÙKYJJNÈËÈÜ[œÈÚ\Ýˆ\ÜÙ\JX›Ø]Ú[\˜XÝÜ\ÜÙ\ÊYK˜[ÙK˜[ÙK˜[ÙJJNÈËÈ›Ø\™È
ÛY[
BˆËÈÚ\Ý›Ø]Ý]ÙˆÛÛ›Ûˆ\ÜÙ\ÈÚ]›ÛÛK[ÙHÜ[œÈHÚ\Ý‚ˆ\ÜÙ\J›Ø]Ú[\˜XÝÜ\ÜÙ\ÊYK˜[ÙKYKYJJNÂˆ\ÜÙ\JX›Ø]Ú[\˜XÝÜ\ÜÙ\ÊYK˜[ÙKYK˜[ÙJJNÂ‚ˆËÈÝÚ[™ÜÎˆ›Ø\™[™ÈHšYXX›HØ\[œÝÙ\œÈÓÓ”ÕSQH
›ÈÝÚ[™ÊK‚ˆ\ÜÙ\J\™[[ÝWÛØš™XÝÜÝÚ[™ÜÊ›Û™KYJJNÂˆËÈÝÚ[™ÜÎˆÝ\ˆØš™XÝ[\˜XÝ[ÛœËH›Ø]	ÜÈ›Ø\™[™ÈÛË[œÝÙ\ˆÕPÐÑTÔË‚ˆ\ÜÙ\J™[[ÝWÛØš™XÝÜÝÚ[™ÜÊ›Û™K˜[ÙJJNÂˆËÈ[\˜XÝ[Ûˆ[]Y\ÈÛ›Üˆ™\ÜÛœÙK‚ˆ\ÜÙ\J™[[ÝWÛØš™XÝÜÝÚ[™ÜÊÛÛYJYJK˜[ÙJJNÂˆ\ÜÙ\J\™[[ÝWÛØš™XÝÜÝÚ[™ÜÊÛÛYJ˜[ÙJK˜[ÙJJNÂˆB‚ˆËËÈH[Ù	ÜÈ™KYš[YÚ]Ü[œÈÛ›HÝ™\ˆHÛÜ›Ú]]È^ˆËËÈ\Y[ˆ[™›Ý[™ÈÙ[‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆ[ÙÛÜ[—ØÚ]Ùš[×ÝWØ›ÞÛÛ›WÛÝ™\—ÝWÝÛÜ›

HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽžÂˆ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËØÜ™Y[‹Ù][™ÜË^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛKˆØYÜ™\ÛÝ\˜Ù\ËˆNÂ‚ˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJˆÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠBˆ™^XÝ
œÙ]Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈÈ™\\™Y™\ÛÝ\˜Ù\ÈŠKˆ
NÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽ’[YÜ˜]YÙ\™\‹ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ››ÈY[H[™Ù™ˆŠKˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆÜ[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂ‚ˆËÈ›È^Y\ŽˆH]HØÜ™Y[‹‚ˆØ[YKœ^Y\ˆH›Û™NÂˆ\ÜÙ\JØ[YK›Ü[—ØÚ]
‹ÝŠKš\×Ù\œŠ
JNÂˆ\ÜÙ\JØ[YKZK˜Ú]ÜØÜ™Y[‹š\×Û›Û™J
JNÂˆØ[YKœ^Y\ˆHÛÛYJˆÜš^›Û—ØÛY[Ž›[Ý[ÛŽŽ”^Y\“[Ý[ÛŽŽ›™]ÊˆØ[YK›[Ý[Û‹ˆÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎŽ–‘T“Ëˆ
Bˆ[Ü˜\

Kˆ
NÂˆØ[YKœØÜ™Y[ˆHØÜ™Y[ŽŽ”]\ÙYÂˆ\ÜÙ\JØ[YK›Ü[—ØÚ]
‹ÝŠKš\×Ù\œŠ
JNÂˆ\ÜÙ\JØ[YKZK˜Ú]ÜØÜ™Y[‹š\×Û›Û™J
JNÂˆØ[YKœØÜ™Y[ˆHØÜ™Y[ŽŽ”^Z[™ÎÂˆ\ÜÙ\JØ[YK›Ü[—ØÚ]
‹ÝŠKš\×ÛÚÊ
JNÂˆ\ÜÙ\JØ[YKœØÜ™Y[ˆOHØÜ™Y[ŽŽÚ]
NÂˆ\ÜÙ\Ù\HJØ[YKZK˜Ú]ÜØÜ™Y[‹˜\×Ü™YŠ
K[Ü˜\

Kš[œ]˜[YK‹ÝŠNÂˆËÈHÚ]\ÈHØÜ™Y[ˆÛÎˆHÙXÛÛ™Ø[Ù\Û‰Ý™\XÙHÚ]	ÜÈ\Y‚ˆ\ÜÙ\JØ[YK›Ü[—ØÚ]
š[ÈŠKš\×Ù\œŠ
JNÂˆ\ÜÙ\Ù\HJØ[YKZK˜Ú]ÜØÜ™Y[‹˜\×Ü™YŠ
K[Ü˜\

Kš[œ]˜[YK‹ÝŠNÂˆB‚ˆËËÈHXÚÙ]šY[™ÈÜ˜\]\Ý™]Z[ˆ[š\ÚX›H[\›YYX\žH[]Y\ËˆHÝ\ÝÛBˆËËÈÚ]YÚ[ˆØ[ˆ\ÙH[ˆ\™XKYY™™XÝÛÝYÚÜÙHÙ\™\ˆÜÚ][ÛˆÝÜÈ\][™ÂˆËËÈÚ[H]šY\ÈH^Y\‹Ú]H^Y\ˆÜˆX\šÙ\ˆÝ[™\È]ÈÝÛˆ\ÜÙ[™Ù\‹‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆYÚ[—Ü^Y\—ÜÝXÚ×Ù›ÛÝÜ×Û™]ÛÜš×Û[Ý™\×Ø[™ÛØØ[ØØ\œšY\œÊ
HÂˆ\ÙHÝ\\ŽŽœÝ\\ŽŽžÂˆ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜË^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛKØYÜ™\ÛÝ\˜Ù\ËˆNÂˆ\ÙHÜš^›Û—ØÛY[ŽžÛ[Ý[ÛŽŽ”^Y\“[Ý[Û‹Üš[Ž’Ù^\ßNÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽœ^NŽžÜÙ[‹ÛY[›Ý[™\ÈßNÂˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK[Ü˜\

JNÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆ^XÝ][Û”›ÛNŽÛY[ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆÜ[ÛœË™^XÝ][Û—Ü›ÛHH^XÝ][Û”›ÛNŽÛY[Âˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆ]
ÛY[ÜÙ\™\ŠHHÜš^›Û—Ü›ÝØÛÛŽ˜ÛÛ›™XÝ[ÛŽŽ”\NŽœZ\Š
NÂˆØ[YK˜ÛÛ›™XÝÛØØ[
ÛY[”ÙX]\ÝŠK[Ü˜\

NÂˆØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

K™[]WÚYHÎÂˆØ[YKœ^Y\ˆHÛÛYJ^Y\“[Ý[ÛŽŽ›™]ÊØ[YK›[Ý[Û‹™XÌÎŽ–‘T“ÊK[Ü˜\

JNÂˆØ[YKœÜ]Û—Ù[]J›Z[™XÜ˜Yœ^Y\ˆ‹Ì‹ŒŒËŒKÌŒÈ×JNÂˆØ[YKœÜ]Û—Ù[]JK›Z[™XÜ˜Yœ^Y\ˆ‹ÌŒŒŒKÌŒÈ×JNÂˆØ[YKœÜ]Û—Ù[]J‹›Z[™XÜ˜Y˜\™XWÙY™™XÝØÛÝY‹ÌŒÈ×KÌŒÈ×JNÂˆ]Ù]HÎˆ	›]]Ø[YO	×Ï‹™ZXÛK\ÜÙ[™Ù\œÎˆ™XÏLÌŸÂˆËš[™WÜXÚÙ]
ÎŽ”Ù]\ÜÙ[™Ù\œÊ^NŽ”Ù]\ÜÙ[™Ù\œÈÂˆ™ZXÛKˆ\ÜÙ[™Ù\œËˆJJBˆ[Ü˜\

NÂˆNÂˆÙ]
	›]]Ø[YK™XÈVÍ—JNÂˆÙ]
	›]]Ø[YK‹™XÈVÍËWJNÂˆ]ÛÜÙHH]ˆ™XÌËˆNˆŽˆÂˆ\ÜÙ\Jˆ
]žH
K˜XœÊ
HYKMH	‰ˆ
]žHHJK˜XœÊ
HYKMH	‰ˆ
]žˆHŠK˜XœÊ
HYKMKˆœÜÚ][ÛˆØ]ßK^XÝYÞÞKÞ_KÞŸWH‚ˆ
NÂˆNÂˆ\ÜÙ\JØ[YKœ™[[ÝWÜšYJÙ^\ÎŽ™Y˜][

K
K[Ü˜\

JNÂˆÛÜÙJØ[YKœ^Y\‹˜\×Ü™YŠ
K[Ü˜\

KœÜÚ][Û‹‹ŒKËËŒ
NÂˆÛÜÙJˆØ[YK›[ØœËš]\Š
K™š[™
_KšYOHJK[Ü˜\

KœÜÚ][Û‹ˆ‹ŒˆKËˆËŒˆ
NÂˆØ[YKš[™WÜXÚÙ]
ÎŽ‘[]TÜÚ][Û”Þ[˜Ê^NŽ‘[]TÜÚ][Û”Þ[˜ÈÂˆYˆˆ]ˆ^NŽ”ÜÚ][Û”]Ž“[™X\ŠÌL‹ŒËŒMŒJKˆX]ÎˆKŒˆ]ÚˆŒˆÛ—ÙÜ›Ý[™ˆYKˆJJBˆ[Ü˜\

NÂˆ›ÜˆÈ[ˆ‹ÂˆØ[YKXÚ×Ü™[[ÝWÙ[]Y\ÊYJNÂˆØ[YKœ™[[ÝWÜšYJÙ^\ÎŽ™Y˜][

K
K[Ü˜\

NÂˆBˆÛÜÙJØ[YKœ^Y\‹˜\×Ü™YŠ
K[Ü˜\

KœÜÚ][Û‹L‹ŒŽËMŒ
NÂˆÛÜÙJˆØ[YK›[ØœËš]\Š
K™š[™
_KšYOHJK[Ü˜\

KœÜÚ][Û‹ˆL‹ŒˆŽËˆMŒˆ
NÂˆËÈH[\›YYX\žIÜÈXÚÙ]ÜÚ][Ûˆ]\Ý™[XZ[ˆ[˜Ú[™ÙY›Üˆ™[]]™HXÛÙ[™Ë‚ˆ\ÜÙ\Ù\HJˆØ[YKœ™[[ÝK˜\×Ü™YŠ
K[Ü˜\

K˜XÚÙY
ŠK[Ü˜\

KŒKˆÌŒÈ×Bˆ
NÂˆËÈH™[[ÝH^Y\ˆšY[™ÈHØØ[^Y\ˆ]\Ý›ÛÝÈÝ\ˆÝÛˆ[Ý™[Y[ÛË‚ˆÙ]
	›]]Ø[YK‹™XÈVÍWJNÂˆÙ]
	›]]Ø[YKË™XÈVÍ—JNÂˆØ[YKœ^Y\‹˜\×Û]]

K[Ü˜\

KœÜÚ][ÛˆH™XÌÎŽ›™]ÊŒŒÌŒKŒ
NÂˆØ[YKœÜÚ][Û—ÜšY\œÊ
NÂˆÛÜÙJˆØ[YK›[ØœËš]\Š
K™š[™
_KšYOHJK[Ü˜\

KœÜÚ][Û‹ˆŒŒˆÌKËˆKŒˆ
NÂˆËÈH\™XÝÝXÚÈ\È›È[\›YYX\žIÜÈKZZYÚÙ™œÙ]‚ˆÙ]
	›]]Ø[YK™XÈVÍËWJNÂˆØ[YKœ™[[ÝWÜšYJÙ^\ÎŽ™Y˜][

K
K[Ü˜\

NÂˆÛÜÙJØ[YKœ^Y\‹˜\×Ü™YŠ
K[Ü˜\

KœÜÚ][Û‹L‹ŒŽŒ‹MŒ
NÂˆÛÜÙJˆØ[YK›[ØœËš]\Š
K™š[™
_KšYOHJK[Ü˜\

KœÜÚ][Û‹ˆL‹ŒˆŽŒ‹ˆMŒˆ
NÂˆËÈ™[[Ýš[™ÈHØ\œšY\ˆ]XÚ\È\ÜÙ[™Ù\œÈ]™[ˆÚ]Ý]Hš[˜[[\HXÚÙ]‚ˆØ[YKœ™[[Ý™WÙ[]J
NÂˆ\ÜÙ\Ù\HJØ[YKœšY[™Ë›Û™JNÂˆ\ÜÙ\Ù\HJˆØ[YK›[ØœËš]\Š
K™š[™
_KšYOHJK[Ü˜\

K™^˜K›[Ý[ˆ›Û™Bˆ
NÂˆB‚ˆËËÈ[]Kš\ÐÛÛY[™Ø[™HÛ˜\ÛY[]™[œÞ[˜Ð›ØÚÔÝ]XXZÙ\ÎˆBˆËËÈ›ØÚÈHÙ\™\ˆ]È˜XÚÈÚ\™HH^Y\ˆÝ[™ÈÙ[™È]ÈBˆËËÈÜÚ][ÛˆH™YXÝ[Ûˆ™[Y[X™\™Y[™H›ØÚÈ]ÛÛY\È˜XÚÈÛX\ˆÙ‚ˆËËÈH^Y\‹[ˆ[\HÛ™KÜˆÛ™HH^Y\ˆ\È™Y[ˆ[\ÜYÚ[˜ÙBˆËËÈX]™\È]Ú\™H]\Ë‚ˆÖÝ\ÝBˆ›ˆWØ›ØÚ×ÝWÜÙ\™\—Ü]×Ø˜XÚ×ÜÛ˜\×ÝWÜ^Y\—ÛÝ]ÛÙ—Ú]

HÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽžÐXX˜‹™XÌË›Þ[Ú\_NÂˆËÈH^Y\ˆÝ[™[™È[ˆ›ØÚÈ

NˆˆÚYKKŽ[‚ˆ]^Y\ˆHXX˜ŽŽ›™]Ê™XÌÎŽ›™]ÊŒ‹ŒŒŠK™XÌÎŽ›™]ÊŽKŽŽ
JNÂˆ]ÝX™HH›Þ[Ú\NŽ˜›ØÚÊ
NÂˆ\ÜÙ\JˆÚ\WÛYY]Ê	˜ÝX™KÌK^Y\ŠKˆH›ØÚÈH^Y\ˆ\ÈÝ[™[™È[ˆ‚ˆ
NÂˆ\ÜÙ\J\Ú\WÛYY]Ê	˜ÝX™KÌ‹K^Y\ŠK˜X›Ý™H]ÈXYŠNÂˆ\ÜÙ\Jˆ\Ú\WÛYY]Ê	˜ÝX™KÌKK^Y\ŠKˆH›ØÚÈ™\ÚYH]‚ˆ
NÂˆ\ÜÙ\J\Ú\WÛYY]Ê	•›Þ[Ú\NŽ™[\J
KÌK^Y\ŠJNÂ‚ˆËÈHÛ˜\]Ù[ŽˆH™[Y[X™\™YÜÚ][Û‹[›\ÜÈH[\ÜØ[YHY\‚ˆËÈH™YXÝ[Ûˆ
\Ý[\ÜÙ\]Y[˜ÙHÙ\]Y[˜ÙX
K‚ˆ]]HÌKŒWNÂˆ\ÜÙ\Ù\HJÛ˜\Ý\™Ù]
]LKËYJKÛÛYJ]
JNÂˆ\ÜÙ\Ù\HJÛ˜\Ý\™Ù]
]LKË˜[ÙJK›Û™K˜H›ØÚÈ]ÛX\œÈŠNÂˆ\ÜÙ\Ù\HJÛ˜\Ý\™Ù]
]ËËYJK›Û™K[\ÜY]ÈŠNÂˆ\ÜÙ\Ù\HJˆÛ˜\Ý\™Ù]
]‹ËYJKˆÛÛYJ]
Kˆ[\ÜY™Y›Ü™H]‚ˆ
NÂˆB‚ˆËËÈ][T^Y\‘Ø[YS[ÙK\ÙR][X	ÜÈÚÝ[ÝÚ[™ØˆHÕPÐÑTÔÈ[œÝÙ\œÂˆËËÈÝÚ[™ËÓÓ”ÕSQIÜÈÈ›Ý
ÍJK‚ˆÖÝ\ÝBˆ›ˆWÝ›ÝÛ—Ú][WÜÝÚ[™Ú[™×ÛÛ—ØWÜÙ\™\—Ø[™ØWØÚ\™ÙWÛ›Ý

HÂˆ]ÝXÚÈHYˆ	œÝŸ][TÝXÚÎŽ›™]ÊYJNÂˆËÈÕPÐÑTÔÎˆH›ÝØX›\ËHXÚÙ]H›Ø]H[™KH›ÛÚËHX\‚ˆ›ÜˆY[ˆÂˆ›Z[™XÜ˜Y™[™\—ÜX\›‹ˆ›Z[™XÜ˜YœÜ\ÚÜÝ[Ûˆ‹ˆ›Z[™XÜ˜YÚ[™ØÚ\™ÙH‹ˆ›Z[™XÜ˜YœÛ›ÝØ˜[‹ˆ›Z[™XÜ˜Y™^\šY[˜ÙWØ›ÝH‹ˆ›Z[™XÜ˜Y˜XÚÙ]‹ˆ›Z[™XÜ˜Y›ØZ×Ø›Ø]‹ˆ›Z[™XÜ˜Y˜[™H‹ˆ›Z[™XÜ˜YÜš]X›WØ›ÛÚÈ‹ˆ›Z[™XÜ˜Y›X\‹ˆHÂˆ\ÜÙ\J™[[ÝWÝ\ÙWÜÝÚ[™ÜÊ	œÝXÚÊY
K˜[ÙJKžÚYHÝÚ[™ÜÈŠNÂˆBˆËÈÓÓ”ÕSQNˆHÚ\™Ù\È[™Ý\\Ú[™Ò[œÝ[IÜÈ\Ù\ËHš[šØX›BˆËÈÝ[Ûˆ[[Û™È[H
ÛÛœÝ[XX›XÙ]
K‚ˆ›ÜˆY[ˆÂˆ›Z[™XÜ˜Y˜›ÝÈ‹ˆ›Z[™XÜ˜Y˜Ü›ÜÜØ›ÝÈ‹ˆ›Z[™XÜ˜YšY[‹ˆ›Z[™XÜ˜YœÜYÛ\ÜÈ‹ˆ›Z[™XÜ˜Y™ÛØ]ÚÜ›ˆ‹ˆ›Z[™XÜ˜Y™[™\—Ù^YH‹ˆHÂˆ\ÜÙ\J\™[[ÝWÝ\ÙWÜÝÚ[™ÜÊ	œÝXÚÊY
K˜[ÙJKžÚYHÙ\È›ÝÝÚ[™ÈŠNÂˆBˆËÈHÚ[™]XÈÙX\ÛœÈÚ\™ÙHZÙHHšY[‚ˆ\ÜÙ\J\™[[ÝWÝ\ÙWÜÝÚ[™ÜÊ	œÝXÚÊ›Z[™XÜ˜Yš\›Û—ÜÜX\ˆŠK˜[ÙJJNÂˆËÈHÛÛœÝ[XX›H
›ÛÙHš[šÊHÝ\È]È\ÙHÚ]Ý]HÝÚ[™Ë‚ˆ\ÜÙ\J\™[[ÝWÝ\ÙWÜÝÚ[™ÜÊ	œÝXÚÊ›Z[™XÜ˜Y˜\HŠKYJJNÂˆB‚ˆËËÈZ[™XÜ˜YXÚÎˆÓQS•ÕPÒ×ÑS‘Û›HÚ[HH]™[^\ÝËˆ›Û™H™Y›Ü™BˆËËÈHš\œÝÙÚ[ˆ
ÛÛ™šYÝ\˜][Ûˆ\Èš[š\ÚYHÙ\™\ˆ\È›ÝXÙYˆËËÈH^Y\ˆY]
K[™›Û™Hœ›ÛHÝ\ÛÛ™šYÝ\˜][Ûˆ[[H™^ÙÚ[‹‚ˆÖÝ\ÝBˆ›ˆXÚ×Ù[™×Û™YYÝWÛ]™[ÝWÛÙÚ[—ÛXZÙ\Ê
HÂˆ\ÜÙ\JXÛY[Ú\×Û]™[
˜[ÙK›Û™JKš›Ú[™YØZ][™È›ÜˆÙÚ[ˆŠNÂˆ\ÜÙ\JÛY[Ú\×Û]™[
YK›Û™JK˜Y\ˆÙÚ[ˆŠNÂˆ\ÜÙ\JXÛY[Ú\×Û]™[
YKÛÛYJ
JKœ™XÛÛ™šYÝ\š[™ÈŠNÂˆB‚ˆËËÈÌLŒMˆH›ÞH[Ý™\ÈH^Y\ˆÈ]È™^Ù\™\‹ˆH\ÝÙ\™\‰ÜÂˆËËÈÛÛ[[œÈÝ^H[[H™^ÙÚ[‹]™XÛÛ™šYÝ\™Y[™XYH][ˆBˆËËÈ™^Ù\™\‰ÜÈš[ÛYHX›KÚÜ\ˆÚ[ˆÛ›HH\ÝÙ\™\ˆYBˆËËÈš[ÛYHHØ[YHXÚÜÎˆHÛÛ[[œÈÛ[ˆY\Ý]
H[ˆ\ÈÂˆËËÈ]H[™^\ÈÈŠKˆ˜[š[IÜÈÛX\ÛY[]™[X]™\È›È^Y\‹ÛÂˆËËÈÙ]Ú]X][Û˜[]\ÚXÈ\ÈQS•K[™Ù\™\”™XÛÛ™šYÔØÜ™Y[ˆ
›ÂˆËËÈ]™[ØY[™ÔØÜ™Y[ŠH]È]\ÚXÓX[˜YÙ\‹XÚÈÝ\]‚ˆÖÝ\ÝBˆÖÚYÛ›Ü™HH›™YYÈ™\\™Y™\ÛÝ\˜Ù\È[ˆÔ’V“Ó—ÕTÕÔ‘TÓÕTÑTÈ—Bˆ›ˆWÜ™XÛÛ™šYÝ\š[™×ØÛY[Ü^\×ÛY[WÛ]\ÚX×Ø[™Û™]™\—Ú[™^\×Ü\ÝÝWØš[ÛY\Ê
HÂˆ\ÙHÜ˜]NŽ™Ø[YNŽžÐ]Y[ËØ[YKÜ[ÛœË›ÙÜ™\ÜËÙ][™ÜËØYÜ™\ÛÝ\˜Ù\ßNÂˆ\ÙHÜš^›Û—ØÛY[ŽžØÚ[šÎŽÚ[šË]\ÚXÎŽ“]\ÚXßNÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎÂˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJˆÝŽ™[ŽŽ˜\—ÛÜÊ’Ô’V“Ó—ÕTÕÔ‘TÓÕTÑTÈŠK™^XÝ
œ™\\™Y™\ÛÝ\˜Ù\È™\]Z\™YŠKˆ
NÂˆ]Ù][™ÜÈHÙ][™ÜÎŽ™Y˜][

NÂˆ]ØYYHØYÜ™\ÛÝ\˜Ù\ÊˆÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙT›ÛÝŽ™\™XÝÜžJ›ÛÝ˜ÛÛ™J
JKˆ™XÎŽ›™]Ê
Kˆ›Û™Kˆ	œ›ÛÝˆ	œÙ][™ÜËˆÝŽ[YNŽ’[œÝ[Ž››ÝÊ
Kˆ˜[ÙKˆÜ˜]NŽ™Ø[YNŽ™^XÝ][Û—Ü›ÛNŽ‘^XÝ][Û”›ÛNŽÛY[ˆ	”›ÙÜ™\ÜÎŽ›™]Ê
Kˆ	›]]Ëß[œ™XXÚX›HJ
Kˆ
Bˆ[Ü˜\

NÂˆ]]]Ü[ÛœÈHÜ[ÛœÎŽœ\œÙJÝŽš]\ŽŽ™[\J
JK[Ü˜\

K[Ü˜\

NÂˆÜ[ÛœËœ™\ÛÝ\˜Ù\ÈH›ÛÝ˜ÛÛ™J
NÂˆ]]]Ø[YHHØ[YNŽ›™]Êˆ	›ØYYœ™\ÛÝ\˜Ù\ËˆØYYœXÚËˆØYY™ÝZK[Ü˜\

KˆÜ[ÛœËˆÙ][™ÜËˆ]Y[ÎŽœÚ[[
	šÜš^›Û—Ø\ÜÙ]ÎŽœ™\ÛÝ\˜ÙWÜÛÝ\˜ÙNŽ”™\ÛÝ\˜ÙTÛÝ\˜ÙNŽ™\™XÝÜžJˆ›ÛÝ˜ÛÛ™J
Kˆ
JKˆ›Û™Kˆ
Bˆ[Ü˜\

NÂˆØ[YKš[š]Ü™[[ÝWÙ›Ü—Ü™\^J
K[Ü˜\

NÂˆËÈH\ÝÙ\™\‰ÜÈÛÛ[[ˆ]HÜšYÚ[Žˆ]È^˜Hš[ÛYHØ[YHY\‚ˆËÈHZ[Z[ˆÛ™\ËÛ™H\ÝHX›HH™^Ù\™\ˆX]™\Ë‚ˆ]Ý[HHÛÜ›Øš[ÛY\ÈJØ[YJK›\Ý›[Š
H\ÈLMŽÂˆ]]]ÛÛ[[ˆHÚ[šÎŽ™[\J
NÂˆËÈ]X\MˆX›Ý™HHÛÛ[[‰ÜÈ›ÝÛH]Mˆ›ØÚÈH‚ˆÛÛ[[‹œÙ]Øš[ÛYWØÙ[
M‹Ý[JNÂˆØ[YKÛÜ›ˆ˜Ú[šÜÂˆš[œÙ\


KÝŽœÞ[˜ÎŽ\˜ÎŽ›™]ÊÛÛ[[ŠJNÂˆ\ÜÙ\Ù\HJØ[YK˜š[ÛYWØ]
ÌŒÈ×JK›Z[™XÜ˜YœZ[œÈŠNÂˆ\ÜÙ\Ù\HJˆØ[YK˜š[ÛYWÛ˜[YWØ]
™XÌÎŽ›™]ÊKKJJK˜\×Ù\™YŠ
KˆÛÛYJ›Z[™XÜ˜YœZ[œÈŠBˆ
NÂ‚ˆËÈÝ\ÛÛ™šYÝ\˜][ÛˆY\ˆHÙÚ[Žˆ›È^Y\‹Ù\™\”™XÛÛ™šYÔØÜ™Y[‹‚ˆ]XÚÜÈHØ[YKXÚÜÎÂˆ]ˆHØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

NÂˆ‹›ÙÙÙYÚ[ˆHYNÂˆ‹œ™XÛÛ™šYÝ\š[™ÈHÛÛYJXÚÜÊNÂˆØ[YKœ^Y\ˆH›Û™NÂˆ\ÜÙ\Ù\HJØ[YKœÚ]X][Û˜[Û]\ÚXÊ
KŒÛÛYJ]\ÚXÎŽ›Y[J
JJNÂˆËÈH™]ÈX[˜YÙ\ˆØZ]È]Èš\œÝLXÚÜË[ˆÝ\ÈHY[H˜XÚË‚ˆ]Y[HH
‹LL
K˜[žJßÂˆØ[YKXÚ×Û]\ÚXÊ
NÂˆØ[YK›]\ÚXË˜Ý\œ™[
	™Ø[YK˜]Y[Ë™[™Ú[™JHOHÛÛYJ›Z[™XÜ˜Y›]\ÚXË›Y[HŠBˆJNÂˆ\ÜÙ\JY[KHY[H]\ÚXÈÝ\È[™\ˆÙ\™\”™XÛÛ™šYÔØÜ™Y[ˆŠNÂ‚ˆËÈÚ]H]™[YØZ[ˆHØ[Y\˜H›Ø™H™XYÈ]ÛÛ[[ŽˆZ[œÉÈ]\ÚXË‚ˆØ[YKœ™[[ÝK˜\×Û]]

K[Ü˜\

Kœ™XÛÛ™šYÝ\š[™ÈH›Û™NÂˆ]
]\ÚXËÊHHØ[YKœÚ]X][Û˜[Û]\ÚXÊ
NÂˆ\ÜÙ\J]\ÚXËš\×ÜÛÛYWØ[™
_HOH]\ÚXÎŽ›Y[J
JJNÂˆB‚ˆËËÈHÙ\™\‰ÜÈÛ‹Yš\™H›YÈÙY\ÈH™YXÝYÛÝ[ÝÛˆ
ÚXÚˆËËÈ›ZYØ˜\ÙWÝXÚØ[œÈÝÛŠHœ›ÛH[™[™ÈHš\™NÈHÛX\ˆ›YÈX]™\ÂˆËËÈHÚ[™ÝÉÜÈÝÛˆÛÛXÝš\™H[Û™K‚ˆÖÝ\ÝBˆ›ˆWÜÙ\™\œ×Ùš\™WÙ›Y×ÚÛ×ÝWØ\›š[™×ÜÝ]WÝ›ÝYÚÝWØÛÝ[ÝÛŠ
HÂˆ]]]XÚÜÈHNÂˆ›ÜˆÈ[ˆ‹HÂˆXÚÜÈHÛÜÙ\™\—Ùš\™JXÚÜÈHKYJNÂˆ\ÜÙ\JXÚÜÈˆœÝ[\›š[™ÈŠNÂˆBˆ\ÜÙ\Ù\HJÛÜÙ\™\—Ùš\™JÌYJKÌ˜HÛ™Ù\ˆš\™H\È›ÝÝ]ŠNÂˆ\ÜÙ\Ù\HJÛÜÙ\™\—Ùš\™JË˜[ÙJKËœ™YXÝY]˜Hš\™HÝ^\ÈŠNÂˆ\ÜÙ\Ù\HJÛÜÙ\™\—Ùš\™J˜[ÙJK
NÂˆB‚ˆÖÝ\ÝBˆ›ˆWØ™XXÛÛ—Ý\]WÝY×ÚÙY\×ÝWØ™X[WÝWØÛY[ÝØ[ÙY

HÂˆ]ÛH›ØÚÑ[]NŽ™XXÛÛŠ›ÞŽ›™]Ê™XXÛÛˆÂˆ]™[ÎˆKˆš[X\žNˆÛÛYJ›Z[™XÜ˜YœÜYY‹š[Ê
JKˆÙXÝ[ÛœÎˆ™XÈVÊŽQ‘‘‘KŠWKˆÚXÚÚ[™Îˆ™XÈVÊŽQ‘‘‘KJWKˆ\ÝØÚXÚ×ÞNˆÌˆ‹‘Y˜][Ž™Y˜][

BˆJJNÂˆËÈ™XXÛÛ›ØÚÑ[]K™Ù]\]UYÎˆØ]™PÝ\ÝÛSÛ›K‚ˆ]YÈHYÎŽ˜ÛÛ\Ý[™
Âˆ
šY‹YÎŽ”Ýš[™Ê›Z[™XÜ˜Y˜™XXÛÛˆ‹š[Ê
JJKˆ
ž‹YÎŽ’[

JKˆ
žH‹YÎŽ’[

JKˆ
žˆ‹YÎŽ’[

JKˆ
œš[X\žWÙY™™XÝ‹YÎŽ”Ýš[™Ê›Z[™XÜ˜Yš\ÝH‹š[Ê
JJKˆ
ˆœÙXÛÛ™\žWÙY™™XÝ‹ˆYÎŽ”Ýš[™Ê›Z[™XÜ˜Yœ™YÙ[™\˜][Ûˆ‹š[Ê
JKˆ
Kˆ
“]™[È‹YÎŽ’[

JKˆJNÂˆ]
Ë]]™]ÊHHÜš^›Û—ØÛY[ŽœØ]™NŽ™XÛÙWØ›ØÚ×Ù[]J	YÊK[Ü˜\

NÂˆÙY\ØÛY[ÜÝ]JÛÛYJ	›Û
K	›]]™]ÊNÂˆ]›ØÚÑ[]NŽ™XXÛÛŠŠHH™]È[ÙHÂˆ[šXÈJ››ÝH™XXÛÛˆŠNÂˆNÂˆ\ÜÙ\Ù\HJ‹œÙXÝ[ÛœË™XÈVÊŽQ‘‘‘KŠWJNÂˆ\ÜÙ\Ù\HJ‹˜ÚXÚÚ[™Ë™XÈVÊŽQ‘‘‘KJWJNÂˆ\ÜÙ\Ù\HJ‹›\ÝØÚXÚ×ÞKÌ
NÂˆËÈHÝÙ\œÈ\™HHYÉÜÎÈH]™[ÈÝ^HHÛY[	ÜÈÝÛˆ
›Ý™XY
K‚ˆ\ÜÙ\Ù\HJ‹œš[X\žK˜\×Ù\™YŠ
KÛÛYJ›Z[™XÜ˜Yš\ÝHŠJNÂˆ\ÜÙ\Ù\HJ‹œÙXÛÛ™\žK˜\×Ù\™YŠ
KÛÛYJ›Z[™XÜ˜Yœ™YÙ[™\˜][ÛˆŠJNÂˆ\ÜÙ\Ù\HJ‹›]™[ËJNÂˆB‚ˆÖÝ\ÝBˆ›ˆWÜÙ\™\—ÜÝÜ[™×ÝWÝ\ÙWÛÝÙ\œ×ÝWÜÚY[

HÂˆËÈ^Y\‹™\ØX›TÚY[	ÜÈÝÜ\Ú[™Ò][Nˆ›YÜÈÜˆˆÚ[HBˆËÈÙ™‹Z[™ÚY[\È˜Z\ÙY
ØØ[^Y\‹›Û”Þ[˜ÙY]U\]Y8¡¤‚ˆËÈÝÜ\Ú[™Ò][JKˆÝÜ\Ú[™Ò][HÛX\œÈš][™X]™\Èš]KÛÈBˆËÈXZ[‹Z[™ÝÜ\È[™[ˆÙ™‹Z[™ÝÜ\È‹‚ˆ]Ù™ˆHÝ\\ŽŽœÝ\\ŽŽœÚY[Ž“Ñ‘’S‘ÔÓÕÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJÛÛYJ
Ù™‹LŠJKËßYJK›Û™JNÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJÛÛYJ
ËJJKËßYJK›Û™JNÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJÛÛYJ
Ù™‹LŠJK‹ËßYJK›Û™JNÂˆB‚ˆÖÝ\ÝBˆ›ˆWÜÙ\™\—ÜÝ\[™×ØWÝ\ÙWÜÝ\×Ú]Ú[—Ý]Ú[™

HÂˆ]Ù™ˆHÝ\\ŽŽœÝ\\ŽŽœÚY[Ž“Ñ‘’S‘ÔÓÕÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJ›Û™KKßYJKÛÛYJ

JJNÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJ›Û™KËßYJKÛÛYJ
Ù™‹
JJNÂˆËÈ]š[™Ñ[]KœÝ\\Ú[™Ò][H™YYÈ[ˆ][H[ˆH[™‚ˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJ›Û™KKß˜[ÙJK›Û™JNÂˆB‚ˆËËÈÛÐšYÕÒ[\œÛ]NˆHÜÚ][ÛˆÞ[˜ÈYX\Ý\™\Èœ›ÛHÚ\™HH[]BˆËËÈ\ËH[\Üœ›ÛHÚ\™H]È[\œÛ][Ûˆ\ÈXYY‚ˆÖÝ\ÝBˆ›ˆWÙ˜\—Û[Ý™WÜÛ˜\×ØžWÝWÜXÚÙ]×ÛÝÛ—ÛYX\Ý\™J
HÂˆ\ÙHÝ\\ŽŽžÓ[Ý™TXÚÙ]]ÜÔ›ÝÝ\Y[\œÛ][ÛŸNÂˆ]]HˆÜÔ›ÝÂˆÜÚ][ÛŽˆÞŒŒKˆX]ÎˆŒˆ]ÚˆŒˆNÂˆ]]][™\ˆHÝ\Y[\œÛ][ÛŽŽ›™]ÊÊNÂˆËÈ]XYY›ÜˆL‚ˆ[™\‹š[\œÛ]WÝÊ]
Œ
KÛÛYJ]Ž“[™X\ŠÍLŒŒŒJJK›Û™JNÂˆ]ÈHÌLŒŒŒNÂˆ\ÜÙ\J[Ý™TXÚÙ]Ž”ÜÚ][Û”Þ[˜ËœÛ˜\Ê	š[™\‹]
Œ
KÊJNÂˆ\ÜÙ\JS[Ý™TXÚÙ]Ž•[\ÜœÛ˜\Ê	š[™\‹]
Œ
KÊJNÂˆ\ÜÙ\JS[Ý™TXÚÙ]Ž“[Ý™KœÛ˜\Ê	š[™\‹]
Œ
KÊJNÂˆËÈÚ]›Ý[™È[ˆ›YÚ›ÝYX\Ý\™Hœ›ÛHH[]K‚ˆ]YHHÝ\Y[\œÛ][ÛŽŽ›™]ÊÊNÂˆ\ÜÙ\J[Ý™TXÚÙ]Ž•[\ÜœÛ˜\Ê	šYK]
Œ
KÊJNÂˆ\ÜÙ\JS[Ý™TXÚÙ]Ž•[\ÜœÛ˜\Ê	šYK]
Œ
KÍŒŒŒŒJJNÂˆB‚ˆËËÈH˜\ˆÞ[˜È]ÈH[]H\™H[™›ÜÈHÝ\È]Ø\ÈØ[Ú[™ÎÂˆËËÈH›Ý][Ûˆ[Û™HÙY\ÈHÜÚ][Ûˆ]\ÈXYY›Üˆ[™\›œÈ\™BˆËËÈÝ™\ˆH\]H[\˜[‚ˆÖÝ\ÝBˆ›ˆWÛ]š[™×Ù[]\×Û[Ý™WÜÛ˜\×ÛÜ—ÜÝ\Ê
HÂˆ\ÙHÝ\\ŽŽžÓ[Ý™TXÚÙ]]ÜÔ›ÝÝ\Y[\œÛ][Û‹Ý\YÛ[Ý™_NÂˆ]]HˆX]ÎˆŒÌŸÜÔ›ÝÂˆÜÚ][ÛŽˆÞŒŒKˆX]Ëˆ]ÚˆŒˆNÂˆ]]][™\ˆHÝ\Y[\œÛ][ÛŽŽ›™]ÊÊNÂˆËÈØ[Ú[™ÈÈHÚ[ˆH˜\ˆÞ[˜ÈÛÛY\Ë‚ˆ[™\‹š[\œÛ]WÝÊ]
ŒŒ
KÛÛYJ]Ž“[™X\ŠÍKŒŒŒJJK›Û™JNÂˆ\ÜÙ\J[™\‹˜XÝ]™J
JNÂˆ]˜\ˆHÌLŒŒŒNÂˆ]Û˜\HÝ\YÛ[Ý™Jˆ	›]][™\‹ˆ[Ý™TXÚÙ]Ž”ÜÚ][Û”Þ[˜Ëˆ]
ŒŒ
Kˆ˜\‹ˆÛÛYJ]Ž“[™X\Š˜\ŠJKˆÛÛYJ
LŒŒ
JKˆ
NÂˆ\ÜÙ\Ù\HJÛ˜\ÛÛYJ]
LŒLŒ
JJNÂˆ\ÜÙ\JZ[™\‹˜XÝ]™J
JNÂˆ\ÜÙ\Ù\HJ[™\‹š[\œÛ]J
K›Û™K››ÈÝ\˜XÚÈÝØ\™HŠNÂˆËÈH›Ý][Û‹[Û›H[Ý™HXÚÙ]‚ˆ]\›ˆHÝ\YÛ[Ý™Jˆ	›]][™\‹ˆ[Ý™TXÚÙ]Ž“[Ý™Kˆ]
LŒLŒ
Kˆ˜\‹ˆ›Û™KˆÛÛYJ
NŒŒ
JKˆ
NÂˆ\ÜÙ\Ù\HJ\›‹›Û™JNÂˆ]Ý\Îˆ™XÏÏˆHÝŽš]\ŽŽ™œ›ÛWÙ›Š[™\‹š[\œÛ]J
JK˜ÛÛXÝ

NÂˆ\ÜÙ\Ù\HJˆÝ\Ëš]\Š
K›X\
ßËžX]ÊK˜ÛÛXÝŽ™XÏÏŠ
KˆÌLŒŒMLŒNŒBˆ
NÂˆ\ÜÙ\JÝ\Ëš]\Š
K˜[
ßËœÜÚ][ÛˆOH˜\ŠJNÂˆB‚ˆÖÝ\ÝBˆ›ˆX]Ú[™×Ù›YÜ×ÛX]™WØWÜ[›š[™×Ý\ÙWØ[Û™J
HÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJÛÛYJ
‹ÊJKK‹ßYJKÛÛYJ
‹ÊJJNÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÛØØ[Ý\ÙJ›Û™K‹ßYJK›Û™JNÂˆB‚ˆËËÈÚ]\K›Ý[™™XÛÜ˜]X[™XÛÜ˜]S˜\œ˜][Û˜ˆHš]˜]HY\ÜØYÙBˆËËÈÚÝÜÈ\È”Ý]™HÚ\Ü\œÈÈ[ÝNˆHˆ[™\È˜\œ˜]Y”Ý]™HØ^\ÈHŽÂˆËËÈHÚ]\HH™YÚ\ÝžHXÚÜÈ\ÈZ[™XÜ˜Y˜Ú]	ÜË‚ˆÖÝ\ÝBˆ›ˆÚ]Ú\×Û˜\œ˜]YÝÚ]Ú]×ØÚ]Ý\\×Û˜\œ˜][Û—ÙXÛÜ˜][ÛŠ
HÂˆ\ÙHÝ\\ŽŽžÑXÛÜ˜][Û‹Ú]Ý\WÙXÛÜ˜]K^_NÂˆ\ÙHÜš^›Û—ØÛÜ™NŽ›˜ŽžÒ˜]˜TÝš[™ËYË\œÙWÜÛ˜NÂˆ\ÙHÙ\™WÚœÛÛŽŽšœÛÛŽÂˆ]›Ý[™H^NŽÚ]\P›Ý[™ÂˆÚ]Ý\Nˆˆ˜[YNˆYÎŽ”Ýš[™Ê˜]˜TÝš[™ÎŽ™œ›ÛJ”Ý]™HŠJKˆ\™Ù]ˆ›Û™KˆNÂˆ]ÛÛ[HœÛÛˆJšHŠNÂˆ]
[˜ÛÛZ[™ËÊHH\œÙWÜÛ˜
ˆˆÈžØÚ]žÝ˜[œÛ][Û—ÚÙ^Nˆ˜ÛÛ[X[™Ë›Y\ÜØYÙK™\Ü^Kš[˜ÛÛZ[™È‹\˜[Y]\œÎ–ÈœÙ[™\ˆ‹˜ÛÛ[—KÝ[NžØÛÛÜŽˆ™Ü˜^H‹][XÎŒXŸ_K˜\œ˜][ÛŽžÝ˜[œÛ][Û—ÚÙ^Nˆ˜Ú]\K^›˜\œ˜]H‹\˜[Y]\œÎ–ÈœÙ[™\ˆ‹˜ÛÛ[—__HˆËˆ
Bˆ[Ü˜\

NÂˆ]Ú]HÚ]Ý\WÙXÛÜ˜]JÛÛYJ	š[˜ÛÛZ[™ÊKXÛÜ˜][ÛŽŽÚ]	˜›Ý[™	˜ÛÛ[
NÂˆ\ÜÙ\Ù\HJÚ]È˜[œÛ]H—K˜ÛÛ[X[™Ë›Y\ÜØYÙK™\Ü^Kš[˜ÛÛZ[™ÈŠNÂˆ\ÜÙ\Ù\HJÚ]È˜ÛÛÜˆ—K™Ü˜^HŠNÂˆ\ÜÙ\Ù\HJÚ]ÈÚ]—KœÛÛˆJÈ”Ý]™H‹šH—JJNÂˆ\ÜÙ\Ù\HJˆÚ]Ý\WÙXÛÜ˜]JÛÛYJ	š[˜ÛÛZ[™ÊKXÛÜ˜][ÛŽŽ“˜\œ˜][Û‹	˜›Ý[™	˜ÛÛ[
KˆœÛÛˆJÈ˜[œÛ]HŽˆ˜Ú]\K^›˜\œ˜]H‹Ú]ŽˆÈ”Ý]™H‹šH—_JBˆ
NÂˆ\ÜÙ\Ù\HJˆÚ]Ý\WÙXÛÜ˜]J›Û™KXÛÜ˜][ÛŽŽ“˜\œ˜][Û‹	˜›Ý[™	˜ÛÛ[
KˆœÛÛˆJÈ˜[œÛ]HŽˆ˜Ú]\K^›˜\œ˜]H‹Ú]ŽˆÈ”Ý]™H‹šH—_JBˆ
NÂˆ\ÜÙ\Ù\HJˆÚ]Ý\WÙXÛÜ˜]J›Û™KXÛÜ˜][ÛŽŽÚ]	˜›Ý[™	˜ÛÛ[
VÈ˜[œÛ]H—Kˆ˜Ú]\K^‚ˆ
NÂˆB‚ˆËËÈHØ][›Û‰ÜÈš[]™[\ÈH\ÙH\ÈÛY[™YXÝË[™HÝY\Ý	ÜÂˆËËÈÜÝ›ÜØ\™È]ÈÛÝ[™ˆ]™\žHØ][›ÛˆÝ]HÛÝ[Ë›ÝÛ›HBˆËËÈÛÜœÈ[™]™\œË‚ˆÖÝ\ÝBˆ›ˆ]™\žWØØ][›Û—ØÛÝ[×Ø\×Ú[™ÛÜ\˜]Y

HÂˆ›Üˆ˜[YH[ˆÂˆ›Z[™XÜ˜Y˜Ø][›Ûˆ‹ˆ›Z[™XÜ˜YØ]\—ØØ][›Ûˆ‹ˆ›Z[™XÜ˜Y›]˜WØØ][›Ûˆ‹ˆ›Z[™XÜ˜YœÝÙ\—ÜÛ›Ý×ØØ][›Ûˆ‹ˆHÂˆ\ÜÙ\JÝ\\ŽŽš[™ÛÜ\˜]Y
˜[YJKžÛ˜[Y_HŠNÂˆBˆ\ÜÙ\J\Ý\\ŽŽš[™ÛÜ\˜]Y
›Z[™XÜ˜YØ]\ˆŠJNÂˆB‚ˆËËÈ[ˆ”ÈYÚ[ˆ\ÝÈH”ËÜ]ÛœÈ][™ZÙ\ÈH[žHÙ™ˆH\Ý‚ˆËËÈH[]HÙY\ÈHÚÚ[ˆ]›Ý[™ÈH^Y\ˆÜ]Û™Y™Y›Ü™H]È[žBˆËËÈ\œš]™\ÈZÙ\ÈH[žHÛ˜ÙH]ÛÛY\Ë‚ˆÖÝ\ÝBˆ›ˆWÜ^Y\—Ù[]WÚÙY\×ÝWÜÚÚ[—Ú]×Ù[žWÙØ]™J
HÂˆ\ÙHÜš^›Û—ØÛY[ŽœÚÚ[œÎŽ•^\™\ÎÂˆ\ÙHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚX\Âˆ]œÈH^\™\ÈÂˆÚÚ[ŽˆÛÛYJš‹ËÝ^\™\Ë›Z[™XÜ˜Y›™]Ý^\™KÛœÈ‹š[Ê
JKˆÚYÛ™YˆYKˆ‹•^\™\ÎŽ™Y˜][

BˆNÂˆ]
]]\ÝY]]Ù\
HH
\ÚX\Ž™œ›ÛJÊÝLLŽœË˜ÛÛ™J
JWJK\ÚX\Ž›™]Ê
JNÂˆ]]]^Y\œÈH\ÚX\Ž™œ›ÛJÊšLÌ‹ÝLLŽ
K
Ë
WJNÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽœÚÚ[—Ù[šY\Ê	œ^Y\œË	›\ÝY	›]]Ù\
KˆÊËœË˜ÛÛ™J
JWBˆ
NÂˆ\ÝYœ™[[Ý™J	ÊNÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽœÚÚ[—Ù[šY\Ê	œ^Y\œË	›\ÝY	›]]Ù\
KˆÊËœË˜ÛÛ™J
JWBˆ
NÂˆËÈ^Y\ˆ	ÜÈ[žHÛÛY\ÈY\ˆ]È[]K‚ˆ]]HH^\™\ÈÂˆÚÚ[ŽˆÛÛYJš‹ËÝ^\™\Ë›Z[™XÜ˜Y›™]Ý^\™KÛ]H‹š[Ê
JKˆ‹›œË˜ÛÛ™J
BˆNÂˆ\ÝYš[œÙ\
]K˜ÛÛ™J
JNÂˆ]]][šY\ÈHÝ\\ŽŽœÚÚ[—Ù[šY\Ê	œ^Y\œË	›\ÝY	›]]Ù\
NÂˆ[šY\ËœÛÜØžWÚÙ^J
KÊ_
JNÂˆ\ÜÙ\Ù\HJ[šY\ËÊËœÊK
]JWJNÂˆËÈH™[[Ý™Y[]H›Ü™Ù]È]È›Ùš[HÚ]]‚ˆ^Y\œËœ™[[Ý™J	ŒŠNÂˆÙ\œ™[[Ý™J	ŒŠNÂˆ\ÝY˜ÛX\Š
NÂˆ\ÜÙ\Ù\HJÝ\\ŽŽœÚÚ[—Ù[šY\Ê	œ^Y\œË	›\ÝY	›]]Ù\
K›[Š
KJNÂˆB‚ˆÖÝ\ÝBˆ›ˆX—ÜX›XØ][Û—Ùš[™Ù\œš[Ý˜XÚÜ×Ø[Ú[œ]Ê
HÂˆ]š[™Ù\œš[HYÎˆ	–ÝLLŽKÚÚ[œË]\ßÂˆ
ˆYË›[Š
KˆYËš]\Š
K™›Û
Ü‹]ZYÜˆˆ]ZY
KˆÚÚ[œËˆ]\Ëˆ
BˆNÂˆ]X›\ÚYHš[™Ù\œš[
	–ÍËKKŠNÂˆ\ÜÙ\Ù\HJX›\ÚYš[™Ù\œš[
	–Î×KKŠJNÂˆ\ÜÙ\Û™HJX›\ÚYš[™Ù\œš[
	–ÍËWKKŠJNÂˆ\ÜÙ\Û™HJX›\ÚYš[™Ù\œš[
	–ÍËK‹ŠJNÂˆ\ÜÙ\Û™HJX›\ÚYš[™Ù\œš[
	–ÍËKKÊJNÂˆB‚ˆÖÝ\ÝBˆ›ˆX—ÛØš™XÝÜÚÚ[—ÜX›XØ][Û—Ý˜XÚÜ×ØÚ[™Ù\×Ø[™Ø]\×ØÛX\Š
HÂˆ]]]X›\ÚYHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚX\Ž›™]Ê
NÂˆ]]]Üš]\ÈHÂˆ›ÜˆÚÚ[ˆ[ˆÈœÚÚ[œËÙš\œÝ‹œÚÚ[œËÙš\œÝ‹œÚÚ[œËÜÙXÛÛ™—HÂˆ›ÜˆÙ^H[ˆÈšYÈ‹›˜[YNœ^Y\ˆ—HÂˆYˆÝ\\ŽŽ›Øš™XÝÜÚÚ[—ØÚ[™ÙY
	œX›\ÚYÙ^KÚÚ[ŠHÂˆX›\ÚYš[œÙ\
Ù^K×ÛÝÛ™Y

KÚÚ[‹×ÛÝÛ™Y

JNÂˆÜš]\È
ÏHNÂˆBˆBˆBˆ\ÜÙ\Ù\HJÜš]\Ë[˜Ú[™ÙYÚÚ[œÈÈ›ÝX›\ÚZ]\ˆ[žHŠNÂˆX›\ÚY˜ÛX\Š
NÂˆ\ÜÙ\JÝ\\ŽŽ›Øš™XÝÜÚÚ[—ØÚ[™ÙY
ˆ	œX›\ÚYˆšYÈ‹ˆœÚÚ[œËÜÙXÛÛ™‚ˆ
JNÂˆ\ÜÙ\JÝ\\ŽŽ›Øš™XÝÜÚÚ[—ØÚ[™ÙY
ˆ	œX›\ÚYˆ›˜[YNœ^Y\ˆ‹ˆœÚÚ[œËÜÙXÛÛ™‚ˆ
JNÂˆB‚ˆËËÈ[ˆ”ÈYÚ[‰ÜÈØØ[NˆH˜\ÙHHYÚ[ˆÙ]ËÜˆH[ÙYšY\ˆÛˆK‚ˆÖÝ\ÝBˆ›ˆ]šX]WÝ˜[Y\×ØYÝ[—Û][\WØ\×Ý˜[š[J
HÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽœ^NŽžÐ]šX]S[ÙYšY\‹]šX]TÛ˜\ÚÝNÂˆ][ÙYšY\ˆH[[Ý[Ü\˜][ÛŸ]šX]S[ÙYšY\ˆÂˆYˆ›Z[™XÜ˜Yœ›ÛÙˆ‹š[Ê
Kˆ[[Ý[ˆÜ\˜][Û‹ˆNÂˆ]Û˜\ÚÝH˜\ÙK[ÙYšY\œß]šX]TÛ˜\ÚÝÂˆ]šX]Nˆˆ˜\ÙKˆ[ÙYšY\œËˆNÂˆ\ÜÙ\Ù\HJÝ\\ŽŽœÛ˜\ÚÝÝ˜[YJ	œÛ˜\ÚÝ
‹Œ™XÈV×JJK‹Œ
NÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽœÛ˜\ÚÝÝ˜[YJ	œÛ˜\ÚÝ
KŒ™XÈVÛ[ÙYšY\ŠKŒ
WJJKˆ‹Œˆ
NÂˆËÈ
H
ÈJH
ˆ
H
ÈJHœ›ÛHH˜\ÙK[ˆ
ˆ
H
ÈJHÙˆHÝ[‚ˆ][H™XÈVÛ[ÙYšY\ŠKŒ
K[ÙYšY\ŠKJK[ÙYšY\ŠKŒŠWNÂˆ\ÜÙ\Ù\HJÝ\\ŽŽœÛ˜\ÚÝÝ˜[YJ	œÛ˜\ÚÝ
KŒ[
JK‹Œ
NÂˆB‚ˆ\ÙHÝ\\ŽŽ‘Ø[YNÂ‚ˆËËÈ[]K›ÛÚÐ]ˆX]È˜XÙ\È
Ö‹NL˜XÙ\È
ÖÈ]ÚNLÛÚÜÈ\‚ˆÖÝ\ÝBˆ›ˆÛÚ×Ø]Ý\›œ×ÝÝØ\™×ÝWÝ\™Ù]

HÂˆ]ÛÜÙHH
X]Ë]Ú
Nˆ
ŒÌ‹ŒÌŠKØ[ˆ
ŒÌ‹ŒÌŠ_Âˆ\ÜÙ\Jˆ
X]ÈHØ[Œ
K˜XœÊ
HYKLÈ	‰ˆ
]ÚHØ[ŒJK˜XœÊ
HYKLËˆžÞX]ßHÜ]ÚKØ[ÝØ[ßH‚ˆ
NÂˆNÂˆ]]HÌLŒŒMKŒNÂˆ]Ù™ˆHˆÙÈ×_Ø]ÌH
ÈÌK]ÌWH
ÈÌWK]Ì—H
ÈÌ—WNÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠÌŒŒËŒJJKˆ
ŒŒ
Kˆ
NÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠÌËŒŒŒJJKˆ
NLŒŒ
Kˆ
NÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠËLËŒŒŒJJKˆ
LŒŒ
Kˆ
NÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠÌŒŒLËŒJJKˆ
LNŒŒ
Kˆ
NÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠÌŒ‹Œ‹ŒJJKˆ
ŒMKŒ
Kˆ
NÂˆÛÜÙJˆÝ\\ŽŽ›ÛÚ×Ø]Ü›Ý][ÛŠ]Ù™ŠÌŒMŒŒJJKˆ
ŒKŒ
Kˆ
NÂˆB‚ˆËËÈ[™S[Ý™T^Y\˜[Ý™\ÈH^Y\‰ÜÈ]Ú[™H[Ù[	ÜÈÛ]ÚˆËËÈ
›ÝØ
H[ZÙNˆH™[]]™HÚ[™ÙHÙY\ÈHX\ÙHHXÚÈ™YØ[‹XXÚˆËËÈ[™Û[\YÛˆ]ÈÝÛ‹[™[ˆXœÛÛ]HÛ™HÛ˜\È›Ý
ÌÎ
K‚ˆÖÝ\ÝBˆ›ˆWÜÜÚ][Û—ÜXÚÙ]Û[Ý™\×Ø›ÝÜ]ÚÙ[™Ê
HÂˆ\ÙHÝ\\ŽŽœÜÚ][Û—ÜXÚÙ]Ü]Ú\È]ÚÂˆËÈHXÚÈ\È^Z[™ÈHXYÝÛˆœ›ÛHÌÈÈHXÚÙ]\›œÈL‚ˆ\ÜÙ\Ù\HJˆ
]Ú
ÌŒLŒYJK]Ú
ŒLŒYJJKˆ
ŒLŒ
Bˆ
NÂˆËÈXXÚ[™Û[\ÈÛˆ]ÈÝÛ‹‚ˆ\ÜÙ\Ù\HJˆ
]Ú
ŒŒŒYJK]Ú
ŒŒŒŒYJJKˆ
LŒŒ
Bˆ
NÂˆËÈ[ˆXœÛÛ]H]ÚYÛ›Ü™\È›Ý[™Ë‚ˆ\ÜÙ\Ù\HJˆ
]Ú
ÌŒMKŒ˜[ÙJK]Ú
ŒMKŒ˜[ÙJJKˆ
MKŒMKŒ
Bˆ
NÂˆ\ÜÙ\Ù\HJ]Ú
ŒLŒŒ˜[ÙJKLŒ
NÂˆB‚ˆËËÈH\ÛÝ[™ÙˆÛY[›Ý[™[XYÙQ]™[XÚÙ]›ÛÝÜÈ]È[XYÙBˆËËÈ\H
^Y\‹™Ù]\ÛÝ[™ˆH\IÜÈY™™XÝÊKÚ]HÛÙXÉÜÂˆËËÈ\Y˜][›Üˆ[ˆ[šÛ›ÝÛˆ\H
ÍMJK‚ˆÖÝ\ÝBˆ›ˆWÚ\ÜÛÝ[™Ù›ÛÝÜ×ÝWÙ[XYÙWÙ]™[×Ý\J
HÂˆ\ÙHÝ\\ŽŽš\ÜÛÝ[™Ù›ÜŽÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽœ™YÚ\ÝžNŽ‘[˜[ZXÎÂˆ\ÙHÜš^›Û—ÝÛÜ›Ž™[XYÙWÜÛÝ\˜ÙNŽØ][ÙÎÂˆ]\\ÈHÂˆ
›Z[™XÜ˜Yš[—Ùš\™H‹˜\›š[™ÈŠKˆ
›Z[™XÜ˜Y™›ÝÛˆ‹™›ÝÛš[™ÈŠKˆ
›Z[™XÜ˜YœÝÙY]Ø™\œžWØ\Ú‹œÚÚ[™ÈŠKˆ
›Z[™XÜ˜Y™œ™Y^™H‹™œ™Y^š[™ÈŠKˆ
›Z[™XÜ˜Y˜\œ›ÝÈ‹š\ŠKˆNÂˆËÈH\\ÈHÙ\™\ˆÞ[˜Ú›Ûš^™Y[ˆ™YÚ\ÝžHÜ™\‹‚ˆ]]]™YÚ\ÝšY\ÈH[˜[ZXÎŽ™Y˜][

NÂˆ™YÚ\ÝšY\Ë˜Y
Üš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ]HÂˆ™YÚ\ÝžNˆ›Z[™XÜ˜Y™[XYÙWÝ\H‹š[Ê
Kˆ[šY\Îˆ\\Âˆš]\Š
Bˆ›X\
ˆ
YÊ_Üš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ[žHÂˆYˆ

šY
Kš[Ê
Kˆ]Nˆ›Û™KˆKˆ
Bˆ˜ÛÛXÝ

KˆJNÂˆËÈH[XYÙHØ][ÙÎˆÛ™H[žH\ˆ\HX›Ý™KXXÚÚ]]ÈY™™XÝË‚ˆ]š]ÈHŽˆŒÌŸ›Ü›X]JžÎŒH‹‹×Øš]Ê
JNÂˆ]Ø][ÙÈHØ][ÙÎŽ™œ›ÛWÚœÛÛŠˆ	œÙ\™WÚœÛÛŽŽšœÛÛˆJÂˆ™]WÝ™\œÚ[ÛˆŽˆMŽ™œ›ÛJÜš^›Û—ÝÛÜ›Ž‘UWÕ‘T”ÒSÓŠKˆ\\ÈŽˆ\\Ëš]\Š
K›X\

YY™™XÝÊ_Ù\™WÚœÛÛŽŽšœÛÛˆJÂˆšYŽˆYˆ›Y\ÜØYÙWÚYŽˆYš[WÜÝ\ÛX]Ú\Ê›Z[™XÜ˜YˆŠKˆ™^]\Ý[ÛˆŽˆš]ÊŒJKˆœØØ[[™ÈŽˆÚ[—ØØ]\ÙYØžWÛ]š[™×Û›Û—Ü^Y\ˆ‹ˆ™Y™™XÝÈŽˆY™™XÝËˆ™X]ÛY\ÜØYÙWÝ\HŽˆ™Y˜][‹ˆJJK˜ÛÛXÝŽ™XÏÏŠ
KˆJKˆ	œÙ\™WÚœÛÛŽŽšœÛÛˆJßJKˆ
Bˆ[Ü˜\

NÂˆ›Üˆ
Y
˜[YKY™™XÝÊJH[ˆ\\Ëš]\Š
K™[[Y\˜]J
HÂˆ\ÜÙ\Ù\HJˆ\ÜÛÝ[™Ù›ÜŠ	œ™YÚ\ÝšY\Ë	˜Ø][ÙËY\ÈLÌŠKˆÜš^›Û—ØÛY[ŽšX[Žš\ÜÛÝ[™
Y™™XÝÊKˆžÛ˜[Y_H‚ˆ
NÂˆBˆËÈ[ˆ[šÛ›ÝÛˆY˜[È˜XÚÈÈHÛÙXÉÜÈY˜][Y™™XÝË‚ˆ\ÜÙ\Ù\HJˆ\ÜÛÝ[™Ù›ÜŠ	œ™YÚ\ÝšY\Ë	˜Ø][ÙË\\Ë›[Š
H\ÈLÌŠKˆ›Z[™XÜ˜Y™[]Kœ^Y\‹š\‚ˆ
NÂˆËÈHÙ\™\‹YYš[™Y[XYÙH\HÚ][›[™H•Y™™XÝÈ™YYÈ›È[žH[‚ˆËÈHÛY[	ÜÈZ[Z[ˆØ][ÙË‚ˆ]]]Ý\ÝÛHHÜš^›Û—ØÛÜ™NŽ›˜ŽÛÛ\Ý[™Ž›™]Ê
NÂˆÝ\ÝÛKš[œÙ\
ˆ™Y™™XÝÈ‹š[Ê
KˆÜš^›Û—ØÛÜ™NŽ›˜Ž•YÎŽ”Ýš[™Ê˜\›š[™È‹š[Ê
JKˆ
NÂˆ™YÚ\ÝšY\Ë˜Y
Üš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ]HÂˆ™YÚ\ÝžNˆ›Z[™XÜ˜Y™[XYÙWÝ\H‹š[Ê
Kˆ[šY\Îˆ™XÈVÚÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ[žHÂˆYˆ˜Ý\ÝÛN›]˜WÜÜ\Ú‹š[Ê
Kˆ]NˆÛÛYJÜš^›Û—ØÛÜ™NŽ›˜Ž•YÎŽÛÛ\Ý[™
Ý\ÝÛJJKˆWKˆJNÂˆ\ÜÙ\Ù\HJˆ\ÜÛÝ[™Ù›ÜŠ	œ™YÚ\ÝšY\Ë	˜Ø][ÙË
Kˆ›Z[™XÜ˜Y™[]Kœ^Y\‹š\ÛÛ—Ùš\™H‚ˆ
NÂˆBˆÖÝ\ÝBˆ›ˆÞ[˜Ú›Ûš^™YÙ[XYÙWÙY˜][×ÛÝ™\œšYWØWÙY™™\™[ÛØØ[ÙYš[š][ÛŠ
HÂˆ]ØØ[HÜš^›Û—ÝÛÜ›Ž™[XYÙWÜÛÝ\˜ÙNŽØ][ÙÎŽ™œ›ÛWÚœÛÛŠ	œÙ\™WÚœÛÛŽŽšœÛÛˆJÂˆ™]WÝ™\œÚ[ÛˆŽˆÜš^›Û—ÝÛÜ›Ž‘UWÕ‘T”ÒSÓ‹ˆ\\ÈŽˆÞÈšYŽˆ™[[ÎšX]‹›Y\ÜØYÙWÚYŽˆšX]‹™^]\Ý[ÛˆŽˆŒ‹œØØ[[™ÈŽˆ›™]™\ˆ‹™Y™™XÝÈŽˆ˜\›š[™È‹™X]ÛY\ÜØYÙWÝ\HŽˆ™Y˜][ŸWBˆJK	œÙ\™WÚœÛÛŽŽšœÛÛˆJßJJK[Ü˜\

NÂˆ]]]™YÚ\ÝšY\ÈHÜš^›Û—Ü›ÝØÛÛŽœ™YÚ\ÝžNŽ‘[˜[ZXÎŽ™Y˜][

NÂˆ™YÚ\ÝšY\Ë˜Y
Üš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ]HÂˆ™YÚ\ÝžNˆ›Z[™XÜ˜Y™[XYÙWÝ\H‹š[Ê
Kˆ[šY\Îˆ™XÈVÚÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽ˜ÛÛ™šYÝ\˜][ÛŽŽ”™YÚ\ÝžQ[žHÂˆYˆ™[[ÎšX]‹š[Ê
Kˆ]NˆÛÛYJÜš^›Û—ØÛÜ™NŽ›˜Ž•YÎŽÛÛ\Ý[™
Y˜][Ž™Y˜][

JJKˆWKˆJNÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽš\ÜÛÝ[™Ù›ÜŠ	œ™YÚ\ÝšY\Ë	›ØØ[
Kˆ›Z[™XÜ˜Y™[]Kœ^Y\‹š\‚ˆ
NÂˆBˆËËÈÛY[XÚÙ]\Ý[™\‹š[™TÙ][]T\ÜÙ[™Ù\œÔXÚÙ]ˆH™ZXÛBˆËËÈZ™XÝÈ]È\ÜÙ[™Ù\œË[ˆXXÚ\ÝY[]HÝ\šY[™ØÈ]ÛÂˆËËÈ\Õ™ZXÛXÛÈÚ[HH\Ý\È›Ý[\KÚ]]™\ˆH\ÜÙ[™Ù\‰ÜÂˆËËÈ\NˆHÙ\™\‰ÜÈ][H\Ü^HšY[™ÈH^Y\ˆY\È]^Y\‰ÜÈ˜[YBˆËËÈYÈZÙHH^\Ü^HÙ\È
ÍMŠK‚ˆÖÝ\ÝBˆ›ˆ[žWÝ˜XÚÙYÜ\ÜÙ[™Ù\—ÛXZÙ\×Ú]×Ý™ZXÛWØWÝ™ZXÛJ
HÂˆ\ÙHÝŽ˜ÛÛXÝ[ÛœÎŽžÒ\ÚX\\ÚÙ]NÂˆ]˜XÚÙYHÚ[™ˆ	œÝŸÝ\\ŽŽ•˜XÚÙYÂˆÚ[™ˆÚ[™×ÛÝÛ™Y

KˆÜÚ][ÛŽˆÌŒŒŒKˆÛÙX×Ø˜\ÙNˆÌŒŒŒKˆÚ[[ˆ˜[ÙKˆ™ZXÛNˆ›Û™KˆNÂˆ]]][]Y\ÈH\ÚX\Ž™œ›ÛJÂˆ
‹˜XÚÙY
›Z[™XÜ˜Yœ^Y\ˆŠJKˆ
Ë˜XÚÙY
›Z[™XÜ˜Yœ^Y\ˆŠJKˆ
K˜XÚÙY
›Z[™XÜ˜Yš][WÙ\Ü^HŠJKˆ
L˜XÚÙY
›Z[™XÜ˜Y˜›ØÚ×Ù\Ü^HŠJKˆJNÂˆ]™ZXÛ\ÈHNˆ	’\ÚX\LÌ‹Ý\\ŽŽ•˜XÚÙYŸOˆ\ÚÙ]LÌˆÂˆK˜[Y\Ê
K™š[\—ÛX\
™ZXÛJK˜ÛÛXÝ

BˆNÂˆÝ\\ŽŽœÙX]Ü\ÜÙ[™Ù\œÊ	›]][]Y\Ë‹	–ÎKLJNÂˆ\ÜÙ\Ù\HJ™ZXÛ\Ê	™[]Y\ÊK\ÚÙ]Ž™œ›ÛJÍ—JJNÂˆËÈH\Ý™\XÙ\ÈHÛÛ™NˆLÛÝÙ™‹HÝ[šY\Ë‚ˆÝ\\ŽŽœÙX]Ü\ÜÙ[™Ù\œÊ	›]][]Y\Ë‹	–ÎWJNÂˆ\ÜÙ\Ù\HJ[]Y\ÖÉŒLK™ZXÛK›Û™JNÂˆ\ÜÙ\Ù\HJ[]Y\ÖÉŽWK™ZXÛKÛÛYJŠJNÂˆËÈÝ\šY[™ÈX]™\ÈHÛ™ZXÛHš\œÝ‚ˆÝ\\ŽŽœÙX]Ü\ÜÙ[™Ù\œÊ	›]][]Y\ËË	–ÎWJNÂˆ\ÜÙ\Ù\HJ™ZXÛ\Ê	™[]Y\ÊK\ÚÙ]Ž™œ›ÛJÍ×JJNÂˆËÈ[ˆ[\H\Ýˆ›Ø›ÙHšY\ÈÈ[žH[Ü™K‚ˆÝ\\ŽŽœÙX]Ü\ÜÙ[™Ù\œÊ	›]][]Y\ËË	–×JNÂˆ\ÜÙ\J™ZXÛ\Ê	™[]Y\ÊKš\×Ù[\J
JNÂˆBˆËËÈÛY[›Ý[™ÛÝ[™[]TXÚÙ]ˆHÛÝ[™\ÈX\™Ú\™HH[]H]ˆËËÈ˜[Y\È\È8 %HØØ[^Y\ˆ[˜ÛYYÚXÚ\È›È˜XÚÙY[]H8 %[™ˆËËÈ[ˆY\È]™[Ù\Û‰Ý]™H^\È›Ý[™È][‚ˆÖÝ\ÝBˆ›ˆWÜÛÝ[™Ù[]WÚ\×ÚX\™ÝÚ\™WÚ]×Ù[]WÚ\Ê
HÂˆ\ÙHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚX\Âˆ]]]˜XÚÙYH\ÚX\Ž›™]Ê
NÂˆ˜XÚÙYš[œÙ\
ˆËˆÝ\\ŽŽ•˜XÚÙYÂˆÚ[™ˆ›Z[™XÜ˜Y˜ÛÝÈ‹×ÛÝÛ™Y

KˆÜÚ][ÛŽˆÎKŒLËWKˆÛÙX×Ø˜\ÙNˆÎKŒLËWKˆÚ[[ˆ˜[ÙKˆ™ZXÛNˆ›Û™KˆKˆ
NÂˆ]^Y\ˆHÛÛYJ
‹ÌKÌŒWK˜[ÙJJNÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]J‹^Y\‹›Û™K	˜XÚÙY›Û™JBˆ™š[\Šß\ËœÚ[[
Bˆ›X\
ßËœÜÚ][ÛŠKˆÛÛYJÌKÌŒWJKˆHØØ[^Y\ˆ\È˜[YYžH™[[ÝK™[]WÚY‚ˆ
NÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JË^Y\‹›Û™K	˜XÚÙY›Û™JBˆ™š[\Šß\ËœÚ[[
Bˆ›X\
ßËœÜÚ][ÛŠKˆÛÛYJÎKŒLËWJKˆ˜[›Ý\ˆ[]HHÙ\™\ˆÜ]Û™Y‚ˆ
NÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JK^Y\‹›Û™K	˜XÚÙY›Û™JBˆ™š[\Šß\ËœÚ[[
Bˆ›X\
ßËœÜÚ][ÛŠKˆ›Û™Kˆ™Ù][]H\È[ˆ[™TÛÝ[™[]Q]™[™]\›œÈ‚ˆ
NÂˆ\ÜÙ\Ù\HJˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]J‹›Û™K›Û™K	˜XÚÙY›Û™JBˆ™š[\Šß\ËœÚ[[
Bˆ›X\
ßËœÜÚ][ÛŠKˆ›Û™Kˆ˜H^Y\ˆÚ]›ÈÜÚ][ÛˆY]‚ˆ
NÂˆBˆÖÝ\ÝBˆ›ˆWÜÛÝ[™Ù[]WÜ™]Z[œ×ÜÚ[[˜ÙWÙ›Ü—Ü^Y\œ×Ø[™Û›Û—Û[Ø—Ý\™Ù]Ê
HÂˆ\ÙHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚX\Âˆ]]]˜XÚÙYH\ÚX\Ž›™]Ê
NÂˆ˜XÚÙYš[œÙ\
ˆËˆÝ\\ŽŽ•˜XÚÙYÂˆÚ[™ˆ›Z[™XÜ˜Y˜\œ›ÝÈ‹×ÛÝÛ™Y

KˆÜÚ][ÛŽˆÎKŒLËWKˆÛÙX×Ø˜\ÙNˆÎKŒLËWKˆÚ[[ˆYKˆ™ZXÛNˆ›Û™KˆKˆ
NÂˆ]Ý]HHÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JË›Û™K›Û™K	˜XÚÙY›Û™JK[Ü˜\

NÂˆ\ÜÙ\JÝ]KœÚ[[
NÂˆ\ÜÙ\J\Ý]Kœ™[[Ý™Y
NÂˆ\ÜÙ\Ù\HJÝ]KœÜÚ][Û‹ÎKŒLËWJNÂˆ]^Y\ˆBˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]J‹ÛÛYJ
‹ÌKŒ‹ŒËŒKYJJK›Û™K	˜XÚÙY›Û™JBˆ[Ü˜\

NÂˆ\ÜÙ\J^Y\‹œÚ[[
NÂˆ\ÜÙ\JÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JK›Û™K›Û™K	˜XÚÙY›Û™JKš\×Û›Û™J
JNÂˆBˆÖÝ\ÝBˆ›ˆWÙ˜YÛÛ—ÜÛÝ[™Ý\Ù\×Û]™WÛ[Ý™[Y[ÝÚ]Ý]ÛÜÚ[™×Û[Ø—ÛY]Y]J
HÂˆ\ÙHÜš^›Û—ØÛY[ŽœÛÝ[™Ù[™Ú[™NŽ”ÛÝ[™[]TÝ]NÂˆ]Ý[HHÛÝ[™[]TÝ]HÂˆÜÚ][ÛŽˆÌŒŒŒKˆÚ[[ˆYKˆ™[[Ý™Yˆ˜[ÙKˆNÂˆ]˜XÚÙYHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚX\Ž›™]Ê
NÂˆ›ÜˆÜÚ][Ûˆ[ˆÖÌŒŒLM‹ŒKÌKKKŒLNŒWHÂˆ]Ý]HBˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JË›Û™KÛÛYJÝ[JK	˜XÚÙYÛÛYJ
ËÜÚ][ÛŠJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJÝ]KœÜÚ][Û‹ÜÚ][ÛŠNÂˆ\ÜÙ\JÝ]KœÚ[[
NÂˆ\ÜÙ\J\Ý]Kœ™[[Ý™Y
NÂˆBˆ][œ™[]YBˆÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JË›Û™KÛÛYJÝ[JK	˜XÚÙYÛÛYJ
ÎNKŒÈ×JJJBˆ[Ü˜\

NÂˆ\ÜÙ\Ù\HJ[œ™[]YœÜÚ][Û‹Ý[KœÜÚ][ÛŠNÂˆ\ÜÙ\JÝ\\ŽŽ™[]WÜÛÝ[™ÜÝ]JË›Û™K›Û™K	˜XÚÙYÛÛYJ
ËÎNKŒÈ×JJJKš\×Û›Û™J
JNÂˆB‚ˆËËÈHÝ™\ˆHÛÝÙˆH[™[ÜžHÛˆHÙ\™\ˆ
ÌÌÍMÊNˆH™YXÝY“ÕÂˆËËÈ›ÜÈHÝXÚËÛÈH^Y\‰ÜÈ[™ÝÚ[™ÜË\È]š[™Ñ[]K™›ÜˆËËÈÙ\È›Üˆ™YXÝ[Û‹”‘QPÕQÈHZ[ˆXÚÈ\›ÜÈ›Ý[™Ë‚ˆÖÝ\ÝBˆ›ˆWÜ™YXÝYÝ›Ý×Ùœ›ÛWØWÜÛÝÜÝÚ[™Ü×ÝWÚ[™

HÂˆ\ÙHÜš^›Û—ØÛY[ŽžÂˆ˜[›™\ŽŽ˜[›™\œËˆœ™]Ú[™ÎŽœ™]Ú[™ËˆÜ˜Y[™ÎŽ”™XÚ\\ËˆÜ™X]]™NŽ“˜[Y\Ëˆ[˜Ú[Ž‘[˜Ú[Y[Ëˆ][\ÎŽžÒ][TÝXÚË][\ßKˆY[NŽžÑ[‹[œ]Ú[™Y[_KˆÝ][ÛœÎŽ”Ý][Û”™XÚ\\ËˆNÂˆ]›ÛÝHÝŽœ]Ž”]Ž›™]Ê[ˆJÐT‘Ó×ÓPS’Q‘TÕÑTˆŠJBˆš›Ú[Š‹‹‹Ë‹‹Ë›ØØ[ÚÜš^›Û‹XÛY[ŠBˆš›Ú[ŠÜš^›Û—ØÛÜ™NŽ•S’SWÕ‘T”ÒSÓŠNÂˆYˆ\›ÛÝš›Ú[Šš][\ËšœÛÛˆŠK™^\ÝÊ
HÂˆ™]\›ŽÂˆBˆ]›Û™HH	šÜš^›Û—ÝÛÜ›Ž›ÛÝŽ”XÚÓÝ™\›^NŽ““Ó‘NÂˆ]Y˜][ÈHÜš^›Û—ÝÛÜ›Žš][WÙY˜][ÎŽœ™XY
	œ›ÛÝ
K[Ü˜\

NÂˆ]][\ÈH][\ÎŽ˜Ø][ÙÊ	œ›ÛÝ›Û™K	™Y˜][ÊK[Ü˜\

NÂˆ]™XÚ\\ÈH™XÚ\\ÎŽ›ØY
	œ›ÛÝ	š][\Ë›Û™K	˜Ü˜]NŽ™Ø[YNŽ\ÝÜÙ\™\—ÛØY

JK[Ü˜\

NÂˆ]œ™]Ú[™ÈHœ™]Ú[™ÎŽ›ØY
	œ›ÛÝ	š][\Ë›Û™JNÂˆ]˜[›™\œÈH˜[›™\œÎŽ›ØY
	œ›ÛÝ	™Y˜][ÊNÂˆ]Ý][Û—Ü™XÚ\\ÈBˆÝ][Û”™XÚ\\ÎŽ›ØY
	œ›ÛÝ	š][\Ë›Û™K	˜Ü˜]NŽ™Ø[YNŽ\ÝÜÙ\™\—ÛØY

JNÂˆ][˜Ú[Y[ÈH[˜Ú[Y[ÎŽ™Y˜][

NÂˆ]˜[Y\ÈH˜[Y\ÎŽ™Y˜][

NÂˆ][ˆH[ˆÂˆÜ˜Y[™×Ü™[XZ[™\Žˆ›Û™Kˆ^Y\Žˆ›Û™Kˆ^Y\—Ù[]WÚYˆ›Û™KˆÛÜØÛÛ^ˆ	˜Ü˜]NŽ™Ø[YNŽ\ÝÜÙ\™\—ÛØY

Bˆ›ÛÜØÛÛ^
Üš^›Û—ÝÛÜ›Ž›˜]]™WÛØYØÛÛ^Ž“ÛÜÚYNŽÛY[×JKˆ][\Îˆ	š][\Ëˆ™XÚ\\Îˆ	œ™XÚ\\Ëˆœ™]Ú[™Îˆ	˜œ™]Ú[™Ëˆ˜[›™\œÎˆ	˜˜[›™\œËˆÝ][Û—Ü™XÚ\\Îˆ	œÝ][Û—Ü™XÚ\\Ëˆ[˜Ú[Y[Îˆ	™[˜Ú[Y[Ëˆ˜[Y\Îˆ	›˜[Y\Ëˆ˜YNˆ›Û™KˆX\Îˆ›Û™Kˆ[Z]YØÜ˜Y[™Îˆ˜[ÙKˆÛ›ÝÛ—Ü™XÚ\\Îˆ›Û™KˆNÂˆ]]]Y[HHY[NŽ›™]ÊÚ[™Ž’[™[ÜžJNÂˆ]ÛØ˜›\ÝÛ™HH][TÝXÚÈÂˆ][Nˆ›Z[™XÜ˜Y˜ÛØ˜›\ÝÛ™H‹š[Ê
KˆÛÝ[ˆM‹ˆ[XYÙNˆˆ^˜Nˆ›Û™KˆNÂˆËÈHš\œÝÝ˜\ˆÛÝÙˆ[™[ÜžSY[K‚ˆY[KœÙ]ÜÛÝ
	™[‹Í‹ÛÛYJÛØ˜›\ÝÛ™JJNÂˆ]
Ù[™]ÊHBˆÝ\\ŽŽœ™YXÝØÛXÚÜÊ	›]]Y[K	™[‹	–ÊÍ‹[œ]Ž•›ÝÊWJK[Ü˜\

NÂˆ\ÜÙ\J™]Ë”HÝ™\ˆHÛÝ›ÜÈÛ™H[™ÝÚ[™ÜÈŠNÂˆ\ÜÙ\Ù\HJY[KœÛÝ
ÍŠK›X\
ßË˜ÛÝ[
KÛÛYJMJJNÂˆ\ÜÙ\Ù\HJÙ[›[Š
KJNÂˆ]
Ë™]ÊHHÝ\\ŽŽœ™YXÝØÛXÚÜÊ	›]]Y[K	™[‹	–ÊÍ‹[œ]Ž”XÚÝ\
WJK[Ü˜\

NÂˆ\ÜÙ\J]™]ËœXÚÚ[™ÈHÝXÚÈ\›ÜÈ›Ý[™ÈŠNÂˆ]
Ë™]ÊHBˆÝ\\ŽŽœ™YXÝØÛXÚÜÊ	›]]Y[K	™[‹	–ÊNNNKK[œ]Ž”XÚÝ\
WJK[Ü˜\

NÂˆ\ÜÙ\J™]Ë›Û™HÙˆHØ\œšYYÝXÚÈÝ]ÚYHHÚ[™ÝÈŠNÂˆB‚ˆÖÝ\ÝBˆ›ˆœ™YWÙ\™XÝ[Û—ÜÙ[™×ØWØœ™XZ×Ø™Y›Ü™WÝWÛ[Ý™WÝ]ÙÛÙ\×Ú[×Ú]×ÜÜXÙJ
HÂˆ\ÙHÝ\\ŽŽžÑ[\KË^KØZ]×Ù›Ü—ÝWÝšY]ßNÂˆ]XÝ[ÛˆHXÝ[ÛŸÂˆÎŽ”^Y\XÝ[ÛŠ^NŽ”^Y\XÝ[ÛˆÂˆXÝ[Û‹ˆÜÎˆÌŒ‹Kˆ\™XÝ[ÛŽˆˆÙ\]Y[˜ÙNˆKˆJBˆNÂˆËÈÝ\Ú[™ÙH\™XÝ[Û‹X›Ü[™ÝÜÛÈ]Û˜ÙHšYÚY\ˆHÝY\™Y[Ý™K‚ˆ›ÜˆYÈ[ˆ‹LÈÂˆ\ÜÙ\Jˆ]ØZ]×Ù›Ü—ÝWÝšY]Ê	˜XÝ[ÛŠYÊKYK˜[ÙJKˆ˜XÝ[ÛˆÙYßH‚ˆ
NÂˆBˆËÈH›Ü
K“ÔÒUSJH[™H\ÙHØZ]›ÜˆH[Ý™HÚ]HšY]ÉÜÈÛÚË‚ˆ]\ÙYHÎŽ•\ÙR][J^NŽ•\ÙR][HÂˆ[™ˆˆÙ\]Y[˜ÙNˆ‹ˆX]ÎˆŒˆ]ÚˆŒˆJNÂˆ\ÜÙ\JØZ]×Ù›Ü—ÝWÝšY]Ê	˜XÝ[ÛŠJKYK˜[ÙJJNÂˆ\ÜÙ\JØZ]×Ù›Ü—ÝWÝšY]Ê	\ÙYYK˜[ÙJJNÂˆ\ÜÙ\J]ØZ]×Ù›Ü—ÝWÝšY]Ê	\ÙY˜[ÙK˜[ÙJJNÂˆËÈY\ˆÛÛY][™È[]™\ž][™ÈØZ]Ë[ˆÜ™\ŽÈHÝÚ[™È[Û™H™]™\ˆÙ\Ë‚ˆ\ÜÙ\JØZ]×Ù›Ü—ÝWÝšY]Ê	˜XÝ[ÛŠ
KYKYJJNÂˆ\ÜÙ\JØZ]×Ù›Ü—ÝWÝšY]Ê	”ÎŽ”[˜Ú
[\JK˜[ÙKYJJNÂˆ\ÜÙ\J]ØZ]×Ù›Ü—ÝWÝšY]Ê	”ÎŽ”[˜Ú
[\JKYK˜[ÙJJNÂˆB‚ˆÖÝ\ÝBˆ›ˆXÙ[Y[Ü™]šY]×ØÚXÚÜ×ÛØœÝXÝ[Û—ÝÚ]Ý]ØÚ[™Ú[™×ÝWÛ]™[

HÂˆ\ÙHÝ\\ŽŽ‘žS]™[Âˆ\ÙHÜš^›Û—ØÛÜ™NŽžÜÜÚ][ÛŽŽ›ØÚÔÜË˜[™ÛNŽ“YØXÞT˜[™Û_NÂˆ\ÙHÜš^›Û—ÝÛÜ›ŽžÂˆ]™[ŽžÂˆ™Z]š[Ý\œË]™[XØÙ\ÜËXÙPÛÛ^XÙSÝ]ÛÛYKXÚÜË]NŽ›ØYÜ™YÚ\ÝžKˆKˆ™YÚ\ÝžNŽ›ØÚÔÝ]RYˆNÂ‚ˆÝXÝ]™[ÂˆZ\Žˆ›ØÚÔÝ]RYˆÝÛ™Nˆ›ØÚÔÝ]RYˆÝÙ\Žˆ›ØÚÔÝ]RYˆÛY[ÜÚYNˆ›ÛÛˆØœÝXÝYˆ›ÛÛˆØYYˆ›ÛÛˆÙZ[[™ÎˆLÌ‹ˆBˆ[\Üš^›Û—ÝÛÜ›Ž›]™[Ž›ØÚÑ[]S™ZYÚ›ÜœÈ›Üˆ]™[ßBˆ[\]™[XØÙ\ÜÈ›Üˆ]™[Âˆ›ˆ›ØÚ×ÜÝ]J	œÙ[‹ÜÎˆ›ØÚÔÜÊHOˆ›ØÚÔÝ]RYÂˆYˆÜÈOH›ØÚÔÜÎŽ›™]Ê
HÂˆÙ[‹œÝÛ™BˆH[ÙHYˆÜÈOH›ØÚÔÜÎŽ›™]ÊLKK
HÂˆÙ[‹œÝÙ\‚ˆH[ÙHÂˆÙ[‹˜Z\‚ˆBˆBˆ›ˆÙ]Ü˜]Ê	›]]Ù[‹Îˆ›ØÚÔÜËÎˆ›ØÚÔÝ]RY
HOˆÜ[Û›ØÚÔÝ]RYˆÂˆ[šXÈJœ™]šY]È]\Ý›ÝÜš]HÈH]™H]™[ŠNÂˆBˆ›ˆ\×ÛØYY
	œÙ[‹Îˆ›ØÚÔÜÊHOˆ›ÛÛÂˆÙ[‹›ØYYˆBˆ›ˆ\×ØÛY[ÜÚYJ	œÙ[ŠHOˆ›ÛÛÂˆÙ[‹˜ÛY[ÜÚYBˆBˆ›ˆX^ÞJ	œÙ[ŠHOˆLÌˆÂˆÙ[‹˜ÙZ[[™ÂˆBˆ›ˆ\×Ý[›ØœÝXÝY
	œÙ[‹ÜÎˆ›ØÚÔÜËÎˆ›ØÚÔÝ]RY
HOˆ›ÛÛÂˆ\ÜÙ\Ù\HJÜË›ØÚÔÜÎŽ›™]ÊK
JNÂˆ\Ù[‹›ØœÝXÝYˆBˆ›ˆ˜[™ÛJ	›]]Ù[ŠHOˆ	›]]YØXÞT˜[™ÛHÂˆ[šXÈJœ™]šY]È]\Ý›ÝY˜[˜ÙHH]™H]™[	ÜÈ˜[™ÛHÛÝ\˜ÙHŠNÂˆBˆ›ˆØ[YWÝ[YJ	œÙ[ŠHOˆMÂˆˆBˆB‚ˆËÈ\ÈÚ]HSˆ\ÝË™\\™YØ[YH™\ÛÝ\˜Ù\È\™HÜ[Û˜[ØØ[K‚ˆ]™\ÛÝ\˜Ù\ÈHÝŽœ]Ž”]Ž›™]Ê[ˆJÐT‘Ó×ÓPS’Q‘TÕÑTˆŠJBˆš›Ú[Š‹‹‹Ë‹‹Ë›ØØ[ÚÜš^›Û‹XÛY[ŠBˆš›Ú[ŠÜš^›Û—ØÛÜ™NŽ•S’SWÕ‘T”ÒSÓŠNÂˆYˆ\™\ÛÝ\˜Ù\Ëš›Ú[Š˜›ØÚËX™Z]š[Ý\œËšœÛÛˆŠK™^\ÝÊ
HÂˆ™]\›ŽÂˆBˆ]™YÚ\ÝžHHØYÜ™YÚ\ÝžJ	œ™\ÛÝ\˜Ù\ÊK[Ü˜\

NÂˆ]™Z]š[Ý\œÈH™Z]š[Ý\œÎŽ›ØY
	œ™YÚ\ÝžK	œ™\ÛÝ\˜Ù\ÊK[Ü˜\

NÂˆ]]]˜\ÙHH]™[ÂˆZ\Žˆ™YÚ\ÝžK™Y˜][ÜÝ]J›Z[™XÜ˜Y˜Z\ˆŠK[Ü˜\

KˆÝÛ™Nˆ™YÚ\ÝžK™Y˜][ÜÝ]J›Z[™XÜ˜YœÝÛ™HŠK[Ü˜\

KˆÝÙ\Žˆ™YÚ\ÝžK™Y˜][ÜÝ]J›Z[™XÜ˜Yœ™YÝÛ™WØ›ØÚÈŠK[Ü˜\

KˆÛY[ÜÚYNˆYKˆØœÝXÝYˆ˜[ÙKˆØYYˆYKˆÙZ[[™ÎˆÌŒˆNÂˆ›Üˆ][H[ˆÂˆ›Z[™XÜ˜YœÝÛ™H‹ˆ›Z[™XÜ˜Y›ØZ×ÜÛXˆ‹ˆ›Z[™XÜ˜Y›ØZ×ÙÛÜˆ‹ˆ›Z[™XÜ˜Y‹ˆ›Z[™XÜ˜Yœ\ÝÛˆ‹ˆ›Z[™XÜ˜YœÝXÚÞWÜ\ÝÛˆ‹ˆËÈÜ›ÝÚ[™Ô[XY›ØÚË™Ù]Ý]Q›Ü”XÙ[Y[˜]ÜÈH˜[™ÛHYÙK‚ˆ›Z[™XÜ˜YÚ\Ý[™×Ýš[™\È‹ˆHÂˆ]ÝHXÙPÛÛ^ÂˆÛXÚÙYÜÜÎˆ›ØÚÔÜÎŽ›™]Ê
Kˆ˜XÙNˆÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ‘\™XÝ[ÛŽŽ•\ˆ]ˆÜš^›Û—ØÛÜ™NŽ™Ù[ÛY]žNŽ•™XÌÎŽ›™]ÊKKŒJKˆ[œÚYNˆ˜[ÙKˆX]ÎˆŒˆ]ÚˆKŒˆÛ™XZÚ[™Îˆ˜[ÙKˆ][Nˆ][Kš[Ê
KˆØ[—Ý\ÙWÙØ[YWÛX\Ý\—Ø›ØÚÜÎˆ˜[ÙKˆNÂˆ›Üˆ
ÛY[ÜÚYKØœÝXÝYØYYÙZ[[™ÊH[ˆÂˆ
YK˜[ÙKYKÌŒ
Kˆ
YKYKYKÌŒ
Kˆ
YK˜[ÙK˜[ÙKÌŒ
Kˆ
YK˜[ÙKYKJKˆ
˜[ÙK˜[ÙKYKÌŒ
KˆHÂˆ˜\ÙK˜ÛY[ÜÚYHHÛY[ÜÚYNÂˆ˜\ÙK›ØœÝXÝYHØœÝXÝYÂˆ˜\ÙK›ØYYHØYYÂˆ˜\ÙK˜ÙZ[[™ÈHÙZ[[™ÎÂˆ]˜[YH[ØœÝXÝY	‰ˆØYY	‰ˆÙZ[[™ÈˆNÂˆ]]]™]šY]ÈHžS]™[Ž›™]Ê	˜˜\ÙJNÂˆ]]]XÚÜÈHXÚÜÎŽ™Y˜][

NÂˆ]Ý]ÛÛYHH™Z]š[Ý\œËœXÙWØ›ØÚÊ	›]]™]šY]Ë	›]]XÚÜË	˜Ý
NÂˆ\ÜÙ\Ù\HJˆX]Ú\ÈJÝ]ÛÛYKXÙSÝ]ÛÛYNŽ”XÙYÈ‹ˆJKˆ˜[YˆžÚ][_KØœÝXÝY^ÛØœÝXÝYKØYY^ÛØYYKÙZ[[™Ï^ØÙZ[[™ßH‚ˆ
NÂˆ\ÜÙ\Ù\HJ™]šY]Ë˜Ú[™ÙYš\×Ù[\J
K]˜[Y
NÂˆYˆ˜[Yˆ	‰ˆX]Ú\ÈJˆ][Kˆ›Z[™XÜ˜Yˆ›Z[™XÜ˜Yœ\ÝÛˆˆ›Z[™XÜ˜YœÝXÚÞWÜ\ÝÛˆ‚ˆ
BˆÂˆ\ÜÙ\Ù\HJˆ™YÚ\ÝžBˆœÝ]J™]šY]Ë˜›ØÚ×ÜÝ]J›ØÚÔÜÎŽ›™]ÊK
JJBˆ[Ü˜\

Bˆ›˜[YKˆYˆ][HOH›Z[™XÜ˜Yˆ	‰ˆXÛY[ÜÚYHÂˆ›Z[™XÜ˜Y˜Z\ˆ‚ˆH[ÙHÂˆ][BˆKˆœÝÙ\™YÚ][_KÛY[ÜÚYO^ØÛY[ÜÚY_H‚ˆ
NÂˆ\ÜÙ\Ù\HJˆXÚÜË˜›ØÚ×Ù]™[Ëš\×Ù[\J
KˆÛY[ÜÚYH][HOH›Z[™XÜ˜Y‹ˆ›Û›HHÙ\™\ˆÝ\ÈHÝÙ\™Y\ÝÛˆ‚ˆ
NÂˆBˆ\ÜÙ\Ù\HJ˜\ÙK˜›ØÚ×ÜÝ]J›ØÚÔÜÎŽ›™]ÊK
JK˜\ÙK˜Z\ŠNÂˆBˆBˆB‚ˆÖÝ\ÝBˆ›ˆWØÜ™X]]™WÜÞ[˜×ØÛÝ™\œ×Ú[™[ÜžWÛY[WÜÛÝ×ÍWÝ×ÍWÛÛ˜ÙWÙXXÚ

HÂˆËÈÜ™X]]™R[™[ÜžS\Ý[™\ˆÙ[™È]™\žH[™[ÜžSY[HÛÝ]HÜ˜Y[™ÈÛ™\ÎˆBˆËÈ\›[Ý\ˆ
K‹N
KH[™[ÜžH
K‹M
H[™HÙ™ˆ[™
JK‚ˆ]ÝXÚÈH][Nˆ	œÝŸÛÛYJÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎŽ›™]Ê][KJJNÂˆ][™[ÜžNˆ™XÏÏˆH
‹ŒÍŠBˆ›X\
_ÝXÚÊ	™›Ü›X]J›Z[™XÜ˜Yš][WÞÚ_HŠJJBˆ˜ÛÛXÝ

NÂˆ]Ù™š[™HÝXÚÊ›Z[™XÜ˜YœÚY[ŠNÂˆ]\›[ÜˆHÂˆÝXÚÊ›Z[™XÜ˜Y™X[[Û™Ú[Y]ŠKˆÝXÚÊ›Z[™XÜ˜Y™X[[Û™ØÚ\Ý]HŠKˆÝXÚÊ›Z[™XÜ˜Y™X[[Û™ÛYÙÚ[™ÜÈŠKˆÝXÚÊ›Z[™XÜ˜Y™X[[Û™Ø›ÛÝÈŠKˆNÂˆ]ÝXÚÜÈHÝ\\ŽŽ˜Ü™X]]™WÜÞ[˜×ÜÝXÚÜÊ	š[™[ÜžK	›Ù™š[™	˜\›[ÜŠNÂˆ\ÜÙ\Ù\HJÝXÚÜË›[Š
KÝ\\ŽŽÔ‘PUU‘WÔÖSÑQ
NÂˆ]]]ÛÝÎˆ™XÏLMˆH
‹œÝXÚÜË›[Š
JK›X\
Ý\\ŽŽ˜Ü™X]]™WÜÞ[˜×ÜÛÝ
K˜ÛÛXÝ

NÂˆËÈXXÚ[žHÛÙ\ÈÈHY[HÛÝ]ÛÈ]ÝXÚË‚ˆ›Üˆ
K	œÛÝ
H[ˆÛÝËš]\Š
K™[[Y\˜]J
HÂˆ]Ø[YWÜÛÝHØ[YNŽš[™[ÜžWÜØÜ™Y[—ÜÛÝ
š[™[ÜžH‹LÌŽŽ™œ›ÛJÛÝ
JK[Ü˜\

NÂˆ]^XÝYHX]ÚØ[YWÜÛÝÂˆ‹ŒÍˆOˆ	š[™[ÜžVÙØ[YWÜÛÝKˆLOˆ	›Ù™š[™ˆL‹LOˆ	˜\›[Ü–ÙØ[YWÜÛÝHLKˆÈOˆ[šXÈJ™[žHÚ_HÙ[ÈØ[YHÛÝÙØ[YWÜÛÝHŠKˆNÂˆ\ÜÙ\Ù\HJ	œÝXÚÜÖÚWK^XÝY™[žHÚ_KY[HÛÝÜÛÝHŠNÂˆBˆÛÝËœÛÜ

NÂˆ\ÜÙ\Ù\HJÛÝË
K‹MJK˜ÛÛXÝŽ™XÏLMŠ
JNÂˆB‚ˆÖÝ\ÝBˆ›ˆWÜÙ\™\—ÜÛÝÝ\]WÛX]™\×Ø[—Ý[œÙ[ØÜ™X]]™WØÛXÚ×Ý×Ø™WÜÙ[

HÂˆ]ÝXÚÈH][Nˆ	œÝŸÛÛYJÜš^›Û—ØÛY[Žš][\ÎŽ’][TÝXÚÎŽ›™]Ê][KJJNÂˆ]›Û™HH™XÈVÓ›Û™NÈÝ\\ŽŽÔ‘PUU‘WÔÖSÑQNÂˆËÈÞ[˜ÙYˆ›Ý[™ËˆHÛXÚÈ]ÈH[Y]Ûˆ
[žHÍË›ÝÙ[Y]
K[ˆHÙ\™\‚ˆËÈÙ]ÈÝ˜\ˆÛÝ
HXÚÝ\
NˆÛ›H][žHÛÝ[È\ÈÞ[˜ÙY‚ˆ]]]Þ[˜ÙYH›Û™J
NÂˆ]]]™Y›Ü™HH›Û™J
NÂˆ™Y›Ü™VÌÍ×HHÝXÚÊ›Z[™XÜ˜Y™X[[Û™Ú[Y]ŠNÂˆ]]]›ÝÈH™Y›Ü™K˜ÛÛ™J
NÂˆ›ÝÖÌHHÝXÚÊ›Z[™XÜ˜YœÝXÚÈŠNÂˆÝ\\ŽŽ›X\š×ÜÞ[˜ÙY
	›]]Þ[˜ÙY	˜™Y›Ü™K›ÝË˜ÛÛ™J
JNÂˆ\ÜÙ\Ù\HJÞ[˜ÙYÌK›ÝÖÌJNÂˆ\ÜÙ\Ù\HJˆÞ[˜ÙYÌÍ×K›Û™KˆH[Y]Ý[Y™™\œËÛÈH™^XÚÈÙ[™È]‚ˆ
NÂˆB‚ˆÖÝ\ÝBˆ›ˆ]™\žWÚ[™[ÜžWÛY[WÜÛÝÚ\×ØWÙØ[YWÜÛÝØ[™ÍWÚ\×ÝWÛÙ™—Ú[™

HÂˆËÈ[™[ÜžSY[Nˆ™\Ý[žˆÜšY\›[Ý\‹È
ÈH[™[ÜžHÛÝË[ˆHÙ™ˆ[™‚ˆ\ÜÙ\Ù\HJØ[YNŽš[™[ÜžWÜØÜ™Y[—ÜÛÝ
š[™[ÜžH‹JKÛÛYJL
JNÂˆ]]]ÙY[ˆHÝŽ˜ÛÛXÝ[ÛœÎŽ’\ÚÙ]Ž›™]Ê
NÂˆ›ÜˆÛÝ[ˆ‹ˆÂˆ][\›˜[HØ[YNŽš[™[ÜžWÜØÜ™Y[—ÜÛÝ
š[™[ÜžH‹ÛÝ
NÂˆ\ÜÙ\J[\›˜[š\×ÜÛÛYWØ[™
_ÙY[‹š[œÙ\
JJKœÛÝÜÛÝHŠNÂˆBˆ\ÜÙ\Ù\HJØ[YNŽš[™[ÜžWÜØÜ™Y[—ÜÛÝ
š[™[ÜžH‹ŠK›Û™JNÂˆB‚ˆÖÝ\ÝBˆ›ˆ™[[ÝWØÛXÚÜ×Ùš[™ÜÛÝ×Ü\ÝÝWÛÛÛ\™ÙWØÚ\ÝØÝ]Ù™Š
HÂˆ›Üˆ
Y[K[\›˜[[™^
H[ˆÂˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹NŒÊKˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹NK
Kˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹JKˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹JKˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹ÍLËLÊKˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^H‹
Kˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^‹ÌJKˆ
›Z[™XÜ˜Y™Ù[™\šX×Î^È‹ŒŠKˆ
š[™[ÜžH‹LJKˆHÂˆ\ÜÙ\Ù\HJˆØ[YNŽ›Y[WÜÛÝÚ[™^
Y[K[\›˜[
KˆÛÛYJ[™^
KˆžÛY[_KØ[YHÛÝÚ[\›˜[H‚ˆ
NÂˆBˆ\ÜÙ\Ù\HJØ[YNŽ›Y[WÜÛÝÚ[™^
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹ÍM
K›Û™JNÂˆ\ÜÙ\Ù\HJØ[YNŽ›Y[WÜÛÝÚ[™^
›Z[™XÜ˜Y™Ù[™\šX×Î^ˆ‹L
K›Û™JNÂˆ\ÜÙ\Ù\HJØ[YNŽ›Y[WÜÛÝÚ[™^
[šÛ›ÝÛˆ‹
K›Û™JNÂˆB‚ˆËËÈš\™]ÛÜšÔ›ØÚÙ]][K\ÙX\ÈÕPÐÑTÔÈÚ[HÛY[™ÎˆHXZ[ˆ[™	ÜÂˆËËÈ›ØÚÙ][™ÈHÛXÚÈ[™HÙ™ˆ[™	ÜÈÚY[\È›Ý˜Z\ÙY
ÌÍ
K‚ˆËËÈÛˆ›ÛÝ]\ÜÙ\ËÛÈHÙ™ˆ[™Ý[Ù]È]È\›‹‚ˆÖÝ\ÝBˆ›ˆWÜ›ØÚÙ]Ý\ÙYÝÚ[WÙÛY[™×Ù[™×ÝWØÛXÚÊ
HÂˆ]›ÛÝHÝŽœ]Ž”]YŽŽ™œ›ÛJ[ˆJÐT‘Ó×ÓPS’Q‘TÕÑTˆŠJBˆš›Ú[Š‹‹‹Ë‹‹Ë›ØØ[ÚÜš^›Û‹XÛY[ŠBˆš›Ú[ŠÜš^›Û—ØÛÜ™NŽ•S’SWÕ‘T”ÒSÓŠNÂˆYˆ\›ÛÝš›Ú[Šš][\ËšœÛÛˆŠK™^\ÝÊ
HÂˆ\š[ˆJœÚÚ\[™Îˆ›È™\\™Y™\ÛÝ\˜Ù\È[ˆßH‹›ÛÝ™\Ü^J
JNÂˆ™]\›ŽÂˆBˆ]™YÚ\ÝžHHÜš^›Û—ÝÛÜ›Ž›]™[Ž™]NŽ›ØYÜ™YÚ\ÝžJ	œ›ÛÝ
K[Ü˜\

NÂˆ]˜XÝÈHÜš^›Û—ØÛY[Ž˜›ØÚÜÎŽ›ØÚÑ˜XÝÎŽ›ØY
	œ›ÛÝ	œ™YÚ\ÝžJK[Ü˜\

NÂˆ]Y˜][ÈHÜš^›Û—ÝÛÜ›Žš][WÙY˜][ÎŽœ™XY
	œ›ÛÝ
K[Ü˜\

NÂˆ]][\ÈHÜš^›Û—ØÛY[Žš][\ÎŽ’][\ÎŽ›ØY
ˆ	œ›ÛÝˆ	œ™YÚ\ÝžKˆ	™˜XÝËˆ	šÜš^›Û—ÝÛÜ›Ž›ÛÝŽ”XÚÓÝ™\›^NŽ““Ó‘Kˆ	™Y˜][Ëˆ
Bˆ[Ü˜\

NÂˆ]›ØÚÙ]H][\Ë™Ù]
›Z[™XÜ˜Y™š\™]ÛÜš×Ü›ØÚÙ]ŠK[Ü˜\

NÂˆ\ÜÙ\JÝ\\ŽŽ\ÙWÙ[™×ØÛXÚÊ›ØÚÙ]YJK™ÛY[™ÈŠNÂˆ\ÜÙ\J\Ý\\ŽŽ\ÙWÙ[™×ØÛXÚÊ›ØÚÙ]˜[ÙJK›Ûˆ›ÛÝŠNÂˆËÈÝ\ˆ][\È[œÝÙ\ˆHØ[YHÚ]\ˆÜˆ›ÝH^Y\ˆÛY\Ë‚ˆ]ÚY[H][\Ë™Ù]
›Z[™XÜ˜YœÚY[ŠK[Ü˜\

NÂˆ]ÝXÚÈH][\Ë™Ù]
›Z[™XÜ˜YœÝXÚÈŠK[Ü˜\

NÂˆ›ÜˆÛY[™È[ˆÙ˜[ÙKYWHÂˆ\ÜÙ\JÝ\\ŽŽ\ÙWÙ[™×ØÛXÚÊÚY[ÛY[™ÊJNÂˆ\ÜÙ\J\Ý\\ŽŽ\ÙWÙ[™×ØÛXÚÊÝXÚËÛY[™ÊJNÂˆËÈ›ÛÙÛTÝXÚÒ][K\ÙH\ÜÙ\ÈÛˆHÛY[ˆ›ÈÝÚ[™ÈÚ[HH›ÛÜÝÛÛÛÈÝÛ‹‚ˆ›Üˆ›Ù[ˆÂˆ›Z[™XÜ˜Y˜Ø\œ›ÝÛÛ—ØWÜÝXÚÈ‹ˆ›Z[™XÜ˜YØ\œYÙ[™Ý\×ÛÛ—ØWÜÝXÚÈ‹ˆHÂˆ]›ÙH][\Ë™Ù]
›Ù
K[Ü˜\

NÂˆ\ÜÙ\J\Ý\\ŽŽ\ÙWÙ[™×ØÛXÚÊ›ÙÛY[™ÊKžßH‹›ÙšY
NÂˆBˆBˆB‚ˆËËÈÛY[XÚÙ]\Ý[™\‹š[™QØ[YQ]™[ˆ™\šYžH]HØ[YH]™[XÚÙ]ˆËËÈ\\È›Üˆ\œ›ÝÈ]Y™™\™š\ÚÝ[™È[™[Z]YÜ˜Y[™ÈX]Ú˜[š[K‚ˆÖÝ\ÝBˆ›ˆØ[YWÙ]™[ØÛÛœÝ[×Ø[™Ý\\×ÛX]ÚÝ˜[š[J
HÂˆ\ÙHÜš^›Û—Ü›ÝØÛÛŽœXÚÙ]ÎŽœ^NŽ‘Ø[YQ]™[ÂˆËÈ]™[\\Èœ›ÛHÛY[›Ý[™Ø[YQ]™[XÚÙ]‚ˆËÈˆHVWÐT”“Õ×ÒUÔÓÕS‘ÈHRS—ÓU‘SÐÒS‘ÑKHS‘T—ÓU‘SÐÒS‘ÑKˆËÈHHQ‘‘T—Ñ’TÒÔÕS‘ËLHHSSQQPUWÔ‘TÔUÓ‹LˆHSRUQÐÔQ•S‘Ë‚ˆ]\œ›ÝÈHØ[YQ]™[Âˆ]™[ˆ‹ˆ\˜[NˆŒˆNÂˆ]˜Z[ˆHØ[YQ]™[Âˆ]™[ˆËˆ\˜[NˆKˆNÂˆ][™\ˆHØ[YQ]™[Âˆ]™[ˆˆ\˜[NˆŒ‹ˆNÂˆ]Y™™\ˆHØ[YQ]™[Âˆ]™[ˆKˆ\˜[NˆŒˆNÂˆ]™\Ü]ÛˆHØ[YQ]™[Âˆ]™[ˆLKˆ\˜[NˆKŒˆNÂˆ]Ü˜Y[™ÈHØ[YQ]™[Âˆ]™[ˆL‹ˆ\˜[NˆKŒˆNÂ‚ˆ\ÜÙ\Ù\HJ\œ›ÝË™]™[ŠNÂˆ\ÜÙ\Ù\HJ˜Z[‹™]™[ÊNÂˆ\ÜÙ\Ù\HJ[™\‹™]™[
NÂˆ\ÜÙ\Ù\HJY™™\‹™]™[JNÂˆ\ÜÙ\Ù\HJ™\Ü]Û‹™]™[LJNÂˆ\ÜÙ\Ù\HJÜ˜Y[™Ë™]™[LŠNÂˆ\ÜÙ\JÜ˜Y[™Ëœ\˜[HOHKŒ
NÂˆBŸB‚ˆÖØÙ™Ê\Ý
WBˆÖÜ]Hœ^Y\—ÝXÚ×Ý\ÝËœœÈ—B›[Ù^Y\—ÝXÚ×Ý\ÝÎÂ‚‹ËËÈÚ]\ˆHÞ[˜Ú›Ûš^™Y[Y[œÚ[Ûˆ\IÜÈ˜[Y\È\™H[œÚYHHÛÙXÉÜÈ˜[™Ù\Â‹ËËÈ
[Y[œÚ[Û•\Kš˜]˜XˆHÛÛÜ™[˜]HØØ[HÙˆKŒKMQ˜ÈËŒMØH[ÛœÝ\ˆÜ]Ûˆ›ØÚÈYÚ‹ËËÈ[Z]ÙˆÈMNÈH[XšY[YÚ\È›È˜[™ÙJKÛÈHÜ[	ÜÈ˜][ÈÙˆÛÈØØ[\È\Èš[š]K‚™›ˆØ[™WÙ˜XÝÊ˜XÝÎˆ	šÜš^›Û—ÝÛÜ›Ž™[Y[œÚ[Û—Ý\NŽ‘[Y[œÚ[Û‘˜XÝÊHOˆ›ÛÛÂˆ
Ž™œ›ÛJKŒKMWÙŒÌŠK‹LËŒMÊK˜ÛÛZ[œÊ	™˜XÝË˜ÛÛÜ™[˜]WÜØØ[JBˆ	‰ˆ
‹LMJK˜ÛÛZ[œÊ	™˜XÝË›[ÛœÝ\—ÜÜ]Û—Ø›ØÚ×ÛYÚÛ[Z]
Bˆ	‰ˆ˜XÝË˜[XšY[ÛYÚš\×Ùš[š]J
BŸB‚ˆÖØÙ™Ê\Ý
WB›[Ù]™[Ù˜XÝ×Ý\ÝÈÂˆ\ÙHÝ\\ŽŽœØ[™WÙ˜XÝÎÂˆ\ÙHÜš^›Û—ÝÛÜ›Ž™[Y[œÚ[Û—Ý\NŽ‘[Y[œÚ[Û‘˜XÝÎÂ‚ˆËËÈH\HÝ]ÚYHHÛÙXÉÜÈ˜[™Ù\È\È›ÝZÙ[ŽÈH˜[š[HÛ™\È\™K‚ˆÖÝ\ÝBˆ›ˆWÝ\WÛÝ]ÚYWÝWØÛÙXÜ×Ü˜[™Ù\×Ú\×Ü™Y\ÙY

HÂˆ]]]˜XÝÈH[Y[œÚ[Û‘˜XÝÎŽ›Ý™\ÛÜ›

K˜ÛÛ™J
NÂˆ\ÜÙ\JØ[™WÙ˜XÝÊ	™˜XÝÊJNÂˆ\ÜÙ\JØ[™WÙ˜XÝÊ[Y[œÚ[Û‘˜XÝÎŽ›Ù—ØZ[[—Û]™[
ˆÜš^›Û—ØÛÜ™NŽ“]™[Ù^NŽ“‘UT‚ˆ
JJNÂˆ›ÜˆØØ[H[ˆÌŒŽ™œ›ÛWØš]ÊJKLKŒËŒYMËŽ“S—HÂˆ˜XÝË˜ÛÛÜ™[˜]WÜØØ[HHØØ[NÂˆ\ÜÙ\J\Ø[™WÙ˜XÝÊ	™˜XÝÊKžÜØØ[_HŠNÂˆBˆ˜XÝË˜ÛÛÜ™[˜]WÜØØ[HHKŒÂˆËÈHÛÙXÈÙ]È›È˜[™ÙHÛˆH[XšY[YÚˆ˜XÝË˜[XšY[ÛYÚHKNÂˆ\ÜÙ\JØ[™WÙ˜XÝÊ	™˜XÝÊJNÂˆ˜XÝË›[ÛœÝ\—ÜÜ]Û—Ø›ØÚ×ÛYÚÛ[Z]HMŽÂˆ\ÜÙ\J\Ø[™WÙ˜XÝÊ	™˜XÝÊJNÂˆBŸB