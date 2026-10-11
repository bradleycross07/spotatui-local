use super::mapping::{map_page, playlist_items_page};
use super::requests::{
  is_forbidden_error, is_not_found_error, is_rate_limited_error, spotify_api_request_json_for,
  spotify_get_typed_for,
};
use super::{IoEvent, Network};
use crate::core::app::{
  ActiveBlock, App, PlaylistFolder, PlaylistFolderItem, PlaylistFolderNode, PlaylistFolderNodeType,
  RouteId,
};
use crate::core::pagination::Paged;
use crate::core::plugin_api::{PlayableInfo, PlaylistInfo, ShowInfo, TrackInfo};
use crate::core::sort::Sorter;
use crate::core::source::Source;
use anyhow::anyhow;
use reqwest::Method;
use rspotify::model::{
  idtypes::{AlbumId, PlaylistId, ShowId, TrackId, UserId},
  page::Page,
  playlist::{PlaylistItem, SimplifiedPlaylist},
  track::SavedTrack,
};
use rspotify::prelude::*;
use serde_json::json;
#[cfg(feature = "streaming")]
use std::collections::HashMap;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[cfg(feature = "streaming")]
use crate::infra::player::StreamingPlayer;
#[cfg(feature = "streaming")]
use librespot_core::SpotifyUri;
#[cfg(feature = "streaming")]
use librespot_metadata::{
  Episode as LibrespotEpisode, Metadata, Playlist as LibrespotPlaylist, Track as LibrespotTrack,
};

// Spotify's `me/library` endpoints (contains, save, remove) accept at most 40
// uris per request; anything larger fails with a 400 "Too many uris".
const LIBRARY_CONTAINS_MAX_URIS: usize = 40;
/// Contains batches in flight at once; the answers keep the order of the uris.
const LIBRARY_CONTAINS_PARALLEL: usize = 4;

const EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS: &str = concat!(
  "Spotify Development Mode blocks playlist contents owned by another user. ",
  "Only playlists you own or collaborate on are available."
);

/// How many external-playlist fallbacks are remembered per session. Bounds the
/// cache: each entry holds one classification plus (streaming builds) one
/// playlist's item URIs, so an unbounded map would grow with every pasted URI.
const EXTERNAL_PLAYLIST_FALLBACK_CAP: usize = 32;

/// Parallel librespot metadata requests when resolving rootlist-only
/// playlists; bounded so a large library does not burst the session.
#[cfg(feature = "streaming")]
const ROOTLIST_RESOLVE_CONCURRENCY: usize = 8;

/// Overall time limit for resolving rootlist-only playlists. Folders and the
/// resolved rows publish together, so a hung request must not hold the whole
/// sidebar back. Playlists that miss it show up on the next refresh.
#[cfg(feature = "streaming")]
const ROOTLIST_RESOLVE_BUDGET: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaylistAccess {
  Owned,
  External,
  Unknown,
}

/// Whether a 403 on the items endpoint may use the librespot fallback. Only a
/// confirmed-external playlist with a real 403 qualifies: an owned or
/// collaborative playlist that 403s is a diagnosable server error, and any
/// non-403 failure keeps its real error.
fn should_attempt_external_fallback(access: PlaylistAccess, forbidden: bool) -> bool {
  forbidden && access == PlaylistAccess::External
}

/// Owner id Spotify uses for its generated and editorial playlists.
const SPOTIFY_OWNER_ID: &str = "spotify";

/// Development Mode answers 404, not 403, for the items of a Spotify-owned
/// playlist (Release Radar, daylist, the Mixes). Treat the 404 as hidden, not
/// missing, only when the playlist is already known to be Spotify's, e.g. a
/// sidebar entry resolved from the rootlist. Any other 404 stays a real error.
fn is_hidden_spotify_owned_playlist(not_found: bool, owner_id: Option<&str>) -> bool {
  not_found && owner_id == Some(SPOTIFY_OWNER_ID)
}

/// Next raw-page offset for a converted playlist page, driven by continuation
/// metadata only. Compaction (`playlist_items_page`) and the librespot slice
/// can drop every item of a nonterminal page, so item count must never stop
/// pagination. The response offset must equal the requested one: a stale page
/// would otherwise repeat the same request forever. A zero limit, an overflow,
/// or a non-advancing offset terminates instead of looping.
fn playlist_page_next_offset(
  page: &Paged<(u32, PlayableInfo)>,
  requested_offset: u32,
) -> Option<u32> {
  if page.limit == 0 {
    return None;
  }
  page.next.as_ref()?;
  let next_offset = page.offset.checked_add(page.limit)?;
  (page.offset == requested_offset && next_offset > requested_offset).then_some(next_offset)
}

/// Session-scoped reuse for the external-playlist fallback path, shared by
/// foreground fetch, background prefetch, sort, and search through one
/// `Arc<Mutex<…>>` handle. Locks are held only for map lookups/inserts, never
/// across Spotify or librespot awaits.
pub(crate) type ExternalPlaylistFallbackCache = Arc<Mutex<ExternalPlaylistFallbackCacheInner>>;

pub(crate) fn external_playlist_fallback_cache() -> ExternalPlaylistFallbackCache {
  Arc::new(Mutex::new(ExternalPlaylistFallbackCacheInner::default()))
}

#[derive(Debug, Default)]
pub(crate) struct ExternalPlaylistFallbackCacheInner {
  /// Playlist ids confirmed `External` via metadata after a 403, so later
  /// pages skip the `GET playlists/{id}` classification request.
  external_ids: HashSet<String>,
  /// Resolved librespot contents per external playlist id, so later page
  /// slices reuse the downloaded proto instead of refetching it.
  #[cfg(feature = "streaming")]
  contents: HashMap<String, CachedExternalPlaylistContents>,
  /// Insertion order for the bounded eviction in `retain_newest`.
  order: VecDeque<String>,
}

/// Resolved librespot playlist contents: the item URIs in playlist order plus
/// the base offset and total from the downloaded proto. Absolute positions are
/// `base_offset + index`, so page slices stay aligned with the raw playlist.
#[cfg(feature = "streaming")]
#[derive(Debug, Clone)]
struct CachedExternalPlaylistContents {
  uris: Vec<librespot_core::SpotifyUri>,
  base_offset: u32,
  total: u32,
}

#[cfg(feature = "streaming")]
impl CachedExternalPlaylistContents {
  /// Absolute `(position, item-index)` pairs for the requested raw window:
  /// `offset=0, limit=50` selects `[0..50]` of the cached contents, `50/50`
  /// selects `[50..100]`, and so on.
  fn slice_positions(&self, offset: u32, limit: u32) -> Vec<(u32, usize)> {
    let end = offset.saturating_add(limit.max(1));
    self
      .uris
      .iter()
      .enumerate()
      .map(|(index, _)| (self.base_offset.saturating_add(index as u32), index))
      .filter(|(position, _)| *position >= offset && *position < end)
      .collect()
  }

  fn has_next(&self, offset: u32, limit: u32) -> bool {
    offset.saturating_add(limit.max(1)) < self.total
  }
}

impl ExternalPlaylistFallbackCacheInner {
  fn is_external(&self, playlist_id: &str) -> bool {
    self.external_ids.contains(playlist_id)
  }

  fn mark_external(&mut self, playlist_id: &str) {
    if self.external_ids.insert(playlist_id.to_string()) {
      self.order.push_back(playlist_id.to_string());
    }
    self.retain_newest();
  }

  #[cfg(feature = "streaming")]
  fn contents_for(&self, playlist_id: &str) -> Option<CachedExternalPlaylistContents> {
    self.contents.get(playlist_id).cloned()
  }

  #[cfg(feature = "streaming")]
  fn store_contents(&mut self, playlist_id: &str, contents: CachedExternalPlaylistContents) {
    self.contents.insert(playlist_id.to_string(), contents);
    self.mark_external(playlist_id);
  }

  /// Drop a playlist's resolved contents (e.g. on a fresh open at offset 0)
  /// so the next visit re-downloads instead of showing the first snapshot
  /// all session. The External classification is kept: the relationship
  /// rarely changes, and re-proving it would cost a metadata request again.
  #[cfg(feature = "streaming")]
  fn clear_contents(&mut self, playlist_id: &str) {
    self.contents.remove(playlist_id);
  }

  /// Drop the oldest ids once over capacity. Classification and contents for
  /// an id are evicted together; the next visit simply re-resolves them.
  fn retain_newest(&mut self) {
    while self.order.len() > EXTERNAL_PLAYLIST_FALLBACK_CAP {
      if let Some(oldest) = self.order.pop_front() {
        self.external_ids.remove(&oldest);
        #[cfg(feature = "streaming")]
        self.contents.remove(&oldest);
      } else {
        break;
      }
    }
  }
}

#[derive(Debug, serde::Deserialize)]
struct PlaylistAccessMetadata {
  owner: Option<PlaylistOwnerMetadata>,
}

#[derive(Debug, serde::Deserialize)]
struct PlaylistOwnerMetadata {
  id: String,
}

/// The Web API page is still the primary representation. The librespot page is
/// source-agnostic domain data because its internal metadata API does not
/// expose rspotify's `PlaylistItem` model.
enum PlaylistTracksPage {
  Api(Page<PlaylistItem>),
  #[cfg(feature = "streaming")]
  Librespot(Paged<(u32, PlayableInfo)>),
}

/// Page size for whole-playlist walks (search and sort): the internal API's
/// default page, independent of terminal height.
const PLAYLIST_WALK_PAGE_LIMIT: u32 = 100;
/// Pages in flight at once during a whole-playlist walk. Every request still
/// passes the shared `Retry-After` gate, so a 429 pauses all of them together.
const PLAYLIST_WALK_CONCURRENCY: usize = 6;

/// Minimum spacing between page requests of whole-playlist walks: 2 per
/// second, half the 4/s global pacing. Applies to every walk, including
/// playlists read through the librespot fallback, where each page fans out
/// into one metadata lookup per track on the user's own session every Spotify call already obeys.
/// Walks are the app's only long bursts of requests, and running them at the
/// global ceiling for their whole duration triggered Spotify's rate limit
/// (which then blocks every API call for minutes). Interactive requests keep
/// the global pace; only bulk walks slow down.
const PLAYLIST_WALK_MIN_INTERVAL: Duration = Duration::from_millis(500);

/// A burst-free pacer: each `wait` starts at least `interval` after the
/// previous one's slot. Slots are reserved under the lock and slept on outside
/// it, so concurrent waiters queue in order without serializing their sleeps.
struct WalkPacer {
  interval: Duration,
  next_slot: Mutex<Option<tokio::time::Instant>>,
}

impl WalkPacer {
  fn new(interval: Duration) -> Self {
    Self {
      interval,
      next_slot: Mutex::new(None),
    }
  }

  async fn wait(&self) {
    let start_at = {
      let mut next_slot = self.next_slot.lock().await;
      let now = tokio::time::Instant::now();
      // Clamp to `now` so idle time banks no credit: no bursts after a pause.
      let slot = next_slot.map_or(now, |slot| slot.max(now));
      *next_slot = Some(slot + self.interval);
      slot
    };
    tokio::time::sleep_until(start_at).await;
  }
}

/// One pacer shared by every walk, so a search and a sort running at the same
/// time still total 2 requests per second.
/// Pacer for background heart lookups: one request a second.
fn liked_lookup_pacer() -> &'static WalkPacer {
  static PACER: std::sync::OnceLock<WalkPacer> = std::sync::OnceLock::new();
  PACER.get_or_init(|| WalkPacer::new(Duration::from_secs(1)))
}

fn playlist_walk_pacer() -> &'static WalkPacer {
  static PACER: std::sync::OnceLock<WalkPacer> = std::sync::OnceLock::new();
  PACER.get_or_init(|| WalkPacer::new(PLAYLIST_WALK_MIN_INTERVAL))
}

/// The lock serializing walks of one playlist. Entries are held weakly, so a
/// playlist's lock is freed once no walk holds it, and dead entries are pruned
/// whenever a new one is created, keeping the map as small as the walks in use.
fn playlist_walk_lock(playlist_id: &str) -> Arc<Mutex<()>> {
  use std::sync::{Mutex as StdMutex, OnceLock, Weak};
  static LOCKS: OnceLock<StdMutex<std::collections::HashMap<String, Weak<Mutex<()>>>>> =
    OnceLock::new();

  let mut locks = LOCKS
    .get_or_init(Default::default)
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  if let Some(lock) = locks.get(playlist_id).and_then(Weak::upgrade) {
    return lock;
  }
  locks.retain(|_, lock| lock.strong_count() > 0);
  let lock = Arc::new(Mutex::new(()));
  locks.insert(playlist_id.to_string(), Arc::downgrade(&lock));
  lock
}

/// Fetch every item of a playlist, in order, with their raw positions.
///
/// Page one is fetched alone to learn the total and the page size the server
/// actually honored; the remaining pages are then requested concurrently
/// (`buffered` keeps them in playlist order). Replaces the one-page-at-a-time
/// walk, whose cost grew linearly in round trips with playlist length.
async fn fetch_all_playlist_items(
  app: &Arc<Mutex<App>>,
  playlist_id: &PlaylistId<'_>,
  fallbacks: &ExternalPlaylistFallbackCache,
) -> Result<Vec<(u32, PlayableInfo)>, PlaylistPageError> {
  let id = playlist_id.id().to_string();
  let cache_dir = super::playlist_cache::default_dir();

  // One walk per playlist at a time. A second search or sort of the same
  // playlist waits here for the running walk, then finds its result in the
  // cache below instead of fetching every page again. Without this, typing
  // a few searches into an uncached playlist ran one full walk per search.
  let walk_lock = playlist_walk_lock(&id);
  let _walk_guard = walk_lock.lock().await;

  // Playlists already classified external are served by the librespot
  // fallback; the Web API would only 403 the snapshot request, so skip it.
  let known_external = fallbacks.lock().await.is_external(&id);
  let snapshot_id = match &cache_dir {
    Some(_) if !known_external => fetch_playlist_snapshot_id(&id).await,
    _ => None,
  };

  if let (Some(dir), Some(snapshot)) = (&cache_dir, &snapshot_id) {
    let (dir, load_id, snapshot) = (dir.clone(), id.clone(), snapshot.clone());
    let cached =
      tokio::task::spawn_blocking(move || super::playlist_cache::load(&dir, &load_id, &snapshot))
        .await
        .ok()
        .flatten();
    if let Some(items) = cached {
      log::debug!("playlist cache: hit for {id} ({} items)", items.len());
      return Ok(items);
    }
  }

  let items = fetch_all_playlist_items_uncached(app, playlist_id, fallbacks).await?;

  // Record under the snapshot read *before* the walk: if the playlist changed
  // mid-walk, the stored snapshot is already stale and the next use refetches,
  // so a race can cost one extra walk but never serve outdated contents. The
  // write finishes before the walk lock is released, so a caller waiting on
  // the lock is guaranteed to find the entry.
  if let (Some(dir), Some(snapshot)) = (cache_dir, snapshot_id) {
    let to_store = items.clone();
    let stored = tokio::task::spawn_blocking(move || {
      super::playlist_cache::store(&dir, &id, &snapshot, &to_store).map_err(|error| (id, error))
    })
    .await;
    if let Ok(Err((id, error))) = stored {
      log::warn!("playlist cache: could not store {id}: {error}");
    }
  }

  Ok(items)
}

/// How an open went with the disk cache.
enum CachedOpen {
  /// The playlist came from the cache (or the open was superseded meanwhile).
  Served,
  /// Load from the network; `snapshot_id`, when known, lets the page chain
  /// store the result.
  Miss { snapshot_id: Option<String> },
}

/// Page size for pages rebuilt from the disk cache (the Web API maximum).
const CACHED_PAGE_LIMIT: u32 = 50;

/// Rebuild the playlist page cache from a whole cached playlist, as the pages
/// an open would have downloaded: one page per `CACHED_PAGE_LIMIT` raw
/// positions, contiguous from offset 0, `next` set on all but the last. A page
/// whose positions were all removed tracks is kept empty so the chain of
/// offsets stays unbroken (the continuous table rebuild stops at a gap).
fn pages_from_cached_items(items: Vec<(u32, PlayableInfo)>) -> Vec<Paged<(u32, PlayableInfo)>> {
  let total = items
    .iter()
    .map(|(position, _)| position.saturating_add(1))
    .max()
    .unwrap_or(0);
  let page_count = total.div_ceil(CACHED_PAGE_LIMIT).max(1);
  let mut pages: Vec<Paged<(u32, PlayableInfo)>> = (0..page_count)
    .map(|index| Paged {
      items: Vec::new(),
      offset: index * CACHED_PAGE_LIMIT,
      limit: CACHED_PAGE_LIMIT,
      total,
      next: (index + 1 < page_count).then(|| "spotatui:playlist-cache:next".to_string()),
      previous: None,
    })
    .collect();
  for (position, item) in items {
    if let Some(page) = pages.get_mut((position / CACHED_PAGE_LIMIT) as usize) {
      page.items.push((position, item));
    }
  }
  pages
}

/// Every item of a playlist, when `pages` (sorted by offset, as the page cache
/// keeps them) cover it from offset 0 to a page with no `next`, with no gap
/// between consecutive pages. `None` for anything partial, so an interrupted
/// load is never stored as if it were the whole playlist.
fn complete_playlist_items(
  pages: &[Paged<(u32, PlayableInfo)>],
) -> Option<Vec<(u32, PlayableInfo)>> {
  let mut expected_offset = 0u32;
  let mut items = Vec::new();
  for page in pages {
    if page.offset != expected_offset || page.limit == 0 {
      return None;
    }
    items.extend(page.items.iter().cloned());
    if page.next.is_none() {
      return Some(items);
    }
    expected_offset = expected_offset.checked_add(page.limit)?;
  }
  None
}

/// Store a playlist the open-time page chain finished loading, if its pages
/// are complete. Called with the `snapshot_id` read before the first page, so
/// a playlist edited mid-load is stored under a stale tag and refetched next
/// time rather than served out of date.
fn store_completed_playlist(
  pages: &[Paged<(u32, PlayableInfo)>],
  playlist_id: &PlaylistId<'_>,
  snapshot_id: Option<&str>,
) {
  let (Some(snapshot), Some(dir)) = (snapshot_id, super::playlist_cache::default_dir()) else {
    return;
  };
  let Some(items) = complete_playlist_items(pages) else {
    return;
  };
  let (id, snapshot) = (playlist_id.id().to_string(), snapshot.to_string());
  tokio::task::spawn_blocking(move || {
    match super::playlist_cache::store(&dir, &id, &snapshot, &items) {
      Ok(()) => log::debug!(
        "playlist cache: stored {id} from open ({} items)",
        items.len()
      ),
      Err(error) => log::warn!("playlist cache: could not store {id}: {error}"),
    }
  });
}

/// The playlist's current `snapshot_id` (one request, `fields`-filtered so the
/// response is a few bytes), or `None` when it cannot be read; callers then
/// bypass the cache instead of failing.
async fn fetch_playlist_snapshot_id(playlist_id: &str) -> Option<String> {
  #[derive(serde::Deserialize)]
  struct PlaylistSnapshot {
    snapshot_id: String,
  }

  match spotify_get_typed_for::<PlaylistSnapshot>(
    &format!("playlists/{playlist_id}"),
    &[("fields", "snapshot_id".to_string())],
  )
  .await
  {
    Ok(snapshot) if !snapshot.snapshot_id.is_empty() => Some(snapshot.snapshot_id),
    Ok(_) => None,
    Err(error) => {
      log::debug!("playlist cache: no snapshot_id for {playlist_id}, bypassing cache: {error}");
      None
    }
  }
}

/// The network walk behind [`fetch_all_playlist_items`].
async fn fetch_all_playlist_items_uncached(
  app: &Arc<Mutex<App>>,
  playlist_id: &PlaylistId<'_>,
  fallbacks: &ExternalPlaylistFallbackCache,
) -> Result<Vec<(u32, PlayableInfo)>, PlaylistPageError> {
  use futures::stream::{self, StreamExt, TryStreamExt};

  playlist_walk_pacer().wait().await;
  let first = fetch_playlist_tracks_page(app, playlist_id, 0, PLAYLIST_WALK_PAGE_LIMIT, fallbacks)
    .await?
    .into_domain();

  // Same continuation rules as the sequential walk: no `next` (or a zero
  // limit) means this was the only page.
  if playlist_page_next_offset(&first, 0).is_none() {
    return Ok(first.items);
  }

  // Step by the limit the server reported, so a server-side cap below the
  // requested limit cannot leave gaps between pages.
  let step = first.limit;
  let total = first.total;
  let mut items = first.items;
  let offsets: Vec<u32> = std::iter::successors(Some(step), |offset| offset.checked_add(step))
    .take_while(|offset| *offset < total)
    .collect();

  let pages: Vec<Paged<(u32, PlayableInfo)>> = stream::iter(offsets)
    .map(|offset| async move {
      playlist_walk_pacer().wait().await;
      fetch_playlist_tracks_page(app, playlist_id, offset, step, fallbacks)
        .await
        .map(PlaylistTracksPage::into_domain)
    })
    .buffered(PLAYLIST_WALK_CONCURRENCY)
    .try_collect()
    .await?;

  for page in pages {
    items.extend(page.items);
  }
  Ok(items)
}

impl PlaylistTracksPage {
  fn into_domain(self) -> Paged<(u32, PlayableInfo)> {
    match self {
      Self::Api(page) => playlist_items_page(&page),
      #[cfg(feature = "streaming")]
      Self::Librespot(page) => page,
    }
  }
}

#[derive(Debug)]
enum PlaylistPageError {
  /// Spotify's Development Mode restriction applies to this external playlist
  /// and neither the Web API nor the optional native fallback can serve it.
  UnsupportedExternal,
  Request(anyhow::Error),
}

/// Classify a playlist's 403 by owner relationship only. Spotify's
/// `collaborative` flag says the playlist itself is collaborative; it does
/// NOT prove the current user is one of its collaborators, so it must not
/// decide the fallback. A followed collaborative playlist owned by someone
/// else is External.
fn playlist_access_from_owner(user_id: Option<&str>, owner_id: Option<&str>) -> PlaylistAccess {
  match (user_id, owner_id) {
    (Some(user_id), Some(owner_id)) if owner_id == user_id => PlaylistAccess::Owned,
    (Some(_), Some(_)) => PlaylistAccess::External,
    _ => PlaylistAccess::Unknown,
  }
}

fn known_playlist_info<'a>(app: &'a App, playlist_id: &str) -> Option<&'a PlaylistInfo> {
  app
    .all_playlists()
    .iter()
    .find(|playlist| playlist.id.as_deref() == Some(playlist_id))
    .or_else(|| {
      app
        .search_results()
        .playlists
        .as_ref()?
        .items
        .iter()
        .find(|playlist| playlist.id.as_deref() == Some(playlist_id))
    })
}

fn playlist_access(app: &App, playlist_id: &str) -> PlaylistAccess {
  let Some(playlist) = known_playlist_info(app, playlist_id) else {
    return PlaylistAccess::Unknown;
  };

  playlist_access_from_owner(
    app.user().as_ref().map(|user| user.id.as_str()),
    playlist.owner_id.as_deref(),
  )
}

fn log_playlist_access(app: &App, playlist_id: &str, access: PlaylistAccess) {
  if let Some(playlist) = known_playlist_info(app, playlist_id) {
    log::debug!(
      "playlist content access: id={} access={access:?} owner_id={:?} user_id={:?} collaborative={} public={:?}",
      playlist_id,
      playlist.owner_id,
      app.user().as_ref().map(|user| user.id.as_str()),
      playlist.collaborative,
      playlist.public,
    );
  } else {
    log::debug!(
      "playlist content access: id={} access={access:?} metadata=unknown",
      playlist_id,
    );
  }
}

async fn classify_playlist_after_forbidden(
  app: &Arc<Mutex<App>>,
  playlist_id: &PlaylistId<'_>,
  fallbacks: &ExternalPlaylistFallbackCache,
) -> PlaylistAccess {
  let known_access = {
    let app_guard = app.lock().await;
    let access = playlist_access(&app_guard, playlist_id.id());
    log_playlist_access(&app_guard, playlist_id.id(), access);
    access
  };
  if known_access != PlaylistAccess::Unknown {
    return known_access;
  }

  // A previous page already proved this id external: reuse that without
  // another metadata request.
  if fallbacks.lock().await.is_external(playlist_id.id()) {
    log::debug!(
      "playlist content access: id={} access=External metadata=cached",
      playlist_id.id(),
    );
    return PlaylistAccess::External;
  }

  // Search results and pasted playlist URIs are not necessarily in the user's
  // library. The metadata endpoint remains useful for relationship checks even
  // when the items endpoint is restricted, and the minimal shape avoids asking
  // rspotify to deserialize a response with no item page.
  let metadata = match spotify_get_typed_for::<PlaylistAccessMetadata>(
    &format!("playlists/{}", playlist_id.id()),
    &[],
  )
  .await
  {
    Ok(metadata) => metadata,
    Err(error) => {
      log::debug!(
        "playlist access metadata unavailable for {}: {}",
        playlist_id.id(),
        error
      );
      return PlaylistAccess::Unknown;
    }
  };

  let access = {
    let app_guard = app.lock().await;
    playlist_access_from_owner(
      app_guard.user().as_ref().map(|user| user.id.as_str()),
      metadata.owner.as_ref().map(|owner| owner.id.as_str()),
    )
  };
  log::debug!(
    "playlist content access from metadata: id={} access={access:?} owner_id={:?}",
    playlist_id.id(),
    metadata.owner.as_ref().map(|owner| owner.id.as_str()),
  );
  if access == PlaylistAccess::External {
    fallbacks.lock().await.mark_external(playlist_id.id());
  }
  access
}

/// Fetch playlist items through the Development Mode Web API first. A 403 is
/// special only when the playlist metadata proves another user owns it;
/// all other errors remain real errors.
async fn fetch_playlist_tracks_page(
  app: &Arc<Mutex<App>>,
  playlist_id: &PlaylistId<'_>,
  offset: u32,
  limit: u32,
  fallbacks: &ExternalPlaylistFallbackCache,
) -> Result<PlaylistTracksPage, PlaylistPageError> {
  // A cached external playlist skips the Web API round trip (which would
  // just 403 again) and slices the resolved contents directly. A failure
  // here — e.g. the streaming session went away — falls through to the
  // normal Web API attempt instead of failing the page. The lookup is bound
  // before the `if let` so no MutexGuard is held across the awaits below.
  #[cfg(feature = "streaming")]
  {
    let cached = fallbacks.lock().await.contents_for(playlist_id.id());

    if let Some(cached) = cached {
      if let Ok(session) = streaming_session(app).await {
        if let Ok(page) = resolve_cached_playlist_slice(&session, &cached, offset, limit).await {
          return Ok(PlaylistTracksPage::Librespot(page));
        }
      }
    }
  }

  let path = format!("playlists/{}/items", playlist_id.id());
  let query = vec![("limit", limit.to_string()), ("offset", offset.to_string())];
  match spotify_get_typed_for::<Page<PlaylistItem>>(&path, &query).await {
    Ok(page) => Ok(PlaylistTracksPage::Api(page)),
    Err(error) => {
      let forbidden = is_forbidden_error(&error);
      let hidden_spotify_owned = !forbidden && {
        let app_guard = app.lock().await;
        is_hidden_spotify_owned_playlist(
          is_not_found_error(&error),
          known_playlist_info(&app_guard, playlist_id.id())
            .and_then(|playlist| playlist.owner_id.as_deref()),
        )
      };
      let access = if forbidden {
        classify_playlist_after_forbidden(app, playlist_id, fallbacks).await
      } else if hidden_spotify_owned {
        log::debug!(
          "playlist content access: id={} access=External reason=spotify-owned-404",
          playlist_id.id()
        );
        PlaylistAccess::External
      } else {
        PlaylistAccess::Unknown
      };

      if should_attempt_external_fallback(access, forbidden || hidden_spotify_owned) {
        #[cfg(feature = "streaming")]
        match fetch_librespot_playlist_tracks_page(app, fallbacks, playlist_id, offset, limit).await
        {
          Ok(page) => return Ok(PlaylistTracksPage::Librespot(page)),
          Err(fallback_error) => {
            log::warn!(
              "librespot playlist fallback failed for {}: {}",
              playlist_id.id(),
              fallback_error
            );
          }
        }
        return Err(PlaylistPageError::UnsupportedExternal);
      }

      Err(PlaylistPageError::Request(error))
    }
  }
}

#[cfg(feature = "streaming")]
async fn streaming_session(app: &Arc<Mutex<App>>) -> anyhow::Result<librespot_core::Session> {
  let player = {
    let app_guard = app.lock().await;
    app_guard.streaming_player.clone()
  }
  .ok_or_else(|| anyhow!("native streaming session is unavailable"))?;
  Ok(player.session())
}

#[cfg(feature = "streaming")]
async fn fetch_librespot_playlist_tracks_page(
  app: &Arc<Mutex<App>>,
  fallbacks: &ExternalPlaylistFallbackCache,
  playlist_id: &PlaylistId<'_>,
  offset: u32,
  limit: u32,
) -> anyhow::Result<Paged<(u32, PlayableInfo)>> {
  let session = streaming_session(app).await?;

  // Reuse the proto a previous page already downloaded: later slices resolve
  // only their own window of track metadata. The lookup is bound before the
  // `if let` so no MutexGuard is held across the resolve await.
  let cached = fallbacks.lock().await.contents_for(playlist_id.id());

  if let Some(cached) = cached {
    return resolve_cached_playlist_slice(&session, &cached, offset, limit).await;
  }

  let id = librespot_core::SpotifyId::from_base62(playlist_id.id())
    .map_err(|error| anyhow!("invalid playlist id for librespot: {error}"))?;
  let uri = SpotifyUri::Playlist { user: None, id };
  let playlist = LibrespotPlaylist::get(&session, &uri)
    .await
    .map_err(|error| anyhow!("librespot playlist request failed: {error}"))?;

  if playlist.contents.is_truncated {
    return Err(anyhow!("librespot returned truncated playlist contents"));
  }

  let cached = CachedExternalPlaylistContents {
    total: playlist.length.max(0) as u32,
    base_offset: playlist.contents.position.max(0) as u32,
    uris: playlist
      .contents
      .items
      .iter()
      .map(|item| item.id.clone())
      .collect(),
  };
  fallbacks
    .lock()
    .await
    .store_contents(playlist_id.id(), cached.clone());
  resolve_cached_playlist_slice(&session, &cached, offset, limit).await
}

#[cfg(feature = "streaming")]
async fn resolve_cached_playlist_slice(
  session: &librespot_core::Session,
  cached: &CachedExternalPlaylistContents,
  offset: u32,
  limit: u32,
) -> anyhow::Result<Paged<(u32, PlayableInfo)>> {
  let limit = limit.max(1);
  let selected = cached
    .slice_positions(offset, limit)
    .into_iter()
    .map(|(position, index)| (position, cached.uris[index].clone()))
    .collect::<Vec<_>>();

  let resolved = resolve_playlist_slice(session, selected).await;

  Ok(Paged {
    items: resolved,
    offset,
    limit,
    total: cached.total,
    next: cached
      .has_next(offset, limit)
      .then(|| "librespot:playlist:next".to_string()),
    previous: None,
  })
}

#[cfg(feature = "streaming")]
async fn resolve_playlist_slice(
  session: &librespot_core::Session,
  selected: Vec<(u32, librespot_core::SpotifyUri)>,
) -> Vec<(u32, PlayableInfo)> {
  futures::future::join_all(selected.into_iter().map(|(position, uri)| {
    let session = session.clone();
    async move {
      let item = match &uri {
        SpotifyUri::Track { .. } => match LibrespotTrack::get(&session, &uri).await {
          Ok(track) => librespot_track_info(&track).map(PlayableInfo::Track),
          Err(error) => {
            log::debug!("librespot track metadata failed for {}: {}", uri, error);
            None
          }
        },
        SpotifyUri::Episode { .. } => match LibrespotEpisode::get(&session, &uri).await {
          Ok(episode) => librespot_episode_info(&episode).map(PlayableInfo::Episode),
          Err(error) => {
            log::debug!("librespot episode metadata failed for {}: {}", uri, error);
            None
          }
        },
        _ => {
          log::debug!("librespot playlist item is not playable: {}", uri);
          None
        }
      };
      (position, item)
    }
  }))
  .await
  .into_iter()
  .filter_map(|(position, item)| item.map(|item| (position, item)))
  .collect()
}

/// Default playability for fallback items. Librespot's `restrictions` is the
/// raw per-catalogue list, so `restrictions.is_empty()` marks nearly every
/// fallback item unplayable for Lua/MCP readers. No clearly correct supported
/// signal is exposed, so fallback items default to playable.
#[cfg(feature = "streaming")]
fn external_fallback_track_is_playable() -> bool {
  true
}

#[cfg(feature = "streaming")]
fn librespot_track_info(track: &LibrespotTrack) -> Option<TrackInfo> {
  let id = track.id.to_id();
  let uri = Some(track.id.to_uri());
  let album_id = Some(track.album.id.to_id());
  let artist_refs = track
    .artists
    .iter()
    .map(|artist| crate::core::plugin_api::ArtistRef {
      id: Some(artist.id.to_id()),
      name: artist.name.clone(),
    })
    .collect::<Vec<_>>();
  // librespot reads an absent date as year 0; keep only the year, since a
  // missing month or day reads as January or the 1st.
  let year = track.album.date.year();
  let release_date = (year > 0).then(|| format!("{year:04}"));

  Some(TrackInfo {
    uri,
    name: track.name.clone(),
    artists: artist_refs
      .iter()
      .map(|artist| artist.name.clone())
      .collect(),
    album: track.album.name.clone(),
    duration_ms: track.duration.max(0) as u64,
    id: Some(id),
    album_id,
    artist_refs,
    is_playable: external_fallback_track_is_playable(),
    is_local: false,
    track_number: track.number.max(0) as u32,
    explicit: track.is_explicit,
    image_url: None,
    release_date,
  })
}

#[cfg(feature = "streaming")]
fn librespot_episode_info(
  episode: &LibrespotEpisode,
) -> Option<crate::core::plugin_api::EpisodeInfo> {
  Some(crate::core::plugin_api::EpisodeInfo {
    id: Some(episode.id.to_id()),
    uri: Some(episode.id.to_uri()),
    name: episode.name.clone(),
    duration_ms: episode.duration.max(0) as u64,
    show_name: episode.show_name.clone(),
    description: episode.description.clone(),
    release_date: String::new(),
    is_playable: external_fallback_track_is_playable(),
    resume_point: None,
    image_url: None,
  })
}

#[cfg(test)]
fn next_saved_tracks_offset(page: &Page<SavedTrack>) -> Option<u32> {
  page.next.as_ref().map(|_| page.offset + page.limit)
}

fn uri_batches(uris: &[String]) -> impl Iterator<Item = &[String]> {
  uris.chunks(LIBRARY_CONTAINS_MAX_URIS)
}

async fn contains_batch(uris: String) -> anyhow::Result<Vec<bool>> {
  spotify_get_typed_for::<Vec<bool>>("me/library/contains", &[("uris", uris)]).await
}

fn populate_liked_song_ids_from_saved_tracks(
  liked_song_ids_set: &mut std::collections::HashSet<String>,
  page: &Page<SavedTrack>,
) {
  for item in &page.items {
    if let Some(track_id) = &item.track.id {
      liked_song_ids_set.insert(track_id.id().to_string());
    }
  }
  // Every track in Liked Songs is liked: free answers for the heart cache.
  super::liked_cache::record_all(
    page
      .items
      .iter()
      .filter_map(|item| item.track.id.as_ref())
      .map(|id| (id.id(), true)),
  );
}

fn playlist_track_search_terms(query: &str) -> Vec<String> {
  query
    .split_whitespace()
    .map(|term| term.to_lowercase())
    .filter(|term| !term.is_empty())
    .collect()
}

fn playlist_track_info_matches_terms(track: &TrackInfo, terms: &[String]) -> bool {
  let haystack =
    format!("{} {} {}", track.name, track.album, track.artists.join(" "),).to_lowercase();
  terms.iter().all(|term| haystack.contains(term))
}

#[cfg(test)]
fn domain_track_page(
  offset: u32,
  limit: u32,
  total: u32,
  has_next: bool,
  items: Vec<(u32, PlayableInfo)>,
) -> Paged<(u32, PlayableInfo)> {
  Paged {
    items,
    offset,
    limit,
    total,
    next: has_next.then(|| "https://example.com/playlists/test/items?next".to_string()),
    previous: None,
  }
}

/// Newest songs that key the Liked Songs cache. Fixed rather than the whole
/// first page, whose size follows the terminal height: resizing the window
/// must not invalidate the cache. A like always lands on top, so the newest
/// few are enough to see it.
const LIKED_SONGS_KEY_HEAD: usize = 10;

/// The cache key of Liked Songs as its first page describes it.
fn liked_songs_key(first_page: &Page<SavedTrack>) -> super::liked_songs_cache::LikedSongsKey {
  super::liked_songs_cache::LikedSongsKey {
    total: first_page.total,
    head_ids: first_page
      .items
      .iter()
      .take(LIKED_SONGS_KEY_HEAD)
      .map(|item| {
        item
          .track
          .id
          .as_ref()
          .map(|id| id.id().to_string())
          .unwrap_or_default()
      })
      .collect(),
  }
}

async fn load_liked_songs(key: super::liked_songs_cache::LikedSongsKey) -> Option<Vec<TrackInfo>> {
  let path = super::liked_songs_cache::default_path()?;
  tokio::task::spawn_blocking(move || super::liked_songs_cache::load(&path, &key))
    .await
    .ok()
    .flatten()
}

/// Rebuild the Liked Songs page cache from the first page just fetched and
/// the cached whole list: the remaining pages use the first page's size, so
/// offsets stay contiguous (the table rebuild stops at a gap).
fn saved_track_pages_from_cache(
  first_page: Paged<TrackInfo>,
  tracks: Vec<TrackInfo>,
) -> Vec<Paged<TrackInfo>> {
  let limit = first_page.limit.max(1) as usize;
  let total = tracks.len() as u32;
  let rest: Vec<TrackInfo> = tracks.into_iter().skip(first_page.items.len()).collect();
  let mut pages = vec![Paged {
    total,
    next: (!rest.is_empty()).then(|| "spotatui:liked-songs-cache:next".to_string()),
    ..first_page
  }];
  let chunks: Vec<Vec<TrackInfo>> = rest.chunks(limit).map(<[TrackInfo]>::to_vec).collect();
  let count = chunks.len();
  for (index, items) in chunks.into_iter().enumerate() {
    pages.push(Paged {
      items,
      offset: ((index + 1) * limit) as u32,
      limit: limit as u32,
      total,
      next: (index + 1 < count).then(|| "spotatui:liked-songs-cache:next".to_string()),
      previous: None,
    });
  }
  pages
}

/// Show Liked Songs from the cache: every page at once, every heart known.
fn apply_cached_liked_songs(app: &mut App, first_page: Paged<TrackInfo>, tracks: Vec<TrackInfo>) {
  let count = tracks.len();
  let ids: Vec<String> = tracks.iter().filter_map(|track| track.id.clone()).collect();
  super::liked_cache::record_all(ids.iter().map(|id| (id.as_str(), true)));
  app.liked_song_ids_set_mut().extend(ids);
  for page in saved_track_pages_from_cache(first_page, tracks) {
    app.saved_tracks_mut().upsert_page_by_offset(page);
  }
  // Fills the table only while it shows Liked Songs; the pages are kept either way.
  app.refresh_saved_tracks_table();
  app
    .plugin_data_generations
    .bump(crate::core::app::PluginDataKind::SavedTracks);
  log::debug!("liked songs cache: opened from cache ({count} songs)");
}

/// Store Liked Songs when every page is loaded, under the key read from the
/// first page of this open. A list edited mid-load is stored under a stale
/// key and refetched next time, never served out of date.
fn store_liked_songs_if_complete(app: &App, key: Option<&super::liked_songs_cache::LikedSongsKey>) {
  let (Some(key), Some(path)) = (key, super::liked_songs_cache::default_path()) else {
    return;
  };
  let (tracks, _, has_more) = app.saved_tracks_prefix();
  if has_more || tracks.len() != key.total as usize {
    return;
  }
  let key = key.clone();
  tokio::task::spawn_blocking(move || {
    match super::liked_songs_cache::store(&path, &key, &tracks) {
      Ok(()) => log::debug!("liked songs cache: stored {} songs", tracks.len()),
      Err(error) => log::warn!("liked songs cache: could not store: {error}"),
    }
  });
}

pub async fn prefetch_saved_tracks_page_task(
  app: Arc<Mutex<App>>,
  limit: u32,
  mut offset: u32,
  generation: u64,
  key: Option<super::liked_songs_cache::LikedSongsKey>,
) {
  loop {
    let should_fetch = {
      let mut app = app.lock().await;
      app.saved_tracks_prefetch_generation == generation
        && app
          .library()
          .saved_tracks
          .page_index_for_offset(offset)
          .is_none()
        && app.saved_tracks_prefetch_in_flight.insert(offset)
    };

    if !should_fetch {
      return;
    }

    let query = vec![("limit", limit.to_string()), ("offset", offset.to_string())];
    let Ok(page) =
      spotify_get_typed_for::<Page<rspotify::model::SavedTrack>>("me/tracks", &query).await
    else {
      let mut app_guard = app.lock().await;
      app_guard.saved_tracks_prefetch_in_flight.remove(&offset);
      return;
    };

    if page.items.is_empty() {
      let mut app_guard = app.lock().await;
      app_guard.saved_tracks_prefetch_in_flight.remove(&offset);
      return;
    }

    let next_offset = page.next.as_ref().map(|_| page.offset + page.limit);
    let mut app_guard = app.lock().await;
    app_guard.saved_tracks_prefetch_in_flight.remove(&offset);
    if app_guard.saved_tracks_prefetch_generation != generation {
      return;
    }

    populate_liked_song_ids_from_saved_tracks(app_guard.liked_song_ids_set_mut(), &page);
    let domain_page =
      crate::infra::network::mapping::map_page(&page, |st| TrackInfo::from(&st.track));
    app_guard
      .saved_tracks_mut()
      .upsert_page_by_offset(domain_page);
    app_guard.refresh_saved_tracks_table();
    let Some(candidate_next_offset) = next_offset else {
      // The last page landed: store the whole list if nothing is missing.
      store_liked_songs_if_complete(&app_guard, key.as_ref());
      return;
    };
    let should_prefetch_next = app_guard
      .library()
      .saved_tracks
      .page_index_for_offset(candidate_next_offset)
      .is_none()
      && !app_guard
        .saved_tracks_prefetch_in_flight
        .contains(&candidate_next_offset);
    drop(app_guard);

    if should_prefetch_next {
      offset = candidate_next_offset;
    } else {
      return;
    }
  }
}

// Eight args is the spawned-task threading (client, state, paths, playlist,
// offsets, generation, shared fallback cache); cf. `finish_playlists_fetch`.
#[allow(clippy::too_many_arguments)]
pub async fn prefetch_playlist_tracks_page_task(
  app: Arc<Mutex<App>>,
  limit: u32,
  playlist_id: PlaylistId<'static>,
  mut offset: u32,
  generation: u64,
  fallbacks: ExternalPlaylistFallbackCache,
  snapshot_id: Option<String>,
) {
  loop {
    let should_fetch = {
      let mut app = app.lock().await;
      app.playlist_tracks_prefetch_generation == generation
        && app.is_playlist_track_table_active_for(&playlist_id)
        && app
          .playlist_track_pages
          .page_index_for_offset(offset)
          .is_none()
        && app.playlist_tracks_prefetch_in_flight.insert(offset)
    };

    if !should_fetch {
      return;
    }

    let page = match fetch_playlist_tracks_page(&app, &playlist_id, offset, limit, &fallbacks).await
    {
      Ok(page) => page.into_domain(),
      Err(PlaylistPageError::UnsupportedExternal) => {
        let mut app_guard = app.lock().await;
        app_guard.playlist_tracks_prefetch_in_flight.remove(&offset);
        app_guard.set_error_status_message(EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS, 8);
        return;
      }
      Err(PlaylistPageError::Request(_)) => {
        // Background prefetch stays silent like on main: the user may have
        // left the playlist, or the shared request path may be inside a
        // `Retry-After` window whose synthetic 429 must not open the
        // full-screen error route.
        let mut app_guard = app.lock().await;
        app_guard.playlist_tracks_prefetch_in_flight.remove(&offset);
        return;
      }
    };

    // Pagination runs on continuation metadata only: a compacted page may
    // hold zero domain items while `next` still promises more raw slots, so
    // the empty page is upserted and the next offset is still derived.
    let next_offset = playlist_page_next_offset(&page, offset);
    let mut app_guard = app.lock().await;
    app_guard.playlist_tracks_prefetch_in_flight.remove(&offset);
    if app_guard.playlist_tracks_prefetch_generation != generation
      || !app_guard.is_playlist_track_table_active_for(&playlist_id)
    {
      return;
    }

    app_guard.playlist_track_pages.upsert_page_by_offset(page);
    app_guard.set_playlist_tracks_to_table_continuous();
    let Some(candidate_next_offset) = next_offset else {
      // The last page landed: store the playlist if every page is present.
      store_completed_playlist(
        &app_guard.playlist_track_pages.pages,
        &playlist_id,
        snapshot_id.as_deref(),
      );
      return;
    };
    let should_prefetch_next = app_guard
      .playlist_track_pages
      .page_index_for_offset(candidate_next_offset)
      .is_none()
      && !app_guard
        .playlist_tracks_prefetch_in_flight
        .contains(&candidate_next_offset);
    drop(app_guard);

    if should_prefetch_next {
      offset = candidate_next_offset;
    } else {
      return;
    }
  }
}

pub trait LibraryNetwork {
  async fn get_current_user_playlists(&mut self);
  async fn get_playlist_tracks(&mut self, playlist_id: PlaylistId<'static>, playlist_offset: u32);
  async fn search_playlist_tracks(&mut self, playlist_id: PlaylistId<'static>, query: String);
  async fn get_current_user_saved_tracks(&mut self, offset: Option<u32>);
  async fn get_current_user_saved_albums(&mut self, offset: Option<u32>);
  async fn current_user_saved_albums_contains(&mut self, album_ids: Vec<AlbumId<'static>>);
  async fn current_user_saved_album_delete(&mut self, album_id: AlbumId<'static>);
  async fn current_user_saved_album_add(&mut self, album_id: AlbumId<'static>);
  async fn current_user_saved_shows_contains(&mut self, show_ids: Vec<ShowId<'static>>);
  async fn current_user_saved_shows_delete(&mut self, show_id: ShowId<'static>);
  async fn current_user_saved_shows_add(&mut self, show_id: ShowId<'static>);
  async fn get_current_user_saved_shows(&mut self, offset: Option<u32>);
  async fn user_follow_playlist(
    &mut self,
    playlist_owner_id: UserId<'static>,
    playlist_id: PlaylistId<'static>,
    is_public: Option<bool>,
  );
  async fn user_unfollow_playlist(
    &mut self,
    user_id: UserId<'static>,
    playlist_id: PlaylistId<'static>,
  );
  async fn add_track_to_playlist(
    &mut self,
    playlist_id: PlaylistId<'static>,
    track_id: TrackId<'static>,
  );
  async fn remove_track_from_playlist_at_position(
    &mut self,
    playlist_id: PlaylistId<'static>,
    track_id: TrackId<'static>,
    position: usize,
  );
  async fn toggle_save_track(&mut self, track_id: rspotify::model::idtypes::PlayableId<'static>);
  async fn current_user_saved_tracks_contains(&mut self, ids: Vec<TrackId<'static>>);
  async fn fetch_all_playlist_tracks_and_sort(&mut self, playlist_id: PlaylistId<'static>);
  async fn create_new_playlist(&mut self, name: String, track_ids: Vec<TrackId<'static>>);
}

// Private helper methods
impl Network {
  pub(crate) async fn library_contains_uris(&self, uris: &[String]) -> anyhow::Result<Vec<bool>> {
    if uris.is_empty() {
      return Ok(Vec::new());
    }

    let batches: Vec<String> = uri_batches(uris).map(|batch| batch.join(",")).collect();
    let mut all_results = Vec::with_capacity(uris.len());
    for group in batches.chunks(LIBRARY_CONTAINS_PARALLEL) {
      let answers =
        futures::future::try_join_all(group.iter().map(|batch| contains_batch(batch.clone())))
          .await?;
      all_results.extend(answers.into_iter().flatten());
    }

    Ok(all_results)
  }

  /// Resolve liked state for bare track ids inline and merge it: the CLI's
  /// synchronous path. The `IoEvent` handler defers to the detached worker
  /// instead (it must not block the TUI's serial pump), which the CLI cannot
  /// await for a result. Failures propagate so the CLI reports them instead
  /// of acting on stale membership.
  pub(crate) async fn resolve_liked_state_now(&mut self, ids: &[String]) -> anyhow::Result<()> {
    let uris: Vec<String> = ids.iter().map(|id| format!("spotify:track:{id}")).collect();
    let is_saved_vec = self.library_contains_uris(&uris).await?;
    let mut app = self.app.lock().await;
    for (i, id) in ids.iter().enumerate() {
      match is_saved_vec.get(i) {
        Some(true) => {
          app.liked_song_ids_set_mut().insert(id.clone());
        }
        Some(false) => {
          app.liked_song_ids_set_mut().remove(id);
        }
        None => {}
      }
    }
    Ok(())
  }

  pub(super) async fn library_save_uris(&self, uris: &[String]) -> anyhow::Result<()> {
    self.library_save_uris_inner(uris).await?;
    // Spotify accepted the like: remember it so the heart cache cannot
    // show the old state on the next open.
    super::liked_cache::record_all(
      super::liked_cache::track_ids_of_uris(uris)
        .into_iter()
        .map(|id| (id, true)),
    );
    super::liked_cache::save_in_background();
    Ok(())
  }

  async fn library_save_uris_inner(&self, uris: &[String]) -> anyhow::Result<()> {
    for batch in uri_batches(uris) {
      let query = vec![("uris", batch.join(","))];
      spotify_api_request_json_for(
        Method::PUT,
        "me/library",
        &query,
        Some(json!({ "uris": batch })),
      )
      .await?;
    }
    Ok(())
  }

  async fn playlist_add_uris(&self, playlist_id: &str, uris: &[String]) -> anyhow::Result<()> {
    for batch in uris.chunks(100) {
      spotify_api_request_json_for(
        Method::POST,
        &format!("playlists/{playlist_id}/items"),
        &[],
        Some(json!({ "uris": batch })),
      )
      .await?;
    }
    Ok(())
  }

  pub(super) async fn library_remove_uris(&self, uris: &[String]) -> anyhow::Result<()> {
    self.library_remove_uris_inner(uris).await?;
    super::liked_cache::record_all(
      super::liked_cache::track_ids_of_uris(uris)
        .into_iter()
        .map(|id| (id, false)),
    );
    super::liked_cache::save_in_background();
    Ok(())
  }

  async fn library_remove_uris_inner(&self, uris: &[String]) -> anyhow::Result<()> {
    for batch in uri_batches(uris) {
      let query = vec![("uris", batch.join(","))];
      spotify_api_request_json_for(
        Method::DELETE,
        "me/library",
        &query,
        Some(json!({ "uris": batch })),
      )
      .await?;
    }
    Ok(())
  }

  pub fn spawn_saved_tracks_prefetch(
    &self,
    offset: u32,
    generation: u64,
    key: Option<super::liked_songs_cache::LikedSongsKey>,
  ) {
    let app = self.app.clone();
    let large_search_limit = self.large_search_limit;
    tokio::spawn(async move {
      prefetch_saved_tracks_page_task(app, large_search_limit, offset, generation, key).await;
    });
  }

  pub fn spawn_playlist_tracks_prefetch(
    &self,
    playlist_id: PlaylistId<'static>,
    offset: u32,
    generation: u64,
    snapshot_id: Option<String>,
  ) {
    let app = self.app.clone();
    let large_search_limit = self.large_search_limit;
    let fallbacks = self.external_playlist_fallbacks.clone();
    tokio::spawn(async move {
      prefetch_playlist_tracks_page_task(
        app,
        large_search_limit,
        playlist_id,
        offset,
        generation,
        fallbacks,
        snapshot_id,
      )
      .await;
    });
  }

  /// Serve a playlist open from the disk cache, if it holds this playlist
  /// under its current `snapshot_id`. Playlists read through the librespot
  /// fallback have no tag the Web API will give out, so they always miss.
  async fn open_playlist_from_cache(
    &self,
    playlist_id: &PlaylistId<'static>,
    generation: u64,
  ) -> CachedOpen {
    let id = playlist_id.id().to_string();
    let Some(dir) = super::playlist_cache::default_dir() else {
      return CachedOpen::Miss { snapshot_id: None };
    };
    if self
      .external_playlist_fallbacks
      .lock()
      .await
      .is_external(&id)
    {
      return CachedOpen::Miss { snapshot_id: None };
    }
    let Some(snapshot_id) = fetch_playlist_snapshot_id(&id).await else {
      return CachedOpen::Miss { snapshot_id: None };
    };

    let (load_dir, load_id, load_snapshot) = (dir, id.clone(), snapshot_id.clone());
    let cached = tokio::task::spawn_blocking(move || {
      super::playlist_cache::load(&load_dir, &load_id, &load_snapshot)
    })
    .await
    .ok()
    .flatten();
    let Some(items) = cached else {
      return CachedOpen::Miss {
        snapshot_id: Some(snapshot_id),
      };
    };

    let item_count = items.len();
    let pages = pages_from_cached_items(items);
    let mut app = self.app.lock().await;
    if app.pending_playlist_open.as_deref() == Some(id.as_str()) {
      app.pending_playlist_open = None;
    }
    // Same guard as a page from the network: a newer open or another table
    // replaced this one while the cache was read.
    if app.playlist_tracks_prefetch_generation == generation
      && app.is_playlist_track_table_active_for(playlist_id)
    {
      for page in pages {
        app.playlist_track_pages.upsert_page_by_offset(page);
      }
      app.set_playlist_tracks_to_table_continuous();
      log::debug!("playlist cache: opened {id} from cache ({item_count} items)");
    }
    CachedOpen::Served
  }
}

/// Detached body of `fetch_all_playlist_tracks_and_sort`. On a page failure
/// the error routes through `App::handle_error` exactly as the inline version
/// did.
/// Single detached worker draining [`App::liked_lookup_pending`] in 40-uri
/// batches, merging each batch into `liked_song_ids_set` as it lands. Runs off
/// the IoEvent pump: a big playlist's liked-state sweep is 8-15 batches at
/// 1-2s each, which used to head-of-line-block playback controls (a
/// skip-to-queued-track waited ~20s behind it, #386). One worker at a time
/// (`liked_lookup_worker_running`), so rapid navigation coalesces into the
/// deduped pending set instead of parallel request floods.
async fn liked_lookup_worker_task(app: Arc<Mutex<App>>) {
  const BATCH: usize = 40;
  const MAX_CONSECUTIVE_FAILURES: u32 = 3;
  let mut consecutive_failures = 0u32;
  let mut learned_any = false;
  loop {
    let (batch, epoch) = {
      let mut guard = app.lock().await;
      if guard.liked_lookup_pending.is_empty() {
        guard.liked_lookup_worker_running = false;
        drop(guard);
        if learned_any {
          super::liked_cache::save_in_background();
        }
        return;
      }
      let batch: Vec<String> = guard
        .liked_lookup_pending
        .iter()
        .take(BATCH)
        .cloned()
        .collect();
      for id in &batch {
        guard.liked_lookup_pending.remove(id);
      }
      (batch, guard.liked_state_epoch)
    };
    let uris: Vec<String> = batch
      .iter()
      .map(|id| format!("spotify:track:{id}"))
      .collect();
    // Hearts are cosmetic: at most one lookup a second, so a large table can
    // never crowd out what the user is doing.
    liked_lookup_pacer().wait().await;
    let result =
      spotify_get_typed_for::<Vec<bool>>("me/library/contains", &[("uris", uris.join(","))]).await;
    let mut guard = app.lock().await;
    match result {
      Ok(is_saved_vec) => {
        consecutive_failures = 0;
        if guard.liked_state_epoch != epoch {
          // A local like/unlike landed while this read was in flight; the
          // response may predate it, so re-read instead of applying it.
          guard.liked_lookup_pending.extend(batch);
          continue;
        }
        super::liked_cache::record_all(
          batch
            .iter()
            .zip(&is_saved_vec)
            .map(|(id, liked)| (id.as_str(), *liked)),
        );
        learned_any = true;
        for (i, id) in batch.into_iter().enumerate() {
          match is_saved_vec.get(i) {
            Some(true) => {
              guard.liked_song_ids_set_mut().insert(id);
            }
            Some(false) => {
              guard.liked_song_ids_set_mut().remove(&id);
            }
            None => {}
          }
        }
      }
      Err(_) if consecutive_failures + 1 < MAX_CONSECUTIVE_FAILURES => {
        // Transient failure: back off and retry from THIS worker, so ids
        // enqueued while the request was in flight (which saw the running
        // flag and did not spawn) are not stranded without a worker.
        guard.liked_lookup_pending.extend(batch);
        consecutive_failures += 1;
        drop(guard);
        tokio::time::sleep(std::time::Duration::from_secs(u64::from(
          2 * consecutive_failures,
        )))
        .await;
      }
      Err(e) => {
        // Persistently failing: stop rather than hammer the endpoint. The ids
        // stay pending and the next dispatch restarts the worker.
        guard.liked_lookup_pending.extend(batch);
        guard.liked_lookup_worker_running = false;
        guard.set_status_message(format!("Could not check liked track state: {e}"), 5);
        return;
      }
    }
  }
}

struct PlaylistTrackSearch {
  app: Arc<Mutex<App>>,
  fallbacks: ExternalPlaylistFallbackCache,
  playlist_id: PlaylistId<'static>,
  query: String,
}

/// Background half of `search_playlist_tracks`. Walks the playlist through the
/// disk cache (`fetch_all_playlist_items`), so a repeat search is fast; results
/// apply only while this search is still the current one.
async fn search_playlist_tracks_task(search: PlaylistTrackSearch, terms: Vec<String>) {
  let PlaylistTrackSearch {
    app,
    fallbacks,
    playlist_id,
    query,
  } = search;

  let result = fetch_all_playlist_items(&app, &playlist_id, &fallbacks).await;

  let mut app = app.lock().await;
  if !app.playlist_track_search_is_current(&playlist_id, &query) {
    return;
  }

  let items = match result {
    Ok(items) => items,
    Err(error) => {
      if !app.fail_playlist_track_search(&playlist_id, &query) {
        return;
      }
      match error {
        PlaylistPageError::UnsupportedExternal => {
          app.set_error_status_message(EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS, 8);
        }
        // A rate limit is a wait, not a failure page: the pump shows it the
        // same way.
        PlaylistPageError::Request(error) if is_rate_limited_error(&error) => {
          app.set_error_status_message(error.to_string(), 8);
        }
        PlaylistPageError::Request(error) => app.handle_error(error),
      }
      return;
    }
  };

  let matches: Vec<(TrackInfo, usize)> = items
    .into_iter()
    .filter_map(|(position, item)| match item {
      PlayableInfo::Track(track) if playlist_track_info_matches_terms(&track, &terms) => {
        Some((track, position as usize))
      }
      _ => None,
    })
    .collect();

  let match_count = matches.len();
  if app.apply_playlist_track_search_results(&playlist_id, query.clone(), matches) {
    app.set_status_message(
      format!("{match_count} playlist tracks match \"{query}\""),
      3,
    );
  }
}

async fn fetch_all_playlist_tracks_and_sort_task(
  app: Arc<Mutex<App>>,
  playlist_id: PlaylistId<'static>,
  fallbacks: ExternalPlaylistFallbackCache,
) {
  let playlist_id_string = playlist_id.id().to_string();

  {
    let mut app = app.lock().await;
    if !app.is_playlist_track_table_active_for(&playlist_id) {
      app
        .playlist_sort_fetch_in_flight
        .remove(&playlist_id_string);
      return;
    }
  }

  let items = match fetch_all_playlist_items(&app, &playlist_id, &fallbacks).await {
    Ok(items) => items,
    Err(PlaylistPageError::UnsupportedExternal) => {
      let mut app = app.lock().await;
      app
        .playlist_sort_fetch_in_flight
        .remove(&playlist_id_string);
      app.set_error_status_message(EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS, 8);
      return;
    }
    Err(PlaylistPageError::Request(error)) => {
      let mut app = app.lock().await;
      app
        .playlist_sort_fetch_in_flight
        .remove(&playlist_id_string);
      app.handle_error(error);
      return;
    }
  };

  let mut all_tracks: Vec<TrackInfo> = items
    .into_iter()
    .filter_map(|(_, item)| match item {
      PlayableInfo::Track(track) => Some(track),
      _ => None,
    })
    .collect();

  // Apply sort if any
  let mut app = app.lock().await;
  app
    .playlist_sort_fetch_in_flight
    .remove(&playlist_id_string);
  Sorter::new(app.playlist_sort).sort_tracks(&mut all_tracks);
  let _ = app.apply_sorted_playlist_tracks_if_current(&playlist_id, all_tracks);
}

/// Background half of `get_current_user_playlists`: fetch the remaining pages
/// and the librespot rootlist folder structure, then publish the complete list
/// (unless a newer refresh superseded this one). A page failure here keeps the
/// already-published pages instead of error-routing the UI.
#[allow(clippy::too_many_arguments)]
async fn finish_playlists_fetch(
  app: Arc<Mutex<App>>,
  mut all_playlists: Vec<PlaylistInfo>,
  has_more: bool,
  limit: u32,
  generation: u64,
  original_selection: (Option<String>, usize, Option<usize>),
  page_one_selection: (Option<String>, usize, Option<usize>),
  previous_complete: (
    Vec<PlaylistInfo>,
    Option<Vec<PlaylistFolderNode>>,
    Vec<PlaylistFolderItem>,
  ),
) {
  let mut pagination_error = None;
  if has_more {
    let mut offset = limit;
    loop {
      {
        let app = app.lock().await;
        if app.playlist_refresh_generation != generation {
          return;
        }
      }
      let mut attempts = 0u8;
      let page = loop {
        match spotify_get_typed_for::<Page<SimplifiedPlaylist>>(
          "me/playlists",
          &[("limit", limit.to_string()), ("offset", offset.to_string())],
        )
        .await
        {
          Ok(page) => break Some(page),
          Err(e) if attempts < 2 => {
            attempts += 1;
            log::warn!("playlist page fetch failed at offset {offset}; retry {attempts}/2: {e}");
            tokio::time::sleep(Duration::from_millis(250 * u64::from(attempts))).await;
          }
          Err(e) => {
            pagination_error = Some(e.to_string());
            break None;
          }
        }
      };
      let Some(page) = page else { break };
      if page.items.is_empty() {
        break;
      }
      all_playlists.extend(page.items.iter().map(PlaylistInfo::from_simplified));
      if page.next.is_none() {
        break;
      }
      offset += limit;
    }
  }

  if let Some(error) = pagination_error {
    log::warn!("playlist refresh incomplete: {error}");
    let mut app = app.lock().await;
    if app.playlist_refresh_generation != generation {
      return;
    }
    let (previous_playlists, previous_nodes, previous_items) = previous_complete;
    if !previous_playlists.is_empty() {
      *app.all_playlists_mut() = previous_playlists;
      *app.playlist_folder_nodes_mut() = previous_nodes;
      *app.playlist_folder_items_mut() = previous_items;
      reconcile_playlist_selection(
        &mut app,
        original_selection.0.as_deref(),
        original_selection.1,
        original_selection.2,
      );
    }
    app.set_status_message("Playlist refresh incomplete; kept the previous list.", 8);
    return;
  }

  #[cfg(feature = "streaming")]
  let folder_nodes = {
    let streaming_player = {
      let app = app.lock().await;
      app.streaming_player.clone()
    };
    let nodes = fetch_rootlist_folders(streaming_player.clone()).await;
    // Spotify-owned playlists (Release Radar, daylist, the Mixes, editorial
    // lists) sit in the rootlist, but the Development Mode Web API leaves
    // them out of `me/playlists`. Resolve those rootlist-only ids through the
    // librespot session so they show up in the sidebar. Their owner is
    // Spotify, so opening one takes the external-playlist librespot fallback.
    if let (Some(nodes), Some(player)) = (nodes.as_ref(), streaming_player) {
      let missing = rootlist_only_playlist_ids(nodes, &all_playlists);
      if !missing.is_empty() {
        let resolved = resolve_rootlist_only_playlists(&player.session(), &missing).await;
        log::debug!(
          "rootlist-only playlists: {} missing from the Web API, {} resolved via librespot",
          missing.len(),
          resolved.len()
        );
        all_playlists.extend(resolved);
      }
    }
    nodes
  };
  #[cfg(not(feature = "streaming"))]
  let folder_nodes: Option<Vec<PlaylistFolderNode>> = None;

  let folder_items = if let Some(ref nodes) = folder_nodes {
    structurize_playlist_folders(nodes, &all_playlists)
  } else {
    build_flat_playlist_items(&all_playlists)
  };

  let mut app = app.lock().await;
  if app.playlist_refresh_generation != generation {
    return;
  }
  // Restore the original selection only if the user has not navigated since
  // page 1 was published. Otherwise their newer choice wins.
  let current_selection = (
    app.get_selected_playlist_id(),
    app.current_playlist_folder_id,
    app.view.selected_playlist_index,
  );
  let preferred = if current_selection == page_one_selection {
    original_selection
  } else {
    current_selection
  };

  *app.all_playlists_mut() = all_playlists;
  app
    .plugin_data_generations
    .bump(crate::core::app::PluginDataKind::Playlists);
  *app.playlist_folder_nodes_mut() = folder_nodes;
  *app.playlist_folder_items_mut() = folder_items;

  reconcile_playlist_selection(&mut app, preferred.0.as_deref(), preferred.1, preferred.2);

  maybe_prompt_community_pin(&mut app);
}

/// Offer the one-time community-pin hide prompt once the full playlist list is
/// known. A user who already follows the community playlist never sees it (no
/// pin to explain); if they later unfollow, the pin appears for the first time
/// and this can fire then — intended, not a bug.
fn maybe_prompt_community_pin(app: &mut App) {
  if app.active_source == Source::Spotify
    && !app.runtime_state.community_pin_prompt_shown
    && app.user_config.behavior.pin_community_playlist
    && !app.follows_community_playlist()
    && app.get_current_route().id != RouteId::CommunityPinPrompt
  {
    // Mark shown at push time, not only on dismiss, so paths that close the
    // prompt without going through its handler (the back key, mouse-interactive
    // layout dismissal) can never make it reappear.
    app.mark_community_pin_prompt_shown();
    app.push_navigation_stack(RouteId::CommunityPinPrompt, ActiveBlock::CommunityPinPrompt);
  }
}

impl LibraryNetwork for Network {
  async fn get_current_user_playlists(&mut self) {
    let original_selection = {
      let app = self.app.lock().await;
      (
        app.get_selected_playlist_id(),
        app.current_playlist_folder_id,
        app.view.selected_playlist_index,
      )
    };

    let limit = 50u32;

    // Page 1 first: publish it immediately so the sidebar is usable, then
    // fetch the remaining pages and the rootlist folder structure in a
    // background task instead of blocking the pump for the whole pagination
    // (each page pays the pacing floor; 500 playlists used to block startup
    // dispatches for seconds).
    //
    // Always use the compat path: parses raw JSON to serde_json::Value first,
    // which silently deduplicates keys (last-wins). This handles the known Spotify
    // API bug where "items" appears twice in the same JSON object.
    let first_page = match spotify_get_typed_for::<Page<SimplifiedPlaylist>>(
      "me/playlists",
      &[("limit", limit.to_string()), ("offset", "0".to_string())],
    )
    .await
    {
      Ok(page) => page,
      Err(e) => {
        let mut app = self.app.lock().await;
        app.pending_playlist_track_search = None;
        drop(app);
        self.handle_error(anyhow!(e)).await;
        return;
      }
    };

    // Convert to source-agnostic domain types at the network boundary.
    let first_items: Vec<PlaylistInfo> = first_page
      .items
      .iter()
      .map(PlaylistInfo::from_simplified)
      .collect();
    let has_more = first_page.next.is_some() && !first_page.items.is_empty();
    let mapped_first = map_page(&first_page, PlaylistInfo::from_simplified);

    let (generation, page_one_selection, previous_complete) = {
      let mut app = self.app.lock().await;
      let had_previous_complete = !app.all_playlists().is_empty();
      // Snapshot the existing complete list only when background pagination can
      // fail partway (`has_more`) and there is something worth restoring. A
      // single-page refresh or a first-ever load never uses this, so skip the
      // clone in the common case.
      let previous_complete = if has_more && had_previous_complete {
        (
          app.all_playlists().clone(),
          app.playlist_folder_nodes().clone(),
          app.playlist_folder_items().clone(),
        )
      } else {
        (Vec::new(), None, Vec::new())
      };
      app.playlist_refresh_generation = app.playlist_refresh_generation.wrapping_add(1);
      *app.playlists_mut() = Some(mapped_first);
      let selected_is_in_first_page = original_selection.0.as_ref().is_some_and(|selected| {
        first_items
          .iter()
          .any(|playlist| playlist.id.as_deref() == Some(selected.as_str()))
      });
      if !had_previous_complete || original_selection.0.is_none() || selected_is_in_first_page {
        *app.all_playlists_mut() = first_items.clone();
        app
          .plugin_data_generations
          .bump(crate::core::app::PluginDataKind::Playlists);
        *app.playlist_folder_items_mut() = build_flat_playlist_items(&first_items);
        reconcile_playlist_selection(
          &mut app,
          original_selection.0.as_deref(),
          original_selection.1,
          original_selection.2,
        );
      }
      let page_one_selection = (
        app.get_selected_playlist_id(),
        app.current_playlist_folder_id,
        app.view.selected_playlist_index,
      );
      (
        app.playlist_refresh_generation,
        page_one_selection,
        previous_complete,
      )
    };

    let app = Arc::clone(&self.app);
    tokio::spawn(async move {
      finish_playlists_fetch(
        app,
        first_items,
        has_more,
        limit,
        generation,
        original_selection,
        page_one_selection,
        previous_complete,
      )
      .await;
    });
  }

  async fn get_playlist_tracks(&mut self, playlist_id: PlaylistId<'static>, playlist_offset: u32) {
    // A fresh open re-downloads the fallback contents instead of showing the
    // first snapshot all session. The External classification is kept.
    #[cfg(feature = "streaming")]
    if playlist_offset == 0 {
      self
        .external_playlist_fallbacks
        .lock()
        .await
        .clear_contents(playlist_id.id());
    }

    let generation = {
      let app = self.app.lock().await;
      app.playlist_tracks_prefetch_generation
    };

    // A fresh open is served from the disk cache when the playlist has not
    // changed since it was stored: one small `snapshot_id` request instead of
    // one request per page. On a miss the tag rides the page chain, which
    // stores the playlist once its last page lands.
    let open_snapshot = if playlist_offset == 0 {
      match self
        .open_playlist_from_cache(&playlist_id, generation)
        .await
      {
        CachedOpen::Served => return,
        CachedOpen::Miss { snapshot_id } => snapshot_id,
      }
    } else {
      None
    };

    match fetch_playlist_tracks_page(
      &self.app,
      &playlist_id,
      playlist_offset,
      self.large_search_limit,
      &self.external_playlist_fallbacks,
    )
    .await
    {
      Ok(playlist_tracks) => {
        let mut app = self.app.lock().await;
        app
          .playlist_tracks_prefetch_in_flight
          .remove(&playlist_offset);
        if app.pending_playlist_open.as_deref() == Some(playlist_id.id()) {
          app.pending_playlist_open = None;
        }
        if app.playlist_tracks_prefetch_generation != generation
          || !app.is_playlist_track_table_active_for(&playlist_id)
        {
          return;
        }

        let playlist_tracks_index = app
          .playlist_track_pages
          .upsert_page_by_offset(playlist_tracks.into_domain());
        app.set_playlist_tracks_to_table_continuous();

        let next_offset = app.next_missing_playlist_tracks_offset(playlist_tracks_index);
        let generation = app.playlist_tracks_prefetch_generation;
        // Navigation happens in the handler when the user opens the playlist
        // (`App::open_playlist_tracks`), not here on response arrival.
        if next_offset.is_none() {
          // A one-page playlist: no chain follows, so store it here.
          store_completed_playlist(
            &app.playlist_track_pages.pages,
            &playlist_id,
            open_snapshot.as_deref(),
          );
        }
        drop(app);

        if let Some(next_offset) = next_offset {
          self.spawn_playlist_tracks_prefetch(playlist_id, next_offset, generation, open_snapshot);
        }
      }
      Err(PlaylistPageError::UnsupportedExternal) => {
        let mut app = self.app.lock().await;
        app
          .playlist_tracks_prefetch_in_flight
          .remove(&playlist_offset);
        if app.pending_playlist_open.as_deref() == Some(playlist_id.id()) {
          app.pending_playlist_open = None;
        }
        app.set_error_status_message(EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS, 8);
      }
      Err(PlaylistPageError::Request(error)) => {
        let mut app = self.app.lock().await;
        app
          .playlist_tracks_prefetch_in_flight
          .remove(&playlist_offset);
        if app.pending_playlist_open.as_deref() == Some(playlist_id.id()) {
          app.pending_playlist_open = None;
        }
        drop(app);
        self.handle_error(error).await;
      }
    }
  }

  async fn search_playlist_tracks(&mut self, playlist_id: PlaylistId<'static>, query: String) {
    let terms = playlist_track_search_terms(&query);
    if terms.is_empty() {
      let mut app = self.app.lock().await;
      app.clear_playlist_track_filter();
      return;
    }

    // The walk runs detached like the sort
    // walk; a newer query or another table supersedes it page by page.
    let search = PlaylistTrackSearch {
      app: Arc::clone(&self.app),
      fallbacks: self.external_playlist_fallbacks.clone(),
      playlist_id,
      query,
    };
    tokio::spawn(search_playlist_tracks_task(search, terms));
  }

  async fn get_current_user_saved_tracks(&mut self, offset: Option<u32>) {
    let requested_offset = offset.unwrap_or(0);
    let generation = {
      let app = self.app.lock().await;
      app.saved_tracks_prefetch_generation
    };

    let mut query = vec![("limit", self.large_search_limit.to_string())];
    if let Some(offset) = offset {
      query.push(("offset", offset.to_string()));
    }

    match spotify_get_typed_for::<Page<rspotify::model::SavedTrack>>("me/tracks", &query).await {
      Ok(saved_tracks) => {
        let mut app = self.app.lock().await;
        app
          .saved_tracks_prefetch_in_flight
          .remove(&requested_offset);
        if app.saved_tracks_prefetch_generation != generation {
          return;
        }

        populate_liked_song_ids_from_saved_tracks(app.liked_song_ids_set_mut(), &saved_tracks);
        let domain_page =
          crate::infra::network::mapping::map_page(&saved_tracks, |st| TrackInfo::from(&st.track));

        // A fresh open: the first page doubles as the cache check, so a hit
        // costs no request beyond the one just made.
        let key = offset.is_none().then(|| liked_songs_key(&saved_tracks));
        if let Some(key) = &key {
          if domain_page.next.is_some() {
            drop(app);
            let cached = load_liked_songs(key.clone()).await;
            app = self.app.lock().await;
            if app.saved_tracks_prefetch_generation != generation {
              return;
            }
            if let Some(tracks) = cached {
              apply_cached_liked_songs(&mut app, domain_page, tracks);
              return;
            }
          }
        }

        let saved_tracks_index = app.saved_tracks_mut().upsert_page_by_offset(domain_page);
        app.refresh_saved_tracks_table();
        app
          .plugin_data_generations
          .bump(crate::core::app::PluginDataKind::SavedTracks);

        let next_offset = app.next_missing_saved_tracks_offset(saved_tracks_index);
        let generation = app.saved_tracks_prefetch_generation;
        if next_offset.is_none() {
          // Everything fit on one page: store it now, no chain follows.
          store_liked_songs_if_complete(&app, key.as_ref());
        }
        drop(app);

        if let Some(next_offset) = next_offset {
          self.spawn_saved_tracks_prefetch(next_offset, generation, key);
        }
      }
      Err(e) => {
        let mut app = self.app.lock().await;
        app
          .saved_tracks_prefetch_in_flight
          .remove(&requested_offset);
        drop(app);
        self.handle_error(anyhow!(e)).await;
      }
    }
  }

  async fn get_current_user_saved_albums(&mut self, offset: Option<u32>) {
    let mut query = vec![("limit", self.large_search_limit.to_string())];
    if let Some(offset) = offset {
      query.push(("offset", offset.to_string()));
    }

    match spotify_get_typed_for::<Page<rspotify::model::SavedAlbum>>("me/albums", &query).await {
      Ok(saved_albums) => {
        let mut app = self.app.lock().await;
        if !saved_albums.items.is_empty() {
          let domain_page = crate::infra::network::mapping::map_page(
            &saved_albums,
            crate::infra::network::mapping::saved_album_info,
          );
          app.store_saved_albums_page(domain_page);
        }
        // Bump even on an empty page: completion is the signal plugin data
        // requests wait on, and an empty library never writes a page.
        app
          .plugin_data_generations
          .bump(crate::core::app::PluginDataKind::SavedAlbums);
      }
      Err(e) => {
        self.handle_error(anyhow!(e)).await;
      }
    }
  }

  async fn current_user_saved_albums_contains(&mut self, album_ids: Vec<AlbumId<'static>>) {
    let uris: Vec<String> = album_ids
      .iter()
      .map(|id| format!("spotify:album:{}", id.id()))
      .collect();

    match self.library_contains_uris(&uris).await {
      Ok(is_saved_vec) => {
        let mut app = self.app.lock().await;
        for (i, id) in album_ids.iter().enumerate() {
          if let Some(is_saved) = is_saved_vec.get(i) {
            if *is_saved {
              app.saved_album_ids_set_mut().insert(id.id().to_string());
            } else if app.saved_album_ids_set().contains(id.id()) {
              app.saved_album_ids_set_mut().remove(id.id());
            }
          };
        }
      }
      Err(e) => {
        self.handle_error(anyhow!(e)).await;
      }
    }
  }

  async fn current_user_saved_album_delete(&mut self, album_id: AlbumId<'static>) {
    let uris = vec![format!("spotify:album:{}", album_id.id())];
    match self.library_remove_uris(&uris).await {
      Ok(_) => {
        let mut app = self.app.lock().await;
        app.saved_album_ids_set_mut().remove(album_id.id());
        app.dispatch(IoEvent::GetCurrentUserSavedAlbums(None));
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn current_user_saved_album_add(&mut self, album_id: AlbumId<'static>) {
    let uris = vec![format!("spotify:album:{}", album_id.id())];
    match self.library_save_uris(&uris).await {
      Ok(_) => {
        let mut app = self.app.lock().await;
        app
          .saved_album_ids_set_mut()
          .insert(album_id.id().to_string());
        app.dispatch(IoEvent::GetCurrentUserSavedAlbums(None));
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn current_user_saved_shows_contains(&mut self, show_ids: Vec<ShowId<'static>>) {
    let uris: Vec<String> = show_ids
      .iter()
      .map(|id| format!("spotify:show:{}", id.id()))
      .collect();
    match self.library_contains_uris(&uris).await {
      Ok(is_saved_vec) => {
        let mut app = self.app.lock().await;
        for (i, id) in show_ids.iter().enumerate() {
          if let Some(is_saved) = is_saved_vec.get(i) {
            if *is_saved {
              app.saved_show_ids_set_mut().insert(id.id().to_string());
            } else if app.saved_show_ids_set().contains(id.id()) {
              app.saved_show_ids_set_mut().remove(id.id());
            }
          };
        }
      }
      Err(e) => {
        self.handle_error(anyhow!(e)).await;
      }
    }
  }

  async fn current_user_saved_shows_delete(&mut self, show_id: ShowId<'static>) {
    let uris = vec![format!("spotify:show:{}", show_id.id())];
    match self.library_remove_uris(&uris).await {
      Ok(_) => {
        let mut app = self.app.lock().await;
        app.saved_show_ids_set_mut().remove(show_id.id());
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn current_user_saved_shows_add(&mut self, show_id: ShowId<'static>) {
    let uris = vec![format!("spotify:show:{}", show_id.id())];
    match self.library_save_uris(&uris).await {
      Ok(_) => {
        let mut app = self.app.lock().await;
        app
          .saved_show_ids_set_mut()
          .insert(show_id.id().to_string());
        // The saved-shows list gets the new show at its top.
        app.dispatch(IoEvent::GetCurrentUserSavedShows(None));
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn get_current_user_saved_shows(&mut self, offset: Option<u32>) {
    let mut query = vec![("limit", self.large_search_limit.to_string())];
    if let Some(offset) = offset {
      query.push(("offset", offset.to_string()));
    }

    match spotify_get_typed_for::<Page<rspotify::model::show::Show>>("me/shows", &query).await {
      Ok(saved_shows) => {
        let mut app = self.app.lock().await;
        if !saved_shows.items.is_empty() {
          let domain_page =
            crate::infra::network::mapping::map_page(&saved_shows, |s| ShowInfo::from(&s.show));
          app.store_saved_shows_page(domain_page);
        }
        // Bump even on an empty page (see saved-albums note above).
        app
          .plugin_data_generations
          .bump(crate::core::app::PluginDataKind::SavedShows);
      }
      Err(e) => {
        self.handle_error(anyhow!(e)).await;
      }
    }
  }

  async fn user_follow_playlist(
    &mut self,
    _playlist_owner_id: UserId<'static>,
    playlist_id: PlaylistId<'static>,
    _is_public: Option<bool>,
  ) {
    match self.library_save_uris(&[playlist_id.uri()]).await {
      Ok(_) => {
        // Optimistic update handled in handler or next refresh
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn user_unfollow_playlist(
    &mut self,
    _user_id: UserId<'static>,
    playlist_id: PlaylistId<'static>,
  ) {
    match self.library_remove_uris(&[playlist_id.uri()]).await {
      Ok(_) => self.app.lock().await.dispatch(IoEvent::GetPlaylists),
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn add_track_to_playlist(
    &mut self,
    playlist_id: PlaylistId<'static>,
    track_id: TrackId<'static>,
  ) {
    match self
      .playlist_add_uris(playlist_id.id(), &[track_id.uri()])
      .await
    {
      Ok(_) => {
        let status_message = {
          let mut app = self.app.lock().await;
          let playlist_name = app
            .all_playlists()
            .iter()
            .find(|playlist| playlist.id.as_deref() == Some(playlist_id.id()))
            .map(|playlist| playlist.name.clone());

          // The added track lands at the end of the playlist, past every page
          // the opened playlist already cached, so those pages are stale.
          // `is_complete` reads false here, making the next search or sort walk
          // Spotify again. This runs on any screen: an add from search results
          // or an agent leaves the cached pages behind too.
          app.mark_playlist_track_pages_stale(&playlist_id);

          if app.is_current_route_playlist_track_table_for(&playlist_id) {
            let playlist_offset = app.playlist_offset;
            // Refresh the page the user is looking at. It doesn't make the
            // rest of the cache current, which is why the mark above stays.
            app.dispatch(IoEvent::GetPlaylistItems(
              playlist_id.id().to_string(),
              playlist_offset,
            ));
          }

          playlist_name
            .map(|name| format!("Added to {}", name))
            .unwrap_or_else(|| "Added to playlist".to_string())
        };
        self.show_status_message(status_message, 3).await;
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn remove_track_from_playlist_at_position(
    &mut self,
    playlist_id: PlaylistId<'static>,
    track_id: TrackId<'static>,
    position: usize,
  ) {
    let body = json!({
        "items": [{
            "uri": format!("spotify:track:{}", track_id.id()),
            "positions": [position]
        }]
    });

    match spotify_api_request_json_for(
      Method::DELETE,
      &format!("playlists/{}/items", playlist_id.id()),
      &[],
      Some(body),
    )
    .await
    {
      Ok(_) => {
        {
          let mut app = self.app.lock().await;
          // The cached pages still hold the removed track. Mark them stale so
          // a search or sort walks Spotify again, and reload the open table:
          // every later row moved up one position.
          app.mark_playlist_track_pages_stale(&playlist_id);
          app.reload_playlist_tracks(playlist_id);
        }
        self
          .show_status_message("Removed from playlist".to_string(), 3)
          .await;
      }
      Err(e) => self.handle_error(anyhow!(e)).await,
    }
  }

  async fn toggle_save_track(&mut self, track_id: rspotify::model::idtypes::PlayableId<'static>) {
    let id_str = match &track_id {
      PlayableId::Track(id) => id.id(),
      PlayableId::Episode(id) => id.id(),
    };
    let uri = match &track_id {
      PlayableId::Track(id) => format!("spotify:track:{}", id.id()),
      PlayableId::Episode(id) => format!("spotify:episode:{}", id.id()),
    };

    let is_liked = {
      let app = self.app.lock().await;
      app.liked_song_ids_set().contains(id_str)
    };

    if is_liked {
      if let Err(e) = self.library_remove_uris(&[uri]).await {
        self.handle_error(anyhow!(e)).await;
      } else {
        let mut app = self.app.lock().await;
        app.liked_song_ids_set_mut().remove(id_str);
        app.liked_state_epoch = app.liked_state_epoch.wrapping_add(1);
      }
    } else if let Err(e) = self.library_save_uris(&[uri]).await {
      self.handle_error(anyhow!(e)).await;
    } else {
      let mut app = self.app.lock().await;
      app.liked_song_ids_set_mut().insert(id_str.to_string());
      app.liked_state_epoch = app.liked_state_epoch.wrapping_add(1);
    }
  }

  async fn current_user_saved_tracks_contains(&mut self, ids: Vec<TrackId<'static>>) {
    // Enqueue and let the single detached worker resolve: these lookups are
    // UI garnish and must never head-of-line-block playback events on the
    // serial pump (see `liked_lookup_worker_task`).
    // Hearts learned within the last day are applied at once from the heart
    // cache; only the rest are asked for. Without this, every table rebuild
    // re-asked for every track, which was most of the app's API traffic.
    let ids: Vec<String> = ids.iter().map(|id| id.id().to_string()).collect();
    {
      // With the whole of Liked Songs loaded, the liked set already answers
      // every track: anything outside it is not liked. Likes made here update
      // the set at once; a change on another device shows once Liked Songs is
      // opened again.
      let app = self.app.lock().await;
      if app.library().saved_tracks.is_complete() {
        return;
      }
    }
    let (known, unknown) = super::liked_cache::partition_known(ids);
    let spawn = {
      let mut app = self.app.lock().await;
      if !known.is_empty() {
        let liked = app.liked_song_ids_set_mut();
        for (id, is_liked) in known {
          if is_liked {
            liked.insert(id);
          } else {
            liked.remove(&id);
          }
        }
      }
      for id in unknown {
        app.liked_lookup_pending.insert(id);
      }
      if app.liked_lookup_pending.is_empty() || app.liked_lookup_worker_running {
        false
      } else {
        app.liked_lookup_worker_running = true;
        true
      }
    };
    if !spawn {
      return;
    }
    let app = Arc::clone(&self.app);
    tokio::spawn(async move {
      liked_lookup_worker_task(app).await;
    });
  }

  async fn fetch_all_playlist_tracks_and_sort(&mut self, playlist_id: PlaylistId<'static>) {
    // The full pagination pays the pacing floor per page (a 3,000-track
    // playlist is 60 pages, 15s+), so it runs detached instead of
    // head-of-line-blocking playback controls and polling on the serial pump.
    // `apply_sorted_playlist_tracks_if_current` already guards a stale result.
    {
      let mut app = self.app.lock().await;
      if !app
        .playlist_sort_fetch_in_flight
        .insert(playlist_id.id().to_string())
      {
        return;
      }
    }
    let app = Arc::clone(&self.app);
    let fallbacks = self.external_playlist_fallbacks.clone();
    tokio::spawn(async move {
      fetch_all_playlist_tracks_and_sort_task(app, playlist_id, fallbacks).await;
    });
  }

  async fn create_new_playlist(&mut self, name: String, track_ids: Vec<TrackId<'static>>) {
    // Use raw API call to avoid rspotify deserializing FullPlaylist, which crashes when
    // Spotify returns a duplicate "items" key in the response (known API migration bug).
    let create_body = json!({
      "name": name,
      "public": false,
      "collaborative": false,
      "description": "Created with spotatui"
    });
    let playlist_value = match spotify_api_request_json_for(
      Method::POST,
      "me/playlists",
      &[],
      Some(create_body),
    )
    .await
    {
      Ok(v) => v,
      Err(e) => {
        self.handle_error(e).await;
        return;
      }
    };

    let playlist_id_str = match playlist_value.get("id").and_then(|v| v.as_str()) {
      Some(id) => id.to_string(),
      None => {
        self
          .show_status_message("Playlist created but could not get its ID".to_string(), 4)
          .await;
        return;
      }
    };

    let playlist_id = match PlaylistId::from_id(playlist_id_str) {
      Ok(id) => id.into_static(),
      Err(e) => {
        self.handle_error(anyhow!(e)).await;
        return;
      }
    };

    if !track_ids.is_empty() {
      let uris: Vec<String> = track_ids.iter().map(|id| id.uri()).collect();
      if let Err(e) = self.playlist_add_uris(playlist_id.id(), &uris).await {
        self.handle_error(e).await;
        return;
      }
    }

    // Refresh playlists
    {
      let mut app = self.app.lock().await;
      app.dispatch(IoEvent::GetPlaylists);
    }

    let status = format!("Playlist \"{}\" created!", name);
    self.show_status_message(status, 4).await;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use chrono::{Duration as ChronoDuration, Utc};
  use rspotify::model::{artist::SimplifiedArtist, track::FullTrack};
  use std::collections::{HashMap, HashSet};

  async fn playlist_search_network(
    base_url: String,
    query: &str,
  ) -> (
    Network,
    Arc<Mutex<App>>,
    crate::infra::network::requests::fake::FakeSpotify,
  ) {
    use crate::core::app::TrackTableContext;
    use crate::infra::network::metadata::tests::fake_spotify_login;
    let mut app = App::default();
    app.reset_playlist_tracks_view(search_playlist_id(), TrackTableContext::MyPlaylists);
    app.pending_playlist_track_search = Some(query.to_string());
    let app = Arc::new(Mutex::new(app));
    let (spotify, _fake) = fake_spotify_login(base_url);
    let network = Network::new(Some(spotify), &app);
    (network, app, _fake)
  }

  fn search_playlist_id() -> PlaylistId<'static> {
    PlaylistId::from_id("37i9dQZF1DX4WYpdgoIcn6")
      .unwrap()
      .into_static()
  }

  #[tokio::test]
  async fn a_playlist_search_returns_before_its_first_page_lands() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
      let (_stream, _) = listener.accept().await.unwrap();
      tokio::time::sleep(Duration::from_secs(10)).await;
    });
    let (mut network, app, _fake) = playlist_search_network(base_url, "song").await;

    tokio::time::timeout(
      Duration::from_secs(1),
      network.search_playlist_tracks(search_playlist_id(), "song".to_string()),
    )
    .await
    .expect("the search must not hold the pump");

    assert_eq!(
      app.lock().await.pending_playlist_track_search.as_deref(),
      Some("song")
    );
    server.abort();
  }

  #[tokio::test]
  async fn a_failed_playlist_search_ends_its_pending_query_and_reports_the_error() {
    use crate::infra::network::metadata::tests::read_http_request;
    use tokio::io::AsyncWriteExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
      let (mut stream, _) = listener.accept().await.unwrap();
      let _ = read_http_request(&mut stream).await;
      let body = r#"{"error":{"status":500,"message":"boom"}}"#;
      let response = format!(
        "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
      );
      stream.write_all(response.as_bytes()).await.unwrap();
    });
    let (mut network, app, _fake) = playlist_search_network(base_url, "song").await;

    network
      .search_playlist_tracks(search_playlist_id(), "song".to_string())
      .await;
    tokio::time::timeout(Duration::from_secs(5), async {
      while app.lock().await.pending_playlist_track_search.is_some() {
        tokio::time::sleep(Duration::from_millis(10)).await;
      }
    })
    .await
    .expect("the failed search must end");
    server.await.unwrap();

    let app = app.lock().await;
    assert_eq!(app.get_current_route().id, RouteId::Error);
    assert!(app.active_playlist_track_filter.is_none());
  }

  #[tokio::test]
  async fn a_completed_playlist_fetch_publishes_the_full_list_under_a_new_library_revision() {
    use crate::core::app::DisplayDomain;
    use crate::core::test_helpers::playlist_info;
    let mut app = App::default();
    app.user_config.behavior.pin_community_playlist = false;
    let generation = app.playlist_refresh_generation;
    let before = app.display_revisions().get(DisplayDomain::Library);
    let app = Arc::new(Mutex::new(app));

    finish_playlists_fetch(
      Arc::clone(&app),
      vec![playlist_info("00000000000000000000p1", "P1", "me", false)],
      false,
      50,
      generation,
      (None, 0, None),
      (None, 0, None),
      (Vec::new(), None, Vec::new()),
    )
    .await;

    let app = app.lock().await;
    assert!(app.display_revisions().get(DisplayDomain::Library) > before);
    assert_eq!(app.all_playlists()[0].name, "P1");
    assert_eq!(app.playlist_folder_items().len(), 1);
  }

  #[test]
  fn playlist_access_classifies_by_owner_only() {
    use crate::core::test_helpers::{playlist_info, user_info};

    let mut app = App::default();
    *app.user_mut() = Some(user_info("me"));
    let mut external = playlist_info("external", "External", "other", false);
    external.public = Some(true);
    *app.all_playlists_mut() = vec![
      playlist_info("owned", "Owned", "me", false),
      // Another user's collaborative playlist: the flag does not prove the
      // current user collaborates, so this stays External (regression: a
      // followed collaborative playlist hit the error page on 403).
      playlist_info("collab", "Collaborative", "other", true),
    ];
    app.set_search_results(crate::core::app::SearchResult {
      playlists: Some(Paged {
        items: vec![external],
        ..Default::default()
      }),
      ..Default::default()
    });

    assert_eq!(playlist_access(&app, "owned"), PlaylistAccess::Owned);
    assert_eq!(playlist_access(&app, "collab"), PlaylistAccess::External);
    assert_eq!(playlist_access(&app, "external"), PlaylistAccess::External);
    assert_eq!(playlist_access(&app, "missing"), PlaylistAccess::Unknown);
  }

  #[test]
  fn playlist_access_without_user_or_owner_is_unknown() {
    use crate::core::test_helpers::{playlist_info, user_info};

    // No current user: even an owned-looking playlist cannot be proven.
    let mut app = App::default();
    *app.all_playlists_mut() = vec![playlist_info("owned", "Owned", "me", false)];
    assert_eq!(playlist_access(&app, "owned"), PlaylistAccess::Unknown);

    // Current user but playlist without owner metadata.
    let mut app = App::default();
    *app.user_mut() = Some(user_info("me"));
    let mut ownerless = playlist_info("ownerless", "Ownerless", "me", false);
    ownerless.owner_id = None;
    *app.all_playlists_mut() = vec![ownerless];
    assert_eq!(playlist_access(&app, "ownerless"), PlaylistAccess::Unknown);
  }

  #[allow(deprecated)]
  fn full_track(id: &str) -> FullTrack {
    FullTrack {
      album: rspotify::model::album::SimplifiedAlbum {
        name: "Album".to_string(),
        ..Default::default()
      },
      artists: vec![SimplifiedArtist {
        name: "Artist".to_string(),
        ..Default::default()
      }],
      available_markets: Vec::new(),
      disc_number: 1,
      duration: ChronoDuration::milliseconds(180_000),
      explicit: false,
      external_ids: HashMap::new(),
      external_urls: HashMap::new(),
      href: None,
      id: Some(TrackId::from_id(id).unwrap().into_static()),
      is_local: false,
      is_playable: Some(true),
      linked_from: None,
      restrictions: None,
      name: format!("Track {id}"),
      popularity: 50,
      preview_url: None,
      track_number: 1,
      r#type: rspotify::model::Type::Track,
    }
  }

  fn saved_track(id: &str) -> SavedTrack {
    SavedTrack {
      added_at: Utc::now(),
      track: full_track(id),
    }
  }

  fn saved_tracks_page(offset: u32, limit: u32, has_next: bool) -> Page<SavedTrack> {
    let ids = match offset {
      0 => vec!["0000000000000000000001", "0000000000000000000002"],
      20 => vec!["0000000000000000000003", "0000000000000000000004"],
      40 => vec!["0000000000000000000005", "0000000000000000000006"],
      _ => vec!["0000000000000000000007", "0000000000000000000008"],
    };

    Page {
      href: "https://example.com/me/tracks".to_string(),
      items: ids.into_iter().map(saved_track).collect(),
      limit,
      next: has_next.then(|| "https://example.com/me/tracks?next".to_string()),
      offset,
      previous: None,
      total: 60,
    }
  }

  /// An API that answers one request with `page`, and its base URL.
  async fn serve_saved_tracks_page(
    page: Page<SavedTrack>,
  ) -> (String, tokio::task::JoinHandle<()>) {
    use crate::infra::network::metadata::tests::read_http_request;
    use tokio::io::AsyncWriteExt;
    let body = serde_json::to_string(&page).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
      let (mut stream, _) = listener.accept().await.unwrap();
      let _ = read_http_request(&mut stream).await;
      let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
      );
      stream.write_all(response.as_bytes()).await.unwrap();
    });
    (base_url, server)
  }

  /// Run one `GetCurrentSavedTracks(None)` against an API that answers with the first saved-tracks page.
  async fn land_first_saved_tracks_page(app: App) -> Arc<Mutex<App>> {
    use crate::infra::network::metadata::tests::fake_spotify_login;
    let (base_url, server) = serve_saved_tracks_page(saved_tracks_page(0, 20, false)).await;
    let app = Arc::new(Mutex::new(app));
    let (spotify, _fake) = fake_spotify_login(base_url);
    let mut network = Network::new(Some(spotify), &app);
    network
      .handle_network_event(IoEvent::GetCurrentSavedTracks(None))
      .await;
    server.await.unwrap();
    app
  }

  #[tokio::test]
  async fn a_liked_songs_page_fills_the_table_only_while_it_shows_liked_songs() {
    use crate::core::app::TrackTableContext;
    let mut app = App::default();
    app.reset_saved_tracks_view();
    let app = land_first_saved_tracks_page(app).await;
    assert_eq!(app.lock().await.track_table.tracks.len(), 2);

    // A playlist opened before the page landed keeps the table; the cache still fills.
    let mut app = App::default();
    app.reset_saved_tracks_view();
    app.reset_playlist_tracks_view(search_playlist_id(), TrackTableContext::MyPlaylists);
    let app = land_first_saved_tracks_page(app).await;
    let app = app.lock().await;
    assert_eq!(
      app.track_table.context,
      Some(TrackTableContext::MyPlaylists)
    );
    assert!(app.track_table.tracks.is_empty());
    assert_eq!(app.saved_tracks_prefix().0.len(), 2);
  }

  #[tokio::test]
  async fn a_prefetched_liked_songs_page_keeps_an_open_playlist_and_moves_the_index() {
    use crate::core::app::TrackTableContext;
    use crate::infra::network::metadata::tests::fake_spotify_login;
    let mut app = App::default();
    app.reset_saved_tracks_view();
    let first = crate::infra::network::mapping::map_page(&saved_tracks_page(0, 20, true), |st| {
      TrackInfo::from(&st.track)
    });
    app.saved_tracks_mut().upsert_page_by_offset(first);
    app.reset_playlist_tracks_view(search_playlist_id(), TrackTableContext::MyPlaylists);
    let generation = app.saved_tracks_prefetch_generation;
    let app = Arc::new(Mutex::new(app));
    let (base_url, server) = serve_saved_tracks_page(saved_tracks_page(20, 20, false)).await;
    let (_spotify, _fake) = fake_spotify_login(base_url);

    prefetch_saved_tracks_page_task(Arc::clone(&app), 20, 20, generation, None).await;
    server.await.unwrap();

    let app = app.lock().await;
    assert_eq!(
      app.track_table.context,
      Some(TrackTableContext::MyPlaylists)
    );
    assert!(app.track_table.tracks.is_empty());
    assert_eq!(app.saved_tracks_prefix().0.len(), 4);
    assert_eq!(app.library().saved_tracks.index, 1);
  }

  fn liked(i: usize) -> TrackInfo {
    TrackInfo::from(&crate::core::test_helpers::full_track(
      &format!("{i:022}"),
      &format!("Song {i}"),
    ))
  }

  #[test]
  fn cached_liked_songs_rebuild_contiguous_pages_after_the_first() {
    let all: Vec<TrackInfo> = (0..8).map(liked).collect();
    let first = Paged {
      items: all[..3].to_vec(),
      offset: 0,
      limit: 3,
      total: 8,
      next: Some("next".to_string()),
      previous: None,
    };
    let pages = saved_track_pages_from_cache(first, all);

    assert_eq!(
      pages.iter().map(|page| page.offset).collect::<Vec<_>>(),
      vec![0, 3, 6]
    );
    assert_eq!(pages[2].items.len(), 2);
    assert!(pages[0].next.is_some() && pages[1].next.is_some());
    assert!(pages[2].next.is_none());
  }

  #[test]
  fn a_cached_liked_songs_open_shows_every_song_with_nothing_left_to_load() {
    let (tx, _rx) = std::sync::mpsc::channel();
    let mut app = App::new(
      tx,
      crate::core::user_config::UserConfig::new(),
      Some(std::time::SystemTime::now()),
    );
    app.reset_saved_tracks_view();
    let all: Vec<TrackInfo> = (0..8).map(liked).collect();
    let first = Paged {
      items: all[..3].to_vec(),
      offset: 0,
      limit: 3,
      total: 8,
      next: Some("next".to_string()),
      previous: None,
    };

    apply_cached_liked_songs(&mut app, first, all);

    let names: Vec<String> = app
      .track_table
      .tracks
      .iter()
      .map(|t| t.name.clone())
      .collect();
    assert_eq!(
      names,
      (0..8).map(|i| format!("Song {i}")).collect::<Vec<_>>()
    );
    let (_, _, has_more) = app.saved_tracks_prefix();
    assert!(!has_more, "no page chain is needed after a cached open");
    assert!(
      app.liked_song_ids_set().contains(&format!("{:022}", 7)),
      "every cached song is known to be liked"
    );
  }

  #[test]
  fn the_liked_songs_key_ignores_the_page_size() {
    // Two opens at different terminal heights: same newest songs, different
    // first-page sizes. The key must not change.
    let mut small = saved_tracks_page(0, 2, true);
    let mut large = saved_tracks_page(0, 30, true);
    let songs: Vec<SavedTrack> = (0..30).map(|i| saved_track(&format!("{i:022}"))).collect();
    small.items = songs[..12].to_vec();
    large.items = songs.clone();
    assert_eq!(liked_songs_key(&small), liked_songs_key(&large));
    assert_eq!(liked_songs_key(&small).head_ids.len(), LIKED_SONGS_KEY_HEAD);
  }

  #[test]
  fn reconcile_restores_selection_against_folders_first_order() {
    use crate::core::app::{App, PlaylistFolder, PlaylistFolderItem};
    use crate::core::test_helpers::playlist_info;

    // Root order: P0, folderA, P1, folderB. Restore selection to P1.
    let mut app = App::default();
    // Keep this test about folder-first ordering; the community pin is covered
    // by its own tests.
    app.user_config.behavior.pin_community_playlist = false;
    *app.all_playlists_mut() = vec![
      playlist_info("00000000000000000000p0", "P0", "me", false),
      playlist_info("00000000000000000000p1", "P1", "me", false),
    ];
    *app.playlist_folder_items_mut() = vec![
      PlaylistFolderItem::Playlist {
        index: 0,
        current_id: 0,
      },
      PlaylistFolderItem::Folder(PlaylistFolder {
        name: "A".to_string(),
        current_id: 0,
        target_id: 1,
      }),
      PlaylistFolderItem::Playlist {
        index: 1,
        current_id: 0,
      },
      PlaylistFolderItem::Folder(PlaylistFolder {
        name: "B".to_string(),
        current_id: 0,
        target_id: 2,
      }),
    ];
    app.user_config.behavior.group_folders_first = true;

    reconcile_playlist_selection(&mut app, Some("00000000000000000000p1"), 0, None);

    // Sorted display view is [A, B, P0, P1]; P1 is display index 3, which maps
    // to sidebar row 4 (row 0 is the leading "+ Add Playlist" entry).
    let idx = app
      .view
      .selected_playlist_index
      .expect("selection restored");
    assert_eq!(idx, 4);
    assert!(matches!(
      app.get_playlist_display_item_at(idx - 1),
      Some(PlaylistFolderItem::Playlist { index: 1, .. })
    ));
  }

  #[test]
  fn community_pin_prompt_marks_shown_at_push_time() {
    let dir = tempfile::tempdir().unwrap();
    // Trigger conditions: Spotify source, flag false, toggle on, not following.
    let mut app = App::default();
    app.state_path = Some(dir.path().join("state.yml"));
    assert!(app.active_source == Source::Spotify);
    assert!(app.user_config.behavior.pin_community_playlist);
    assert!(!app.follows_community_playlist());
    assert!(!app.runtime_state.community_pin_prompt_shown);

    maybe_prompt_community_pin(&mut app);

    // Shown is marked at push time so a non-handler dismissal can't reappear it.
    assert!(app.runtime_state.community_pin_prompt_shown);
    assert_eq!(app.get_current_route().id, RouteId::CommunityPinPrompt);
  }

  #[test]
  fn next_saved_tracks_offset_uses_page_limit() {
    let page = saved_tracks_page(20, 20, true);
    assert_eq!(next_saved_tracks_offset(&page), Some(40));
  }

  #[test]
  fn next_saved_tracks_offset_returns_none_without_next_link() {
    let page = saved_tracks_page(20, 20, false);
    assert_eq!(next_saved_tracks_offset(&page), None);
  }

  #[test]
  fn uri_batches_split_large_contains_requests() {
    let uris = (0..120)
      .map(|index| format!("spotify:track:{index:022}"))
      .collect::<Vec<_>>();
    let batches = uri_batches(&uris)
      .map(|batch| batch.len())
      .collect::<Vec<_>>();

    assert_eq!(batches, vec![40, 40, 40]);
  }

  #[tokio::test]
  async fn library_contains_sends_its_batches_together_and_keeps_the_uri_order() {
    use crate::infra::network::metadata::tests::{fake_spotify_login, read_http_request};
    use tokio::io::AsyncWriteExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
      // All three batches must be in flight before any is answered.
      let mut held = Vec::new();
      for _ in 0..3 {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_http_request(&mut stream).await;
        let request_line = request.lines().next().unwrap_or_default().to_string();
        let answers: Vec<bool> = request_line
          .split("spotify%3Atrack%3A")
          .skip(1)
          .map(|id| {
            let digits: String = id.chars().take_while(char::is_ascii_digit).collect();
            digits.parse::<u32>().unwrap() % 3 == 0
          })
          .collect();
        held.push((stream, serde_json::to_string(&answers).unwrap()));
      }
      for (mut stream, body) in held {
        let response = format!(
          "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
          body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
      }
    });
    let app = Arc::new(Mutex::new(App::default()));
    let (spotify, _fake) = fake_spotify_login(base_url);
    let network = Network::new(Some(spotify), &app);
    let uris = (0..100)
      .map(|index| format!("spotify:track:{index:022}"))
      .collect::<Vec<_>>();

    let answers = tokio::time::timeout(
      Duration::from_secs(10),
      network.library_contains_uris(&uris),
    )
    .await
    .expect("the batches must not wait for each other")
    .unwrap();

    server.await.unwrap();
    let expected: Vec<bool> = (0..100).map(|index| index % 3 == 0).collect();
    assert_eq!(answers, expected);
  }

  #[test]
  fn populate_liked_song_ids_from_saved_tracks_uses_raw_track_ids() {
    let page = saved_tracks_page(0, 20, false);
    let mut liked_song_ids_set = HashSet::new();

    populate_liked_song_ids_from_saved_tracks(&mut liked_song_ids_set, &page);

    assert!(liked_song_ids_set.contains("0000000000000000000001"));
    assert!(liked_song_ids_set.contains("0000000000000000000002"));
    assert!(!liked_song_ids_set.contains("spotify:track:0000000000000000000001"));
  }

  #[test]
  fn playlist_track_filter_matches_title_artist_album_case_insensitively() {
    let mut track = TrackInfo::from(&full_track("0000000000000000000001"));
    track.name = "Midnight City".to_string();
    track.artists = vec!["M83".to_string()];
    track.album = "Hurry Up".to_string();

    assert!(playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("midnight")
    ));
    assert!(playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("m83")
    ));
    assert!(playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("hurry")
    ));
    assert!(playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("MIDNIGHT m83")
    ));
  }

  #[test]
  fn playlist_track_filter_requires_every_query_term() {
    let mut track = TrackInfo::from(&full_track("0000000000000000000001"));
    track.name = "Midnight City".to_string();
    track.artists = vec!["M83".to_string()];
    track.album = "Hurry Up".to_string();

    assert!(playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("city hurry")
    ));
    assert!(!playlist_track_info_matches_terms(
      &track,
      &playlist_track_search_terms("city missing")
    ));
  }

  #[test]
  fn external_fallback_only_for_confirmed_external_403s() {
    use PlaylistAccess::{External, Owned, Unknown};

    // The one path that may use the librespot fallback.
    assert!(should_attempt_external_fallback(External, true));
    // An owned playlist that 403s is a diagnosable server error, never a
    // silent fallback.
    assert!(!should_attempt_external_fallback(Owned, true));
    assert!(!should_attempt_external_fallback(Unknown, true));
    // Non-403 failures keep their real error whatever the relationship.
    assert!(!should_attempt_external_fallback(External, false));
    assert!(!should_attempt_external_fallback(Owned, false));
    assert!(!should_attempt_external_fallback(Unknown, false));
  }

  #[test]
  fn only_a_404_on_a_known_spotify_owned_playlist_counts_as_hidden() {
    // Release Radar / daylist: the Web API hides them behind a 404.
    assert!(is_hidden_spotify_owned_playlist(true, Some("spotify")));
    // A 404 on anyone else's playlist, or an unknown one, is a real miss.
    assert!(!is_hidden_spotify_owned_playlist(true, Some("someone")));
    assert!(!is_hidden_spotify_owned_playlist(true, None));
    // Other failures on a Spotify-owned playlist keep their real error.
    assert!(!is_hidden_spotify_owned_playlist(false, Some("spotify")));
  }

  fn cached_track(position: u32) -> (u32, PlayableInfo) {
    let id = format!("{position:022}");
    (
      position,
      PlayableInfo::Track(TrackInfo {
        uri: Some(format!("spotify:track:{id}")),
        name: format!("Track {position}"),
        artists: vec!["Artist".to_string()],
        album: "Album".to_string(),
        duration_ms: 1,
        id: Some(id),
        album_id: None,
        artist_refs: Vec::new(),
        is_playable: true,
        is_local: false,
        track_number: 1,
        explicit: false,
        image_url: None,
        release_date: None,
      }),
    )
  }

  #[test]
  fn cached_items_become_contiguous_pages_with_next_on_all_but_the_last() {
    // 120 positions with a removed track at 60..=69: three pages, the gap kept.
    let items: Vec<_> = (0..120)
      .filter(|p| !(60..70).contains(p))
      .map(cached_track)
      .collect();
    let pages = pages_from_cached_items(items);

    assert_eq!(
      pages.iter().map(|page| page.offset).collect::<Vec<_>>(),
      vec![0, 50, 100]
    );
    assert!(pages[0].next.is_some() && pages[1].next.is_some());
    assert!(pages[2].next.is_none());
    assert_eq!(pages[1].items.len(), 40);
    assert!(pages.iter().all(|page| page.total == 120));
  }

  #[test]
  fn an_empty_cached_playlist_is_one_terminal_page() {
    let pages = pages_from_cached_items(Vec::new());
    assert_eq!(pages.len(), 1);
    assert!(pages[0].items.is_empty() && pages[0].next.is_none());
  }

  #[test]
  fn cached_pages_round_trip_through_the_completeness_check() {
    let items: Vec<_> = (0..137).map(cached_track).collect();
    let pages = pages_from_cached_items(items.clone());
    assert_eq!(complete_playlist_items(&pages), Some(items));
  }

  #[test]
  fn a_partial_load_is_never_treated_as_the_whole_playlist() {
    let items: Vec<_> = (0..137).map(cached_track).collect();
    let pages = pages_from_cached_items(items);

    // A missing middle page: a gap between offsets.
    let gap = vec![pages[0].clone(), pages[2].clone()];
    assert_eq!(complete_playlist_items(&gap), None);
    // Interrupted before the last page: no terminal page.
    assert_eq!(complete_playlist_items(&pages[..2]), None);
    // A load that started past offset 0.
    assert_eq!(complete_playlist_items(&pages[1..]), None);
    assert_eq!(complete_playlist_items(&[]), None);
  }

  #[test]
  fn a_cached_open_fills_the_table_with_every_track_and_nothing_left_to_load() {
    let (tx, _rx) = std::sync::mpsc::channel();
    let mut app = App::new(
      tx,
      crate::core::user_config::UserConfig::new(),
      Some(std::time::SystemTime::now()),
    );
    let playlist_id = PlaylistId::from_id("37i9dQZF1DX4WYpdgoIcn6")
      .unwrap()
      .into_static();
    app.reset_playlist_tracks_view(
      playlist_id,
      crate::core::app::TrackTableContext::MyPlaylists,
    );

    let items: Vec<_> = (0..137).filter(|p| *p != 75).map(cached_track).collect();
    for page in pages_from_cached_items(items) {
      app.playlist_track_pages.upsert_page_by_offset(page);
    }
    app.set_playlist_tracks_to_table_continuous();

    assert_eq!(app.track_table.tracks.len(), 136);
    assert_eq!(app.track_table.tracks[0].name, "Track 0");
    assert_eq!(app.track_table.tracks[135].name, "Track 136");
    let positions = app.playlist_track_positions.clone().unwrap();
    assert_eq!(positions[75], 76, "positions skip the removed track");
    assert!(!app.current_playlist_has_more_tracks());
    assert_eq!(app.next_missing_playlist_tracks_offset_continuous(), None);
  }

  #[tokio::test]
  async fn a_second_walk_of_the_same_playlist_waits_for_the_first() {
    let first = playlist_walk_lock("lockSameA");
    let guard = first.lock().await;
    let second = playlist_walk_lock("lockSameA");
    assert!(Arc::ptr_eq(&first, &second));
    assert!(second.try_lock().is_err(), "must wait while a walk runs");
    drop(guard);
    assert!(second.try_lock().is_ok(), "free once the walk finishes");
  }

  #[tokio::test]
  async fn walks_of_different_playlists_do_not_block_each_other() {
    let a = playlist_walk_lock("lockDiffA");
    let _guard = a.lock().await;
    let b = playlist_walk_lock("lockDiffB");
    assert!(!Arc::ptr_eq(&a, &b));
    assert!(b.try_lock().is_ok());
  }

  #[tokio::test]
  async fn walk_pacer_spaces_sequential_waits_by_the_interval() {
    let interval = Duration::from_millis(40);
    let pacer = WalkPacer::new(interval);
    let start = tokio::time::Instant::now();
    for _ in 0..4 {
      pacer.wait().await;
    }
    // The first wait is immediate; the other three each wait one interval.
    assert!(start.elapsed() >= interval * 3);
  }

  #[tokio::test]
  async fn walk_pacer_spaces_concurrent_waiters_too() {
    let interval = Duration::from_millis(40);
    let pacer = WalkPacer::new(interval);
    let start = tokio::time::Instant::now();
    futures::future::join_all((0..4).map(|_| pacer.wait())).await;
    assert!(start.elapsed() >= interval * 3);
  }

  #[tokio::test]
  async fn walk_pacer_banks_no_burst_credit_while_idle() {
    let interval = Duration::from_millis(40);
    let pacer = WalkPacer::new(interval);
    pacer.wait().await;
    tokio::time::sleep(interval * 5).await;
    let start = tokio::time::Instant::now();
    pacer.wait().await;
    pacer.wait().await;
    // After idling, the first wait is immediate but the second still waits.
    assert!(start.elapsed() >= interval);
  }

  #[test]
  fn playlist_page_next_offset_ignores_compacted_item_count() {
    // A compacted page: zero domain items, but `next` promises more raw
    // slots. Pagination must still advance to the adjacent raw offset.
    let empty_nonterminal = domain_track_page(0, 50, 120, true, vec![]);
    assert_eq!(playlist_page_next_offset(&empty_nonterminal, 0), Some(50));

    let empty_terminal = domain_track_page(100, 50, 120, false, vec![]);
    assert_eq!(playlist_page_next_offset(&empty_terminal, 100), None);

    // A zero limit cannot advance: terminate instead of looping on one offset.
    let zero_limit = domain_track_page(0, 0, 120, true, vec![]);
    assert_eq!(playlist_page_next_offset(&zero_limit, 0), None);

    // A stale response offset must not repeat pagination: requesting 50 but
    // receiving offset 0 would otherwise request 50 again forever.
    let stale = domain_track_page(0, 50, 120, true, vec![]);
    assert_eq!(playlist_page_next_offset(&stale, 50), None);

    // An overflowing next offset terminates instead of wrapping around.
    let overflowing = domain_track_page(u32::MAX, 1, u32::MAX, true, vec![]);
    assert_eq!(playlist_page_next_offset(&overflowing, u32::MAX), None);
  }

  #[test]
  fn external_fallback_cache_remembers_confirmed_external_ids() {
    let mut cache = ExternalPlaylistFallbackCacheInner::default();
    assert!(!cache.is_external("external"));

    cache.mark_external("external");
    assert!(cache.is_external("external"));
    assert!(!cache.is_external("other"));
  }

  #[test]
  fn external_fallback_cache_stays_bounded() {
    let mut cache = ExternalPlaylistFallbackCacheInner::default();
    for index in 0..EXTERNAL_PLAYLIST_FALLBACK_CAP + 5 {
      cache.mark_external(&format!("playlist-{index}"));
    }

    assert!(cache.order.len() <= EXTERNAL_PLAYLIST_FALLBACK_CAP);
    assert!(cache.external_ids.len() <= EXTERNAL_PLAYLIST_FALLBACK_CAP);
    // The newest confirmation survives eviction.
    assert!(cache.is_external(&format!("playlist-{}", EXTERNAL_PLAYLIST_FALLBACK_CAP + 4)));
  }

  #[test]
  fn external_unavailable_status_names_development_mode() {
    assert!(EXTERNAL_PLAYLIST_UNAVAILABLE_STATUS.contains("Development Mode"));
  }

  #[cfg(feature = "streaming")]
  #[test]
  fn fallback_track_defaults_to_playable() {
    // Librespot's `restrictions` is the raw per-catalogue list, so the
    // fallback must not derive playability from it.
    assert!(external_fallback_track_is_playable());
  }

  #[cfg(feature = "streaming")]
  #[test]
  fn cached_contents_slice_windows_match_raw_positions() {
    use librespot_core::SpotifyUri;

    let uris = (1..=4)
      .map(|index| {
        SpotifyUri::from_uri(&format!("spotify:track:000000000000000000{index:04}")).unwrap()
      })
      .collect::<Vec<_>>();
    let cached = CachedExternalPlaylistContents {
      uris,
      base_offset: 0,
      total: 4,
    };

    // Slices reuse the same cached contents window by window.
    assert_eq!(
      cached
        .slice_positions(0, 2)
        .into_iter()
        .map(|(position, _)| position)
        .collect::<Vec<_>>(),
      vec![0, 1]
    );
    assert_eq!(
      cached
        .slice_positions(2, 2)
        .into_iter()
        .map(|(position, _)| position)
        .collect::<Vec<_>>(),
      vec![2, 3]
    );
    assert!(cached.has_next(0, 2));
    assert!(!cached.has_next(2, 2));
  }

  #[cfg(feature = "streaming")]
  #[test]
  fn fallback_cache_clear_refreshes_contents_but_keeps_classification() {
    use librespot_core::SpotifyUri;

    let mut cache = ExternalPlaylistFallbackCacheInner::default();
    let uris = vec![SpotifyUri::from_uri("spotify:track:0000000000000000000001").unwrap()];
    cache.store_contents(
      "external",
      CachedExternalPlaylistContents {
        uris,
        base_offset: 0,
        total: 1,
      },
    );
    assert!(cache.contents_for("external").is_some());

    // A fresh open drops the snapshot so the next visit re-downloads, while
    // the confirmed-External classification survives (no metadata re-request).
    cache.clear_contents("external");
    assert!(cache.contents_for("external").is_none());
    assert!(cache.is_external("external"));
  }

  #[cfg(feature = "streaming")]
  #[test]
  fn fallback_cache_reuses_stored_contents_across_slices() {
    use librespot_core::SpotifyUri;

    let mut cache = ExternalPlaylistFallbackCacheInner::default();
    let uris = (1..=3)
      .map(|index| {
        SpotifyUri::from_uri(&format!("spotify:track:000000000000000000{index:04}")).unwrap()
      })
      .collect::<Vec<_>>();
    cache.store_contents(
      "external",
      CachedExternalPlaylistContents {
        uris,
        base_offset: 0,
        total: 3,
      },
    );

    // Storing contents confirms the id, so classification is reused too.
    assert!(cache.is_external("external"));
    let cached = cache.contents_for("external").expect("stored contents");
    assert_eq!(cached.total, 3);
    assert_eq!(cached.slice_positions(0, 50).len(), 3);
    assert_eq!(cached.slice_positions(50, 50).len(), 0);
    assert!(!cached.has_next(0, 50));
  }
}

#[cfg(feature = "streaming")]
async fn fetch_rootlist_folders(
  streaming_player: Option<Arc<StreamingPlayer>>,
) -> Option<Vec<PlaylistFolderNode>> {
  let player = streaming_player?;
  let session = player.session();

  let bytes = match session.spclient().get_rootlist(0, Some(100_000)).await {
    Ok(bytes) => bytes,
    Err(_) => return None,
  };

  use protobuf::Message;
  let selected: librespot_protocol::playlist4_external::SelectedListContent =
    Message::parse_from_bytes(&bytes).ok()?;

  let contents = selected.contents.as_ref()?;
  Some(parse_rootlist_items(&contents.items))
}

/// Playlist ids that appear in the librespot rootlist but not in the Web API
/// playlist list, in rootlist order, without duplicates. Folders are walked
/// recursively; non-playlist URIs are ignored.
#[cfg(feature = "streaming")]
fn rootlist_only_playlist_ids(
  nodes: &[PlaylistFolderNode],
  playlists: &[PlaylistInfo],
) -> Vec<String> {
  fn walk(
    nodes: &[PlaylistFolderNode],
    known: &HashSet<&str>,
    seen: &mut HashSet<String>,
    out: &mut Vec<String>,
  ) {
    for node in nodes {
      match node.node_type {
        PlaylistFolderNodeType::Folder => walk(&node.children, known, seen, out),
        PlaylistFolderNodeType::Playlist => {
          let Some(id) = node.uri.strip_prefix("spotify:playlist:") else {
            continue;
          };
          if !known.contains(id) && seen.insert(id.to_string()) {
            out.push(id.to_string());
          }
        }
      }
    }
  }

  let known: HashSet<&str> = playlists
    .iter()
    .filter_map(|playlist| playlist.id.as_deref())
    .collect();
  let mut seen = HashSet::new();
  let mut out = Vec::new();
  walk(nodes, &known, &mut seen, &mut out);
  out
}

/// Sidebar entry for a playlist resolved through librespot. The owner id is
/// the Spotify username (`spotify` for generated and editorial playlists),
/// which is what `playlist_access_from_owner` compares against.
#[cfg(feature = "streaming")]
fn rootlist_playlist_info(
  id: &str,
  name: &str,
  owner_username: &str,
  track_count: u32,
  collaborative: bool,
) -> PlaylistInfo {
  let owner = if owner_username == SPOTIFY_OWNER_ID {
    "Spotify".to_string()
  } else {
    owner_username.to_string()
  };
  PlaylistInfo {
    uri: format!("spotify:playlist:{id}"),
    name: name.to_string(),
    owner,
    track_count,
    id: Some(id.to_string()),
    owner_id: Some(owner_username.to_string()),
    collaborative,
    public: None,
    image_url: None,
  }
}

/// Resolve rootlist-only playlist ids to sidebar entries via librespot,
/// preserving input order. Ids that fail to resolve (deleted or region-locked
/// playlists answer 404) are skipped, and so are ids still pending when
/// `ROOTLIST_RESOLVE_BUDGET` runs out.
#[cfg(feature = "streaming")]
async fn resolve_rootlist_only_playlists(
  session: &librespot_core::Session,
  ids: &[String],
) -> Vec<PlaylistInfo> {
  use protobuf::Message;

  let lookups = ids.iter().cloned().map(|id| async move {
    let spotify_id = librespot_core::SpotifyId::from_base62(&id).ok()?;
    let bytes = match session.spclient().get_playlist(&spotify_id).await {
      Ok(bytes) => bytes,
      Err(error) => {
        log::debug!("rootlist playlist {id} unavailable via librespot: {error}");
        return None;
      }
    };
    let selected: librespot_protocol::playlist4_external::SelectedListContent =
      Message::parse_from_bytes(&bytes).ok()?;
    let attributes = selected.attributes.as_ref();
    let name = attributes.map(|a| a.name()).unwrap_or_default();
    if name.is_empty() {
      return None;
    }
    Some(rootlist_playlist_info(
      &id,
      name,
      selected.owner_username(),
      selected.length().max(0) as u32,
      attributes.is_some_and(|a| a.collaborative()),
    ))
  });

  resolve_within_budget(
    lookups,
    ROOTLIST_RESOLVE_CONCURRENCY,
    ROOTLIST_RESOLVE_BUDGET,
  )
  .await
}

/// Run `lookups` with at most `concurrency` in flight and keep the `Some`
/// results, in input order. Every lookup shares one deadline `budget` from
/// now; one still pending when it passes counts as `None`. Lookups complete
/// out of order, so a slow one never holds a slot the others are waiting on.
#[cfg(feature = "streaming")]
async fn resolve_within_budget<T, F>(
  lookups: impl IntoIterator<Item = F>,
  concurrency: usize,
  budget: Duration,
) -> Vec<T>
where
  F: std::future::Future<Output = Option<T>>,
{
  use futures::stream::{self, StreamExt};

  let deadline = tokio::time::Instant::now() + budget;
  let lookups: Vec<F> = lookups.into_iter().collect();
  let mut resolved: Vec<(usize, T)> = stream::iter(lookups.into_iter().enumerate())
    .map(|(position, lookup)| async move {
      let item = tokio::time::timeout_at(deadline, lookup)
        .await
        .ok()
        .flatten()?;
      Some((position, item))
    })
    .buffer_unordered(concurrency.max(1))
    .filter_map(|item| async move { item })
    .collect()
    .await;
  resolved.sort_unstable_by_key(|(position, _)| *position);
  resolved.into_iter().map(|(_, item)| item).collect()
}

fn build_flat_playlist_items(playlists: &[PlaylistInfo]) -> Vec<PlaylistFolderItem> {
  playlists
    .iter()
    .enumerate()
    .map(|(index, _)| PlaylistFolderItem::Playlist {
      index,
      current_id: 0,
    })
    .collect()
}

fn reconcile_playlist_selection(
  app: &mut App,
  preferred_playlist_id: Option<&str>,
  preferred_folder_id: usize,
  preferred_selected_index: Option<usize>,
) {
  if app.playlist_folder_items().is_empty() {
    app.current_playlist_folder_id = 0;
    app.view.selected_playlist_index = None;
    return;
  }

  let folder_has_visible = |folder_id: usize, app: &App| {
    app.playlist_folder_items().iter().any(|item| match item {
      PlaylistFolderItem::Folder(folder) => folder.current_id == folder_id,
      PlaylistFolderItem::Playlist { current_id, .. } => *current_id == folder_id,
      PlaylistFolderItem::CommunityPin => false,
    })
  };

  app.current_playlist_folder_id = if folder_has_visible(preferred_folder_id, app) {
    preferred_folder_id
  } else {
    0
  };

  if let Some(playlist_id) = preferred_playlist_id {
    let visible_playlist_index = app
      .get_playlist_display_items()
      .into_iter()
      .enumerate()
      .find_map(|(display_idx, item)| match item {
        PlaylistFolderItem::Playlist { index, .. } => app
          .all_playlists()
          .get(*index)
          .filter(|playlist| playlist.id.as_deref() == Some(playlist_id))
          .map(|_| display_idx),
        PlaylistFolderItem::Folder(_) | PlaylistFolderItem::CommunityPin => None,
      });

    if let Some(display_idx) = visible_playlist_index {
      // Sidebar rows are offset by the leading "+ Add Playlist" row.
      app.view.selected_playlist_index = Some(display_idx + 1);
      return;
    }

    let mut target_folder: Option<usize> = None;
    for item in app.playlist_folder_items() {
      if let PlaylistFolderItem::Playlist { index, current_id } = item {
        if let Some(playlist) = app.all_playlists().get(*index) {
          if playlist.id.as_deref() == Some(playlist_id) {
            target_folder = Some(*current_id);
            break;
          }
        }
      }
    }

    if let Some(folder_id) = target_folder {
      app.current_playlist_folder_id = folder_id;
      let display_idx = app
        .get_playlist_display_items()
        .into_iter()
        .enumerate()
        .find_map(|(idx, item)| match item {
          PlaylistFolderItem::Playlist { index, .. } => app
            .all_playlists()
            .get(*index)
            .filter(|playlist| playlist.id.as_deref() == Some(playlist_id))
            .map(|_| idx),
          PlaylistFolderItem::Folder(_) | PlaylistFolderItem::CommunityPin => None,
        });
      if let Some(idx) = display_idx {
        // Sidebar rows are offset by the leading "+ Add Playlist" row.
        app.view.selected_playlist_index = Some(idx + 1);
        return;
      }
    }
  }

  // Sidebar rows run 0..=count: row 0 is "+ Add Playlist", items follow, so a
  // preferred row clamps to `count` (the last item), not `count - 1`.
  let visible_count = app.get_playlist_display_count();
  if visible_count == 0 {
    app.current_playlist_folder_id = 0;
    let root_count = app.get_playlist_display_count();
    app.view.selected_playlist_index = if root_count == 0 {
      None
    } else {
      Some(preferred_selected_index.unwrap_or(0).min(root_count))
    };
    return;
  }

  app.view.selected_playlist_index = Some(preferred_selected_index.unwrap_or(0).min(visible_count));
}

#[cfg(feature = "streaming")]
fn parse_rootlist_items(
  items: &[librespot_protocol::playlist4_external::Item],
) -> Vec<PlaylistFolderNode> {
  let mut root: Vec<PlaylistFolderNode> = Vec::new();
  let mut stack: Vec<Vec<PlaylistFolderNode>> = Vec::new();
  let mut name_stack: Vec<(String, String)> = Vec::new();

  for item in items {
    let uri = item.uri();

    if let Some(rest) = uri.strip_prefix("spotify:start-group:") {
      let (group_id, name) = match rest.find(':') {
        Some(pos) => (rest[..pos].to_string(), rest[pos + 1..].to_string()),
        None => (rest.to_string(), String::new()),
      };
      name_stack.push((group_id, name));
      stack.push(std::mem::take(&mut root));
      root = Vec::new();
    } else if uri.starts_with("spotify:end-group:") {
      if let Some((group_id, name)) = name_stack.pop() {
        let children = std::mem::take(&mut root);
        root = stack.pop().unwrap_or_default();
        root.push(PlaylistFolderNode {
          name: Some(name),
          node_type: PlaylistFolderNodeType::Folder,
          uri: format!("spotify:folder:{}", group_id),
          children,
        });
      }
    } else {
      root.push(PlaylistFolderNode {
        name: None,
        node_type: PlaylistFolderNodeType::Playlist,
        uri: uri.to_string(),
        children: Vec::new(),
      });
    }
  }

  while let Some((group_id, name)) = name_stack.pop() {
    let children = std::mem::take(&mut root);
    root = stack.pop().unwrap_or_default();
    root.push(PlaylistFolderNode {
      name: Some(name),
      node_type: PlaylistFolderNodeType::Folder,
      uri: format!("spotify:folder:{}", group_id),
      children,
    });
  }

  root
}

fn structurize_playlist_folders(
  nodes: &[PlaylistFolderNode],
  playlists: &[PlaylistInfo],
) -> Vec<PlaylistFolderItem> {
  use std::collections::{HashMap, HashSet};

  let playlist_map: HashMap<String, usize> = playlists
    .iter()
    .enumerate()
    .filter_map(|(idx, playlist)| playlist.id.clone().map(|id| (id, idx)))
    .collect();

  let mut items: Vec<PlaylistFolderItem> = Vec::new();
  let mut next_folder_id: usize = 1;
  let mut used_playlist_indices: HashSet<usize> = HashSet::new();

  fn walk(
    nodes: &[PlaylistFolderNode],
    current_folder_id: usize,
    items: &mut Vec<PlaylistFolderItem>,
    next_folder_id: &mut usize,
    playlist_map: &std::collections::HashMap<String, usize>,
    used_playlist_indices: &mut std::collections::HashSet<usize>,
  ) {
    for node in nodes {
      match node.node_type {
        PlaylistFolderNodeType::Folder => {
          let folder_id = *next_folder_id;
          *next_folder_id += 1;

          let name = node.name.as_deref().unwrap_or("Unnamed Folder").to_string();

          items.push(PlaylistFolderItem::Folder(PlaylistFolder {
            name: name.clone(),
            current_id: current_folder_id,
            target_id: folder_id,
          }));

          items.push(PlaylistFolderItem::Folder(PlaylistFolder {
            name: format!("\u{2190} {}", name),
            current_id: folder_id,
            target_id: current_folder_id,
          }));

          walk(
            &node.children,
            folder_id,
            items,
            next_folder_id,
            playlist_map,
            used_playlist_indices,
          );
        }
        PlaylistFolderNodeType::Playlist => {
          let playlist_id = node
            .uri
            .strip_prefix("spotify:playlist:")
            .unwrap_or(&node.uri);

          if let Some(&index) = playlist_map.get(playlist_id) {
            items.push(PlaylistFolderItem::Playlist {
              index,
              current_id: current_folder_id,
            });
            used_playlist_indices.insert(index);
          }
        }
      }
    }
  }

  walk(
    nodes,
    0,
    &mut items,
    &mut next_folder_id,
    &playlist_map,
    &mut used_playlist_indices,
  );

  for (index, _) in playlists.iter().enumerate() {
    if !used_playlist_indices.contains(&index) {
      items.push(PlaylistFolderItem::Playlist {
        index,
        current_id: 0,
      });
    }
  }

  items
}

#[cfg(all(test, feature = "streaming"))]
mod rootlist_only_tests {
  use super::*;
  use crate::core::test_helpers::playlist_info;

  fn playlist_node(id: &str) -> PlaylistFolderNode {
    PlaylistFolderNode {
      name: None,
      node_type: PlaylistFolderNodeType::Playlist,
      uri: format!("spotify:playlist:{id}"),
      children: Vec::new(),
    }
  }

  fn folder_node(id: &str, children: Vec<PlaylistFolderNode>) -> PlaylistFolderNode {
    PlaylistFolderNode {
      name: Some(format!("folder {id}")),
      node_type: PlaylistFolderNodeType::Folder,
      uri: format!("spotify:folder:{id}"),
      children,
    }
  }

  #[test]
  fn rootlist_ids_missing_from_the_web_api_are_returned_in_rootlist_order() {
    let nodes = vec![
      playlist_node("37i9dQZEVXbe5af5Lkta9S"),
      playlist_node("00000000000000000000p1"),
      folder_node("f1", vec![playlist_node("37i9dQZF1EP6YuccBxUcC1")]),
    ];
    let known = vec![playlist_info("00000000000000000000p1", "Mine", "me", false)];

    assert_eq!(
      rootlist_only_playlist_ids(&nodes, &known),
      vec!["37i9dQZEVXbe5af5Lkta9S", "37i9dQZF1EP6YuccBxUcC1"]
    );
  }

  #[test]
  fn rootlist_ids_already_listed_by_the_web_api_are_not_duplicated() {
    let nodes = vec![
      playlist_node("00000000000000000000p1"),
      playlist_node("00000000000000000000p2"),
    ];
    let known = vec![
      playlist_info("00000000000000000000p1", "One", "me", false),
      playlist_info("00000000000000000000p2", "Two", "friend", false),
    ];

    assert!(rootlist_only_playlist_ids(&nodes, &known).is_empty());
  }

  #[test]
  fn a_rootlist_id_repeated_across_folders_is_resolved_once() {
    let nodes = vec![
      playlist_node("37i9dQZEVXbe5af5Lkta9S"),
      folder_node("f1", vec![playlist_node("37i9dQZEVXbe5af5Lkta9S")]),
      PlaylistFolderNode {
        name: None,
        node_type: PlaylistFolderNodeType::Playlist,
        uri: "spotify:collection".to_string(),
        children: Vec::new(),
      },
    ];

    assert_eq!(
      rootlist_only_playlist_ids(&nodes, &[]),
      vec!["37i9dQZEVXbe5af5Lkta9S"]
    );
  }

  #[test]
  fn a_spotify_owned_rootlist_playlist_is_classified_external() {
    let info = rootlist_playlist_info(
      "37i9dQZEVXbe5af5Lkta9S",
      "Release Radar",
      "spotify",
      29,
      false,
    );

    assert_eq!(info.owner, "Spotify");
    assert_eq!(info.owner_id.as_deref(), Some("spotify"));
    assert_eq!(info.uri, "spotify:playlist:37i9dQZEVXbe5af5Lkta9S");
    assert_eq!(info.track_count, 29);
    assert_eq!(
      playlist_access_from_owner(Some("me"), info.owner_id.as_deref()),
      PlaylistAccess::External
    );
  }

  #[tokio::test]
  async fn lookups_past_the_budget_are_dropped_and_the_rest_keep_their_order() {
    // 0 and 3 finish, 1 hangs past the budget, 2 fails.
    let lookups = [(20u64, true), (10_000, true), (0, false), (0, true)]
      .into_iter()
      .enumerate()
      .map(|(index, (delay, ok))| async move {
        tokio::time::sleep(Duration::from_millis(delay)).await;
        ok.then_some(index)
      });

    let started = std::time::Instant::now();
    let resolved = resolve_within_budget(lookups, 8, Duration::from_millis(200)).await;

    assert_eq!(resolved, vec![0, 3]);
    assert!(started.elapsed() < Duration::from_secs(5));
  }

  #[tokio::test]
  async fn a_hung_lookup_does_not_stop_the_ones_queued_behind_it() {
    // Only 2 run at a time and the first never finishes in time.
    let lookups = (0..6).map(|index| async move {
      let delay = if index == 0 { 10_000 } else { 10 };
      tokio::time::sleep(Duration::from_millis(delay)).await;
      Some(index)
    });

    let resolved = resolve_within_budget(lookups, 2, Duration::from_millis(500)).await;

    assert_eq!(resolved, vec![1, 2, 3, 4, 5]);
  }

  #[test]
  fn resolved_rootlist_playlists_take_their_rootlist_position_in_the_sidebar() {
    let nodes = vec![
      playlist_node("37i9dQZEVXbe5af5Lkta9S"),
      playlist_node("00000000000000000000p1"),
    ];
    let mut playlists = vec![playlist_info("00000000000000000000p1", "Mine", "me", false)];
    playlists.push(rootlist_playlist_info(
      "37i9dQZEVXbe5af5Lkta9S",
      "Release Radar",
      "spotify",
      29,
      false,
    ));

    let items = structurize_playlist_folders(&nodes, &playlists);
    let order: Vec<&str> = items
      .iter()
      .filter_map(|item| match item {
        PlaylistFolderItem::Playlist { index, .. } => Some(playlists[*index].name.as_str()),
        _ => None,
      })
      .collect();
    assert_eq!(order, vec!["Release Radar", "Mine"]);
  }
}
