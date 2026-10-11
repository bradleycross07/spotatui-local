//! Remembered liked ("heart") state per track, in memory and on disk.
//!
//! Every time a track table is rebuilt, spotatui asks Spotify which of its
//! tracks are liked, 40 per request. Without memory, reopening a playlist
//! re-asked for every track each time; in practice these lookups were most
//! of the app's Web API traffic, enough to trigger Spotify's rate limit.
//!
//! Each answer is kept with the time it was learned and reused for
//! [`HEART_TTL_SECS`]. Likes and unlikes made in spotatui are recorded the
//! moment Spotify accepts them, so they show at once; a change made on
//! another device can take up to the TTL to appear here.
//!
//! Stored at `<app cache dir>/liked_cache.json`. A missing, corrupt or older
//! file is an empty cache, so the cache can only ever save requests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// How long a remembered heart is trusted: one day.
pub(crate) const HEART_TTL_SECS: u64 = 24 * 60 * 60;

/// Bumped whenever the stored layout changes; older files load as empty.
const CACHE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct HeartCache {
  /// Track id -> (liked, unix seconds when learned).
  entries: HashMap<String, (bool, u64)>,
}

#[derive(Serialize, Deserialize)]
struct StoredHeartCache {
  version: u32,
  entries: HashMap<String, (bool, u64)>,
}

impl HeartCache {
  /// The remembered state, if learned within the TTL.
  pub(crate) fn lookup(&self, track_id: &str, now: u64) -> Option<bool> {
    let (liked, learned_at) = *self.entries.get(track_id)?;
    (now.saturating_sub(learned_at) < HEART_TTL_SECS).then_some(liked)
  }

  pub(crate) fn record(&mut self, track_id: &str, liked: bool, now: u64) {
    self.entries.insert(track_id.to_string(), (liked, now));
  }

  /// Drop expired entries so the file only holds what can still be used.
  fn prune(&mut self, now: u64) {
    self
      .entries
      .retain(|_, (_, learned_at)| now.saturating_sub(*learned_at) < HEART_TTL_SECS);
  }

  pub(crate) fn load(path: &Path) -> Self {
    let Ok(bytes) = std::fs::read(path) else {
      return Self::default();
    };
    match serde_json::from_slice::<StoredHeartCache>(&bytes) {
      Ok(stored) if stored.version == CACHE_FORMAT_VERSION => Self {
        entries: stored.entries,
      },
      Ok(_) => Self::default(),
      Err(error) => {
        log::debug!("liked cache: ignoring unreadable file: {error}");
        Self::default()
      }
    }
  }

  /// Write the unexpired entries, through a temporary file renamed into place
  /// so a crash or a second spotatui never reads half a file.
  pub(crate) fn store(&self, path: &Path, now: u64) -> std::io::Result<()> {
    let mut live = self.clone();
    live.prune(now);
    let stored = StoredHeartCache {
      version: CACHE_FORMAT_VERSION,
      entries: live.entries,
    };
    let bytes = serde_json::to_vec(&stored).map_err(std::io::Error::other)?;
    if let Some(dir) = path.parent() {
      std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
      let _ = std::fs::remove_file(&tmp);
    })
  }
}

pub(crate) fn now_secs() -> u64 {
  SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|elapsed| elapsed.as_secs())
    .unwrap_or(0)
}

/// The shared cache is off in test builds: tests run in one process, so it
/// would carry hearts from one test into the next, and it would write into the
/// developer's real cache directory. `HeartCache` itself is tested directly.
const SHARED_CACHE_ENABLED: bool = !cfg!(test);

/// `$XDG_CACHE_HOME/spotatui` or `~/.cache/spotatui`. Builds without a
/// Spotify session have no app cache directory, so their caches stay off.
#[cfg(feature = "spotify-session")]
pub(crate) fn cache_root() -> Option<PathBuf> {
  crate::core::paths::app_cache_dir()
}

#[cfg(not(feature = "spotify-session"))]
pub(crate) fn cache_root() -> Option<PathBuf> {
  None
}

fn default_path() -> Option<PathBuf> {
  super::liked_cache::cache_root().map(|dir| dir.join("liked_cache.json"))
}

/// The process-wide cache, loaded from disk on first use. Its lock is only
/// ever held for map lookups and inserts, never across an await.
fn shared() -> &'static Mutex<HeartCache> {
  static CACHE: OnceLock<Mutex<HeartCache>> = OnceLock::new();
  CACHE.get_or_init(|| {
    Mutex::new(
      default_path()
        .map(|p| HeartCache::load(&p))
        .unwrap_or_default(),
    )
  })
}

fn with_cache<T>(f: impl FnOnce(&mut HeartCache) -> T) -> T {
  let mut cache = shared()
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  f(&mut cache)
}

/// Split `ids` into those with a remembered state (returned with it) and
/// those that still need asking.
pub(crate) fn partition_known(ids: Vec<String>) -> (Vec<(String, bool)>, Vec<String>) {
  if !SHARED_CACHE_ENABLED {
    return (Vec::new(), ids);
  }
  let now = now_secs();
  with_cache(|cache| {
    let mut known = Vec::new();
    let mut unknown = Vec::new();
    for id in ids {
      match cache.lookup(&id, now) {
        Some(liked) => known.push((id, liked)),
        None => unknown.push(id),
      }
    }
    (known, unknown)
  })
}

/// Remember states learned from Spotify (lookups, the Liked Songs list, or a
/// like/unlike it just accepted).
pub(crate) fn record_all<'a>(states: impl IntoIterator<Item = (&'a str, bool)>) {
  if !SHARED_CACHE_ENABLED {
    return;
  }
  let now = now_secs();
  with_cache(|cache| {
    for (id, liked) in states {
      cache.record(id, liked, now);
    }
  });
}

/// Persist the cache in the background. Cheap enough to call after each batch
/// of new answers: the file is a few hundred KB at most.
pub(crate) fn save_in_background() {
  if !SHARED_CACHE_ENABLED {
    return;
  }
  let Some(path) = default_path() else {
    return;
  };
  let snapshot = with_cache(|cache| cache.clone());
  tokio::task::spawn_blocking(move || {
    if let Err(error) = snapshot.store(&path, now_secs()) {
      log::warn!("liked cache: could not save: {error}");
    }
  });
}

/// Track ids among `spotify:track:<id>` URIs (albums, episodes and shows are
/// not tracked here).
pub(crate) fn track_ids_of_uris(uris: &[String]) -> Vec<&str> {
  uris
    .iter()
    .filter_map(|uri| uri.strip_prefix("spotify:track:"))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_recorded_heart_is_remembered_until_it_expires() {
    let mut cache = HeartCache::default();
    cache.record("t1", true, 1_000);
    cache.record("t2", false, 1_000);
    assert_eq!(cache.lookup("t1", 1_000 + HEART_TTL_SECS - 1), Some(true));
    assert_eq!(cache.lookup("t2", 1_000 + HEART_TTL_SECS - 1), Some(false));
    assert_eq!(cache.lookup("t1", 1_000 + HEART_TTL_SECS), None);
    assert_eq!(cache.lookup("unknown", 1_000), None);
  }

  #[test]
  fn a_newer_answer_replaces_the_old_one() {
    // A like made in spotatui must win over the remembered unliked state.
    let mut cache = HeartCache::default();
    cache.record("t1", false, 1_000);
    cache.record("t1", true, 1_005);
    assert_eq!(cache.lookup("t1", 1_010), Some(true));
  }

  #[test]
  fn the_cache_survives_a_save_and_load_without_expired_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("liked_cache.json");
    let now = 10 * HEART_TTL_SECS;
    let mut cache = HeartCache::default();
    cache.record("fresh", true, now - 10);
    cache.record("stale", true, now - HEART_TTL_SECS - 10);
    cache.store(&path, now).unwrap();

    let loaded = HeartCache::load(&path);
    assert_eq!(loaded.lookup("fresh", now), Some(true));
    assert_eq!(loaded.entries.len(), 1, "expired entries are not written");
    let leftovers = std::fs::read_dir(path.parent().unwrap()).unwrap().count();
    assert_eq!(leftovers, 1, "no temporary file left behind");
  }

  #[test]
  fn missing_corrupt_and_old_files_load_as_empty() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
      HeartCache::load(&dir.path().join("absent.json")),
      HeartCache::default()
    );

    let corrupt = dir.path().join("corrupt.json");
    std::fs::write(&corrupt, b"{not json").unwrap();
    assert_eq!(HeartCache::load(&corrupt), HeartCache::default());

    let old = dir.path().join("old.json");
    std::fs::write(&old, br#"{"version":0,"entries":{"t1":[true,1]}}"#).unwrap();
    assert_eq!(HeartCache::load(&old), HeartCache::default());
  }

  #[test]
  fn only_track_uris_are_remembered() {
    let uris = vec![
      "spotify:track:abc".to_string(),
      "spotify:album:def".to_string(),
      "spotify:episode:ghi".to_string(),
    ];
    assert_eq!(track_ids_of_uris(&uris), vec!["abc"]);
  }
}
