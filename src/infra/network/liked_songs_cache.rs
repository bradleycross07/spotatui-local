//! On-disk cache of the whole Liked Songs list.
//!
//! Opening Liked Songs downloads every page of it (one paced request per
//! page). Liked Songs has no `snapshot_id`, so an entry is keyed on what the
//! first page already tells us: the total count and the ids on that page.
//! Liked Songs is always newest first, so any like changes the first page and
//! any unlike changes the count; an entry is reused only while both match.
//!
//! Stored at `<app cache dir>/liked_songs.json`. A missing, corrupt or older
//! file is a miss, so the cache can only ever save requests.

use crate::core::plugin_api::TrackInfo;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bumped whenever the stored layout changes; older files become misses.
const CACHE_FORMAT_VERSION: u32 = 1;

/// What identifies one state of Liked Songs: its total and first-page ids.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LikedSongsKey {
  pub(crate) total: u32,
  pub(crate) head_ids: Vec<String>,
}

#[derive(Deserialize)]
struct Stored {
  version: u32,
  key: LikedSongsKey,
  tracks: Vec<TrackInfo>,
}

#[derive(Serialize)]
struct StoredRef<'a> {
  version: u32,
  key: &'a LikedSongsKey,
  tracks: &'a [TrackInfo],
}

/// `<app cache dir>/liked_songs.json`, or `None` with no home directory.
///
/// Always `None` in test builds, like the other caches: tests fake Spotify
/// with scripted servers and must not touch the developer's real cache.
pub(crate) fn default_path() -> Option<PathBuf> {
  if cfg!(test) {
    return None;
  }
  super::liked_cache::cache_root().map(|dir| dir.join("liked_songs.json"))
}

/// The cached list, if one is stored under exactly `key` and is whole (its
/// length matches the total it was stored with).
pub(crate) fn load(path: &Path, key: &LikedSongsKey) -> Option<Vec<TrackInfo>> {
  let bytes = std::fs::read(path).ok()?;
  let stored: Stored = match serde_json::from_slice(&bytes) {
    Ok(stored) => stored,
    Err(error) => {
      log::debug!("liked songs cache: ignoring unreadable file: {error}");
      return None;
    }
  };
  (stored.version == CACHE_FORMAT_VERSION
    && &stored.key == key
    && stored.tracks.len() == key.total as usize)
    .then_some(stored.tracks)
}

/// Record the whole list under `key`, through a temporary file renamed into
/// place so a crash or a second spotatui never reads half a file.
pub(crate) fn store(path: &Path, key: &LikedSongsKey, tracks: &[TrackInfo]) -> std::io::Result<()> {
  let bytes = serde_json::to_vec(&StoredRef {
    version: CACHE_FORMAT_VERSION,
    key,
    tracks,
  })
  .map_err(std::io::Error::other)?;
  if let Some(dir) = path.parent() {
    std::fs::create_dir_all(dir)?;
  }
  let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
  std::fs::write(&tmp, bytes)?;
  std::fs::rename(&tmp, path).inspect_err(|_| {
    let _ = std::fs::remove_file(&tmp);
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::test_helpers::full_track;

  fn tracks(n: usize) -> Vec<TrackInfo> {
    (0..n)
      .map(|i| TrackInfo::from(&full_track(&format!("{i:022}"), &format!("Song {i}"))))
      .collect()
  }

  fn key(total: u32, head: &[&str]) -> LikedSongsKey {
    LikedSongsKey {
      total,
      head_ids: head.iter().map(|id| id.to_string()).collect(),
    }
  }

  #[test]
  fn a_stored_list_loads_back_under_the_same_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("liked_songs.json");
    let k = key(3, &["a", "b"]);
    store(&path, &k, &tracks(3)).unwrap();
    assert_eq!(load(&path, &k), Some(tracks(3)));
  }

  #[test]
  fn a_new_like_or_an_unlike_is_a_miss() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("liked_songs.json");
    store(&path, &key(3, &["a", "b"]), &tracks(3)).unwrap();
    // A like puts a new song on top of the first page.
    assert_eq!(load(&path, &key(4, &["new", "a"]),), None);
    // An unlike lowers the count.
    assert_eq!(load(&path, &key(2, &["a", "b"])), None);
  }

  #[test]
  fn a_list_shorter_than_its_total_is_never_served() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("liked_songs.json");
    let k = key(5, &["a"]);
    store(&path, &k, &tracks(3)).unwrap();
    assert_eq!(load(&path, &k), None);
  }

  #[test]
  fn missing_corrupt_and_old_files_are_misses() {
    let dir = tempfile::tempdir().unwrap();
    let k = key(0, &[]);
    assert_eq!(load(&dir.path().join("absent.json"), &k), None);
    let corrupt = dir.path().join("corrupt.json");
    std::fs::write(&corrupt, b"{not json").unwrap();
    assert_eq!(load(&corrupt, &k), None);
    let old = dir.path().join("old.json");
    std::fs::write(
      &old,
      br#"{"version":0,"key":{"total":0,"head_ids":[]},"tracks":[]}"#,
    )
    .unwrap();
    assert_eq!(load(&old, &k), None);
  }
}
