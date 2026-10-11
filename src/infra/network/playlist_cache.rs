//! On-disk cache of whole-playlist contents, used by playlist opens, search
//! and sort.
//!
//! All of them need every item of a playlist, which costs one paced request
//! per page of tracks. Each entry is stored with the playlist's
//! `snapshot_id`, a version tag Spotify changes on every edit from any client,
//! so an entry is only reused while it still describes the live playlist:
//! callers fetch the current `snapshot_id` (one request) and pass it to
//! [`load`], which rejects any entry recorded under a different one.
//!
//! Files live in `<app cache dir>/playlist_cache/<playlist id>.json`. Every
//! failure (missing, corrupt, older format, unwritable directory) degrades to
//! a cache miss, so the cache can only ever save requests, never break a walk.

use crate::core::plugin_api::PlayableInfo;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bumped whenever the stored layout changes; older files become misses.
const CACHE_FORMAT_VERSION: u32 = 1;

#[derive(Deserialize)]
struct CachedPlaylist {
  version: u32,
  snapshot_id: String,
  items: Vec<(u32, PlayableInfo)>,
}

#[derive(Serialize)]
struct CachedPlaylistRef<'a> {
  version: u32,
  snapshot_id: &'a str,
  items: &'a [(u32, PlayableInfo)],
}

/// The cache directory under the app cache dir (`$XDG_CACHE_HOME/spotatui`
/// or `~/.cache/spotatui`), or `None` when no home directory can be resolved.
///
/// Always `None` in test builds: the walk and open tests fake Spotify with a
/// scripted server, and a cache would add its `snapshot_id` request to their
/// script, and read or write the developer's real cache directory. The cache
/// itself is tested against temporary directories.
pub(crate) fn default_dir() -> Option<PathBuf> {
  if cfg!(test) {
    return None;
  }
  super::liked_cache::cache_root().map(|dir| dir.join("playlist_cache"))
}

/// The entry path for a playlist. Spotify ids are base62, so anything else is
/// refused rather than joined into a path (no `..`, separators, or empties).
fn entry_path(dir: &Path, playlist_id: &str) -> Option<PathBuf> {
  let valid = !playlist_id.is_empty() && playlist_id.bytes().all(|b| b.is_ascii_alphanumeric());
  valid.then(|| dir.join(format!("{playlist_id}.json")))
}

/// The cached items for `playlist_id`, if an entry exists in the current
/// format and was recorded under exactly `snapshot_id`.
pub(crate) fn load(
  dir: &Path,
  playlist_id: &str,
  snapshot_id: &str,
) -> Option<Vec<(u32, PlayableInfo)>> {
  let bytes = std::fs::read(entry_path(dir, playlist_id)?).ok()?;
  let cached: CachedPlaylist = match serde_json::from_slice(&bytes) {
    Ok(cached) => cached,
    Err(error) => {
      log::debug!("playlist cache: ignoring unreadable entry for {playlist_id}: {error}");
      return None;
    }
  };
  (cached.version == CACHE_FORMAT_VERSION && cached.snapshot_id == snapshot_id)
    .then_some(cached.items)
}

/// Record `items` for `playlist_id` under `snapshot_id`, replacing any older
/// entry. Written to a temporary file and renamed into place, so a crash or a
/// concurrent reader never sees a half-written entry.
pub(crate) fn store(
  dir: &Path,
  playlist_id: &str,
  snapshot_id: &str,
  items: &[(u32, PlayableInfo)],
) -> std::io::Result<()> {
  let path = entry_path(dir, playlist_id)
    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid playlist id"))?;
  std::fs::create_dir_all(dir)?;
  let entry = CachedPlaylistRef {
    version: CACHE_FORMAT_VERSION,
    snapshot_id,
    items,
  };
  let bytes = serde_json::to_vec(&entry).map_err(std::io::Error::other)?;
  let tmp = dir.join(format!("{playlist_id}.json.tmp.{}", std::process::id()));
  std::fs::write(&tmp, bytes)?;
  std::fs::rename(&tmp, &path).inspect_err(|_| {
    let _ = std::fs::remove_file(&tmp);
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::plugin_api::TrackInfo;

  fn track(id: &str, name: &str) -> PlayableInfo {
    PlayableInfo::Track(TrackInfo {
      uri: Some(format!("spotify:track:{id}")),
      name: name.to_string(),
      artists: vec!["Artist".to_string()],
      album: "Album".to_string(),
      duration_ms: 180_000,
      id: Some(id.to_string()),
      album_id: None,
      artist_refs: Vec::new(),
      is_playable: true,
      is_local: false,
      track_number: 1,
      explicit: false,
      image_url: None,
      release_date: None,
    })
  }

  fn sample_items() -> Vec<(u32, PlayableInfo)> {
    vec![
      (0, track("aaa111", "First")),
      (1, track("bbb222", "Second")),
    ]
  }

  #[test]
  fn a_stored_entry_loads_back_under_the_same_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), "plist1", "snapA", &sample_items()).unwrap();
    assert_eq!(load(dir.path(), "plist1", "snapA"), Some(sample_items()));
  }

  #[test]
  fn a_different_snapshot_is_a_miss() {
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), "plist1", "snapA", &sample_items()).unwrap();
    assert_eq!(load(dir.path(), "plist1", "snapB"), None);
  }

  #[test]
  fn a_newer_store_replaces_the_old_entry() {
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), "plist1", "snapA", &sample_items()).unwrap();
    let newer = vec![(0, track("ccc333", "Third"))];
    store(dir.path(), "plist1", "snapB", &newer).unwrap();
    assert_eq!(load(dir.path(), "plist1", "snapA"), None);
    assert_eq!(load(dir.path(), "plist1", "snapB"), Some(newer));
  }

  #[test]
  fn missing_corrupt_and_old_format_entries_are_misses() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(load(dir.path(), "absent", "snapA"), None);

    std::fs::write(dir.path().join("corrupt.json"), b"{not json").unwrap();
    assert_eq!(load(dir.path(), "corrupt", "snapA"), None);

    std::fs::write(
      dir.path().join("oldfmt.json"),
      br#"{"version":0,"snapshot_id":"snapA","items":[]}"#,
    )
    .unwrap();
    assert_eq!(load(dir.path(), "oldfmt", "snapA"), None);
  }

  #[test]
  fn ids_that_are_not_base62_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    for bad in ["", "../escape", "a/b", "a.b", "a b"] {
      assert!(store(dir.path(), bad, "snapA", &sample_items()).is_err());
      assert_eq!(load(dir.path(), bad, "snapA"), None);
    }
  }

  #[test]
  fn store_creates_the_directory_and_leaves_no_temp_files() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("nested").join("playlist_cache");
    store(&dir, "plist1", "snapA", &sample_items()).unwrap();
    let names: Vec<String> = std::fs::read_dir(&dir)
      .unwrap()
      .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
      .collect();
    assert_eq!(names, vec!["plist1.json".to_string()]);
  }
}
