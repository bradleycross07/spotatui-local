# spotatui-local

My spotatui build: upstream https://github.com/LargeModGames/spotatui plus my local patches on the `local` branch.

Build: `cargo build --release --no-default-features --features tui,streaming,discord-rpc,mpris,cover-art,scripting`

Update: rebase `local` onto the newest upstream release tag, rebuild,
then `git push --force-with-lease origin local`.
