# PBG Grading — Station Setup

Each grading station runs this app **locally** (it must reach the OAK PoE
cameras on the `10.80.32.x` subnet, so it cannot be cloud-hosted). Setup is
once per machine; after that, launching is a double-click.

## One-time setup (per station)

1. **Clone the repo** to the station (any folder the operator can write to —
   no admin rights needed):

   ```
   git clone <repo-url> "PBG Grading"
   ```

2. **Copy the per-station files** that are *not* in git:
   - `.env` — Dropbox credentials (copy from an existing station or the
     team password store).
   - `config.yaml` is in git as a starting point; edit it for this bench
     (camera IPs, calibration, crop). See "Updating" below for how local
     edits interact with `git pull`.

3. **Run the setup script:**
   - **Windows:** double-click `setup_windows.bat`
   - **macOS:** double-click `setup_macos.command`
     (if macOS blocks it: right-click → Open, or run
     `chmod +x setup_macos.command` first)

   The script downloads **micromamba** (a single small binary — no admin
   rights, doesn't touch any system Python or Anaconda install) and builds
   the pinned Python environment from `environment.yml` into `envs/grading`
   inside this folder. First run downloads ~330 MB and takes several
   minutes. It finishes by verifying every import, and on Windows it also
   puts a **"PBG Grading"** shortcut on the Desktop.

## Daily use

- **Windows:** Desktop shortcut "PBG Grading" (or `Start PBG Grading.bat`)
- **macOS:** double-click `Start PBG Grading.command`

The app starts and the browser opens to it automatically. Closing the
terminal/console window stops the app.

Calibration is unchanged: run `calibrate.py` in the same environment —
the PyQt6 dialogs open locally as before:

```
.micromamba/micromamba run -p ./envs/grading python calibrate.py    # macOS
.micromamba\micromamba.exe run -p .\envs\grading python calibrate.py  # Windows
```

## Per-station settings

| What | Where |
| --- | --- |
| Camera IPs, model, calibration, crop, tracker | `config.yaml` (per bench — differences between stations are intentional) |
| Dropbox credentials | `.env` (never committed) |
| Serve to other LAN machines | uncomment `PBG_HOST=0.0.0.0` in the launcher (default `127.0.0.1` = this machine only, which is safer) |
| Port / suppress auto-browser | `PBG_PORT` / `PBG_NO_BROWSER=1` in the launcher |

## Updating a station

```
git pull
```

- If `config.yaml` has local edits, `git pull` may refuse: use
  `git stash && git pull && git stash pop` to keep the bench's settings.
- If `environment.yml` changed in the pull, **re-run the setup script**
  (it rebuilds the env from scratch; cached downloads make this fast).
- Code-only changes need nothing else — just relaunch.

`environment.yml` pins the exact known-good versions (the camera /
firmware / driver stack is the fragile layer). Don't upgrade packages on a
station by hand; change the pin on a dev machine with cameras attached,
verify, commit, and let stations pull it.

## Troubleshooting

- **"Camera won't connect" / device busy** — an OAK device allows only
  **one host connection at a time**. A stale app instance (this machine or
  another) blocks new ones. Close other instances of the app, wait a few
  seconds, retry.
- **Browser doesn't open / "connection refused"** — the console window
  shows the URL (default `http://127.0.0.1:8050`); check it for errors.
  If the port is taken, set `PBG_PORT` in the launcher.
- **Broken environment** (failed update, deleted files) — just re-run the
  setup script; it rebuilds `envs/grading` from scratch.
- **`environment.yml` vs `config.yaml`** — the `.yml` file is the Python
  environment spec; the `.yaml` file is the runtime camera/calibration
  config. They are unrelated.
