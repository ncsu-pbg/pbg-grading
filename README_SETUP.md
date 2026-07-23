# PBG Grading — Station Setup

Each grading station runs this app **locally** (it must reach the OAK PoE
cameras' HTTP API on the `10.80.32.x` subnet, port 8808, so it cannot be
cloud-hosted). Setup is once per machine; after that, launching is a
double-click.

The heavy lifting — detection, tracking, measurement, calibration — runs
**on the cameras themselves** (the pod-counter app, deployed from the
"Oak 4 App" repository). This app is a pure web front-end: it starts/stops
runs over HTTP, shows live counts, and collects the finished CSVs. Note
the camera API is unauthenticated HTTP; keep the cameras on the lab
network. Optional: install `oakctl` (Luxonis) so the app can power the
camera apps on/off from the UI — without it everything else still works.

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
     (camera IPs only — calibration lives on the cameras now). See
     "Updating" below for how local edits interact with `git pull`.

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

Calibration happens **in the browser, on the camera**: click "Camera
Setup" on a camera panel (or open `http://<camera-ip>:8808/setup`
directly). Mount alignment, focus, and target calibration all save to a
config overlay on the camera itself — nothing calibration-related is
stored on this machine.

Deploying or updating the on-camera app (new models, tracker settings)
is done from the **Oak 4 App repository** (`stage.py` + `oakctl app run`),
not from here.

## Per-station settings

| What | Where |
| --- | --- |
| Camera IPs / ports, output folder | `config.yaml` (per bench) |
| Calibration, crop, tracker, models | **on the camera** (setup page / Oak 4 App repo) |
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

`environment.yml` pins the exact known-good versions. Don't upgrade
packages on a station by hand; change the pin on a dev machine, verify,
commit, and let stations pull it. (The fragile depthai/firmware layer now
lives on the cameras, not in this environment.)

## Troubleshooting

- **Panel shows "offline"** — the camera app isn't running or the camera
  is unreachable. The app tries one `oakctl app start` automatically;
  press "▶ App" to retry, or check power/network (`ping 10.80.32.101`,
  then `http://10.80.32.101:8808/status` in a browser).
- **A run was going and the PC crashed** — nothing is lost: the run kept
  counting on the camera. Reopen the app; the panel adopts the
  in-progress run and Stop works normally. If the camera app was
  restarted instead, the finished CSV is under "Stored Runs…".
- **Browser doesn't open / "connection refused"** — the console window
  shows the URL (default `http://127.0.0.1:8050`); check it for errors.
  If the port is taken, set `PBG_PORT` in the launcher.
- **Broken environment** (failed update, deleted files) — just re-run the
  setup script; it rebuilds `envs/grading` from scratch.
- **App suddenly won't launch and `envs/grading` shrank** — if the station
  folder lives inside Dropbox, Dropbox's space-saving can offload/delete
  files out of the environment. The setup scripts mark `envs/` and
  `.micromamba/` with Dropbox's ignore attribute to prevent this; re-run the
  setup script to rebuild and re-apply the marker.
- **`environment.yml` vs `config.yaml`** — the `.yml` file is the Python
  environment spec; the `.yaml` file is the station config (camera IPs).
  They are unrelated.
- **"...is not digitally signed / cannot be loaded" on Windows** — that's a
  PowerShell execution-policy block, common on domain/managed machines. Setup
  is pure `.bat` (cmd) specifically to avoid it, so this shouldn't appear. If
  you see it, you're running an old `setup_windows.ps1` — delete it and use
  `setup_windows.bat`. The launchers are also pure `.bat` and are unaffected.
