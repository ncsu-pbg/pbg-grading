"""
Backend tests against a fake pod-counter HTTP server.

The fake implements just enough of the on-camera API (see the Oak 4 App's
oakapp/pod-counter/main.py) for CameraController: /status, /run/start,
/run/stop, /runs, DELETE /runs/<name>/csv. No camera or Dash needed.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from grading_backend import CameraController, SessionLog, format_elapsed

CSV_TEXT = (
    "track_id,length_px,width_px,area_px2,length_mm_raw,width_mm_raw,"
    "area_mm2_raw,length_mm,width_mm,area_mm2,hunter_l,hunter_a,hunter_b,"
    "color_hex,width_quality,meas_quality,plot_name,plot_weight\n"
    "1,100.0,50.0,4000.0,18.6,9.3,139.0,17.9,8.6,123.0,55.1,4.2,18.0,"
    "#a08050,clean,mask,101,150\n"
    "2,110.0,52.0,4400.0,20.5,9.7,152.9,19.7,8.9,135.3,54.0,4.0,17.5,"
    "#a08050,clean,mask,101,150\n"
)


class FakePodCounter:
    """In-memory stand-in for the on-camera service."""

    def __init__(self):
        self.state = "idle"
        self.run_name = ""
        self.plot_name = ""
        self.plot_weight = ""
        self.save_frames = False
        self.unique = 0
        self.next_unique = 2
        self.deleted_csvs = []
        self.deleted_runs = []
        self.stored_runs = [
            {"name": "old_run_1", "files": 12, "bytes": 3_400_000,
             "has_csv": True, "active": False},
        ]

        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):       # silence request logging
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/status":
                    return self._json({
                        "app_version": "0.6.0", "label": "fake",
                        "config_fingerprint": "abc123", "pending_fingerprint": "",
                        "config_source": "overlay", "staged_at": "2026-07-01",
                        "staged_mode": "pod", "state": fake.state,
                        "run_name": fake.run_name if fake.state == "running" else "",
                        "plot_name": fake.plot_name if fake.state == "running" else "",
                        "fps": 28.5, "frames_processed": 100,
                        "save_frames": fake.save_frames, "frames_saved": 0,
                        "frames_dropped": 0,
                        "elapsed_s": 12.0 if fake.state == "running" else 0.0,
                        "unique": fake.unique if fake.state == "running" else 0,
                        "active": 1 if fake.state == "running" else 0,
                    })
                if self.path == "/runs":
                    return self._json(fake.stored_runs)
                return self._json({"error": "not found"}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if self.path == "/run/start":
                    if fake.state == "running":
                        return self._json({"error": "a run is already active"}, 409)
                    fake.state = "running"
                    fake.plot_name = body.get("plot_name", "")
                    fake.plot_weight = body.get("plot_weight", "")
                    fake.save_frames = bool(body.get("save_frames"))
                    fake.run_name = f"{fake.plot_name.replace(' ', '_')}_x"
                    fake.unique = fake.next_unique
                    return self._json({"run_name": fake.run_name})
                if self.path == "/run/stop":
                    if fake.state != "running":
                        return self._json({"error": "no active run"}, 409)
                    fake.state = "idle"
                    # Like the real camera: a zero-count run returns a
                    # header-only CSV (no data rows), and the CSV carries
                    # whatever plot_name/plot_weight came in at /run/start
                    # (blank weight stays blank — the seed flow relies on it).
                    csv_text = (CSV_TEXT if fake.unique
                                else CSV_TEXT.split("\n")[0] + "\n")
                    csv_text = csv_text.replace(
                        ",101,150\n",
                        f",{fake.plot_name},{fake.plot_weight}\n")
                    return self._json({
                        "run_name": fake.run_name, "n_pods": fake.unique,
                        "elapsed_s": 12.3, "frames_saved": 0,
                        "frames_dropped": 0, "csv": csv_text,
                    })
                return self._json({"error": "not found"}, 404)

            def do_DELETE(self):
                if self.path.endswith("/csv"):
                    fake.deleted_csvs.append(self.path.split("/")[2])
                    return self._json({"ok": True})
                fake.deleted_runs.append(self.path.split("/")[2])
                return self._json({"ok": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture()
def fake_cam():
    f = FakePodCounter()
    yield f
    f.close()


@pytest.fixture()
def controller(fake_cam, tmp_path):
    session = SessionLog(tmp_path, "pod")
    ctl = CameraController("pod", "Pod", "127.0.0.1", fake_cam.port, session)
    ctl._auto_start_tried = True     # no oakctl in tests
    ctl.poll_once()                  # come online
    return ctl


def _wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_format_elapsed():
    assert format_elapsed(0) == "00:00"
    assert format_elapsed(75.9) == "01:15"


def test_offline_snapshot(tmp_path):
    ctl = CameraController("pod", "Pod", "127.0.0.1", 1,
                           SessionLog(tmp_path, "pod"))
    ctl._auto_start_tried = True
    ctl.poll_once()
    s = ctl.snapshot()
    assert s["online"] is False and s["state"] == "offline"
    assert "offline" in ctl.start("s1", "101", "150", False)


def test_start_requires_name_and_weight(controller):
    assert "Sample Name" in controller.start("", "101", "150", False)
    assert "Weight is required" in controller.start("s1", "101", "", False)
    assert "positive" in controller.start("s1", "101", "-5", False)
    assert "number" in controller.start("s1", "101", "abc", False)
    # test mode relaxes the weight requirement entirely
    assert controller.start("anything", "", "", False,
                            require_weight=False) is None


def test_start_poll_stop_roundtrip(controller, fake_cam, tmp_path):
    assert controller.start("101", "101", "150", False) is None
    controller.poll_once()
    s = controller.snapshot()
    assert s["running"] and s["count"] == 2 and s["elapsed"] == "00:12"

    # double start rejected (cached status now says running)
    assert "already in progress" in controller.start("s2", "102", "10", False)

    assert controller.stop() is None
    assert _wait(lambda: not controller.snapshot()["stopping"])

    # local CSV written with the app's naming convention
    csvs = list(tmp_path.glob("pod_101_*.csv"))
    assert len(csvs) == 1
    assert csvs[0].read_text(encoding="utf-8") == CSV_TEXT

    # session row appended with metadata read back from the CSV itself
    runs = controller.session.snapshot()["session"]
    assert len(runs) == 1
    r = runs[0]
    assert (r["mode"], r["plot"], r["weight"], r["count"]) == \
        ("Pod", "101", "150g", 2)
    assert r["csv_file"] == csvs[0].name

    # CSV cleared from the camera, frames/run dir untouched
    assert fake_cam.deleted_csvs == [fake_cam.run_name]
    assert fake_cam.deleted_runs == []

    # session persisted to disk under this camera's own prefix
    files = list((tmp_path / "Sessions").glob("session_pod_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["runs"][0]["count"] == 2


def test_adopts_run_already_in_progress(fake_cam, tmp_path):
    # Camera is already counting before this app connects (host crashed).
    fake_cam.state = "running"
    fake_cam.run_name = "7_x"
    fake_cam.plot_name = "7"
    fake_cam.unique = 5

    ctl = CameraController("pod", "Pod", "127.0.0.1", fake_cam.port,
                           SessionLog(tmp_path, "pod"))
    ctl._auto_start_tried = True
    ctl.poll_once()
    s = ctl.snapshot()
    assert s["running"] and s["count"] == 5
    assert ctl._attached_run == "7_x"
    assert "Adopted run" in s["log"]

    # ...and stop works on the adopted run.
    assert ctl.stop() is None
    assert _wait(lambda: not ctl.snapshot()["stopping"])
    assert len(ctl.session.snapshot()["session"]) == 1


def test_run_ended_outside_app(controller, fake_cam):
    assert controller.start("101", "101", "150", False) is None
    controller.poll_once()
    fake_cam.state = "idle"          # e.g. app restarted via oakctl
    controller.poll_once()
    s = controller.snapshot()
    assert not s["running"]
    assert "ended outside this app" in s["log"]
    assert controller._attached_run == ""


def test_zero_count_run_saves_nothing(controller, fake_cam, tmp_path):
    fake_cam.next_unique = 0
    assert controller.start("999", "999", "50", False) is None
    controller.poll_once()
    assert controller.stop() is None
    assert _wait(lambda: not controller.snapshot()["stopping"])
    assert list(tmp_path.glob("*.csv")) == []
    r = controller.session.snapshot()["session"][0]
    assert r["count"] == 0 and r["csv_file"] == "(none)"
    # No CSV rows to read metadata back from — the start-time values must
    # survive into the session row (regression: showed 'unnamed' / blank).
    assert r["plot"] == "999" and r["weight"] == "50g"
    assert fake_cam.deleted_csvs == []   # nothing fetched -> nothing deleted


def test_stored_runs_listing(controller, fake_cam):
    runs = controller.list_runs()
    assert runs and runs[0]["name"] == "old_run_1"
    controller.delete_run("old_run_1")
    assert fake_cam.deleted_runs == ["old_run_1"]


def test_session_load_roundtrip(tmp_path):
    s1 = SessionLog(tmp_path, "pod")
    s1.add_run("Pod", "2026_T_NC_X_101", "101", "150", 42, "01:00", "a.csv")
    path = str(s1.session_path)

    # A seed session lives in its own file and never sees pod sessions.
    seed = SessionLog(tmp_path, "seed")
    seed.add_run("Seed", "2026_T_NC_X_101", "101", "80", 120, "02:00", "b.csv")
    assert all("session_seed_" in s["path"] for s in seed.list_sessions())
    assert all("session_pod_" in s["path"] for s in s1.list_sessions())

    s2 = SessionLog(tmp_path, "pod")
    n, err = s2.load_session(path)
    assert err is None and n == 1
    assert s2.snapshot()["session"][0]["plot"] == "101"
    assert str(s2.session_path) == path   # new runs continue in that file

    sessions = s2.list_sessions()
    assert any(x["path"] == path for x in sessions)


def test_weight_after_stop_flow(fake_cam, tmp_path):
    session = SessionLog(tmp_path, "seed")
    ctl = CameraController("seed", "Seed", "127.0.0.1", fake_cam.port,
                           session, weight_after_stop=True)
    ctl._auto_start_tried = True
    ctl.poll_once()

    # Weight not required at start for this mode…
    assert ctl.start("1001", "1001", "", False) is None
    ctl.poll_once()
    assert ctl.stop() is None
    assert _wait(lambda: not ctl.snapshot()["stopping"])

    # …so the finished run flags a pending weight request.
    pending = ctl.snapshot()["pending_weight"]
    assert pending and pending["plot"] == "1001" and pending["count"] == 2
    row = session.snapshot()["session"][0]
    assert row["weight"] == ""

    # Invalid answers are rejected and keep the request pending.
    assert "number" in ctl.set_pending_weight("abc")
    assert ctl.snapshot()["pending_weight"] is not None

    # A valid weight lands in the session row AND the CSV column.
    assert ctl.set_pending_weight("205") is None
    assert ctl.snapshot()["pending_weight"] is None
    row = session.snapshot()["session"][0]
    assert row["weight"] == "205g"
    csv_file = list(tmp_path.glob("seed_1001_*.csv"))[0]
    assert ",205" in csv_file.read_text(encoding="utf-8").splitlines()[1]


def test_weight_after_stop_skip(fake_cam, tmp_path):
    ctl = CameraController("seed", "Seed", "127.0.0.1", fake_cam.port,
                           SessionLog(tmp_path, "seed"),
                           weight_after_stop=True)
    ctl._auto_start_tried = True
    ctl.poll_once()
    assert ctl.start("1002", "1002", "", False) is None
    ctl.poll_once()
    assert ctl.stop() is None
    assert _wait(lambda: not ctl.snapshot()["stopping"])
    assert ctl.snapshot()["pending_weight"] is not None
    ctl.set_pending_weight(None)          # "No recorded weight"
    assert ctl.snapshot()["pending_weight"] is None
    assert ctl.session.snapshot()["session"][0]["weight"] == ""


def test_on_run_complete_hook_carries_meta(controller, fake_cam):
    seen = []
    controller.on_run_complete = lambda c, info: seen.append(info)
    assert controller.start("s1", "101", "150", False,
                            meta={"trial_path": "/x/y.xlsx"}) is None
    controller.poll_once()
    assert controller.stop() is None
    assert _wait(lambda: seen)
    info = seen[0]
    assert info["meta"]["trial_path"] == "/x/y.xlsx"
    assert info["plot"] == "101" and info["count"] == 2
    assert info["csv_path"] is not None


def test_hook_deferred_until_weight_answered(fake_cam, tmp_path):
    ctl = CameraController("seed", "Seed", "127.0.0.1", fake_cam.port,
                           SessionLog(tmp_path, "seed"),
                           weight_after_stop=True)
    ctl._auto_start_tried = True
    ctl.poll_once()
    seen = []
    ctl.on_run_complete = lambda c, info: seen.append(info)
    assert ctl.start("s3", "1003", "", False,
                     meta={"trial_path": "/t.xlsx"}) is None
    ctl.poll_once()
    assert ctl.stop() is None
    assert _wait(lambda: not ctl.snapshot()["stopping"])
    assert not seen                  # held until the weight is answered
    assert ctl.set_pending_weight("205") is None
    assert _wait(lambda: seen)
    assert seen[0]["weight"] == "205"
    assert seen[0]["meta"]["trial_path"] == "/t.xlsx"


def test_session_naming(tmp_path):
    s1 = SessionLog(tmp_path, "pod")
    s1.add_run("Pod", "2026_T_NC_X_101", "101", "150", 42, "01:00", "a.csv")
    s1.set_name("LEW MET day 1")
    assert s1.snapshot()["name"] == "LEW MET day 1"

    # Name shows in the saved-session labels and survives a reload.
    labels = [x["label"] for x in s1.list_sessions()]
    assert any(lbl.startswith("LEW MET day 1 — ") for lbl in labels)
    s2 = SessionLog(tmp_path, "pod")
    n, err = s2.load_session(str(s1.session_path))
    assert err is None and n == 1
    assert s2.snapshot()["name"] == "LEW MET day 1"
