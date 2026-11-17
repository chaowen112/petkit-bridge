from fastapi.testclient import TestClient
from app import create_app, redact_source


def test_source_redaction_removes_userinfo_and_query():
    assert redact_source("rtsp://alice:secret@cam.example/live?token=hidden") == "rtsp://cam.example/live"


def test_status_and_stream_require_token():
    client = TestClient(create_app(token="test-token", source_url=""))
    assert client.get("/health").status_code == 404
    assert client.get("/test-token/health").json()["connected"] is False
    assert client.get("/wrong/stream").status_code == 404


def test_health_does_not_disclose_source_credentials():
    client = TestClient(create_app(token="test-token", source_url="rtsp://alice:secret@host/live?token=hidden"))
    body = client.get("/test-token/health").text
    for secret in ("alice", "secret", "hidden"):
        assert secret not in body


def test_unconfigured_source_shows_setup_and_refuses_stream():
    with TestClient(create_app(token="test-token", source_url="")) as client:
        health = client.get("/test-token/health").json()
        assert health["configured"] is False
        assert "Setup required" in client.get("/test-token").text
        assert client.get("/test-token/stream").status_code == 503


def test_cat_confidence_only_matches_cat_above_threshold():
    from types import SimpleNamespace
    from app import cat_confidence

    result = SimpleNamespace(
        names={0: "person", 15: "cat", 16: "dog"},
        boxes=SimpleNamespace(cls=[0, 15, 16], conf=[0.99, 0.72, 0.98]),
    )
    assert cat_confidence(result, 0.6) == 0.72


def test_cat_confidence_ignores_below_threshold_and_non_cat():
    from types import SimpleNamespace
    from app import cat_confidence

    result = SimpleNamespace(
        names={0: "person", 15: "cat"},
        boxes=SimpleNamespace(cls=[0, 15], conf=[0.99, 0.59]),
    )
    assert cat_confidence(result, 0.6) is None


def test_cat_alert_cooldown_never_faster_than_ten_seconds(tmp_path, monkeypatch):
    from app import Viewer

    monkeypatch.setenv("YOLO_CAT_ALERT_TARGET", "telegram:-100123:4478")
    monkeypatch.setenv("YOLO_CAT_ALERT_COOLDOWN_SECONDS", "5")
    viewer = Viewer("")
    viewer.alert_state_path = tmp_path / "last-alert-time"
    assert viewer._claim_cat_alert_slot(now=100.0) is True
    assert viewer._claim_cat_alert_slot(now=109.99) is False
    assert viewer._claim_cat_alert_slot(now=110.0) is True


def test_cat_alert_cooldown_survives_viewer_restart(tmp_path, monkeypatch):
    from app import Viewer

    monkeypatch.setenv("YOLO_CAT_ALERT_TARGET", "telegram:-100123:4478")
    state_path = tmp_path / "last-alert-time"
    first = Viewer("")
    first.alert_state_path = state_path
    assert first._claim_cat_alert_slot(now=100.0) is True

    restarted = Viewer("")
    restarted.alert_state_path = state_path
    assert restarted._claim_cat_alert_slot(now=109.99) is False
    assert restarted._claim_cat_alert_slot(now=110.0) is True


def test_cat_detection_dispatches_only_one_annotated_photo_per_cooldown(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import app
    from app import Viewer

    monkeypatch.setenv("YOLO_CAT_ALERT_TARGET", "telegram:-100123:4478")
    viewer = Viewer("")
    viewer.alert_state_path = tmp_path / "last-alert-time"
    sent = []
    viewer._send_cat_photo = lambda annotated_frame, confidence: sent.append((annotated_frame, confidence))

    class ImmediateThread:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(app.threading, "Thread", ImmediateThread)
    result = SimpleNamespace(
        names={15: "cat"},
        boxes=SimpleNamespace(cls=[15], conf=[0.83]),
    )
    class Frame:
        def copy(self):
            return Frame()

    frame = Frame()

    viewer._maybe_send_cat_alert(result, frame, now=100.0)
    viewer._maybe_send_cat_alert(result, frame, now=109.99)

    assert len(sent) == 1
    assert sent[0][1] == 0.83
    assert sent[0][0] is not frame


def test_cat_photo_uses_hermes_sender_and_removes_private_temp_file(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from app import Viewer, cv2

    target = "telegram:-100123:4478"
    monkeypatch.setenv("YOLO_CAT_ALERT_TARGET", target)
    viewer = Viewer("")
    viewer.alert_dir = tmp_path
    monkeypatch.setattr("app.shutil.which", lambda _: "/mock/hermes")
    sent = []

    def fake_run(command, **kwargs):
        sent.append(command)
        caption = command[-1]
        image_path = Path(caption.split("MEDIA:", 1)[1])
        assert image_path.exists()
        assert image_path.read_bytes().startswith(b"\xff\xd8")
        assert image_path.stat().st_mode & 0o777 == 0o600
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("app.subprocess.run", fake_run)
    viewer._send_cat_photo(cv2.UMat(16, 16, cv2.CV_8UC3), 0.82)

    assert sent[0][1:5] == ["send", "--to", target, "--quiet"]
    assert "YOLO detected cat (82%)" in sent[0][-1]
    assert list(tmp_path.iterdir()) == []
    assert viewer.last_cat_alert is not None


def test_prediction_ignores_bowl_class_case_insensitively():
    from app import predict_without_classes

    class Model:
        names = {0: "person", 15: "cat", 45: "Bowl"}

        def predict(self, frame, **kwargs):
            self.kwargs = kwargs
            return ["result"]

    model = Model()
    assert predict_without_classes(model, "frame", "mps") == "result"
    assert model.kwargs == {"device": "mps", "verbose": False, "classes": [0, 15]}
