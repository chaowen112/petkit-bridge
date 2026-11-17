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
