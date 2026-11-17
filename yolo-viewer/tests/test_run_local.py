import pytest
from urllib.parse import unquote, urlsplit

from run_local import find_petkit_stream, make_rtsp_url, validate_url


def test_api_and_rtsp_urls_reject_embedded_credentials():
    assert validate_url("http://relay.example:1984/", "http", "GO2RTC_API_URL") == "http://relay.example:1984"
    assert validate_url("rtsp://relay.example:8554/", "rtsp", "GO2RTC_RTSP_BASE_URL") == "rtsp://relay.example:8554"
    with pytest.raises(RuntimeError):
        validate_url("http://user:password@relay.example:1984", "http", "GO2RTC_API_URL")
    with pytest.raises(RuntimeError):
        validate_url("http://relay.example:8554", "rtsp", "GO2RTC_RTSP_BASE_URL")


def test_find_petkit_stream_selects_one_or_explicit_choice():
    streams = {"petkit_device-1": {}, "other": {}}
    assert find_petkit_stream(streams) == "petkit_device-1"
    assert find_petkit_stream(streams, "petkit_device-1") == "petkit_device-1"


def test_find_petkit_stream_requires_selection_when_ambiguous():
    with pytest.raises(RuntimeError, match="found 0"):
        find_petkit_stream({"other": {}})
    with pytest.raises(RuntimeError, match="found 2"):
        find_petkit_stream({"petkit_one": {}, "petkit_two": {}})


def test_rtsp_url_uses_basic_auth_and_encodes_credentials():
    result = make_rtsp_url("rtsp://relay.example:8554", "viewer", "fake:password", "petkit_device-1")
    parts = urlsplit(result)
    assert parts.hostname == "relay.example"
    assert parts.port == 8554
    assert parts.username == "viewer"
    assert parts.password is not None
    assert unquote(parts.password) == "fake:password"
    assert parts.path == "/petkit_device-1"


def test_telegram_alert_target_requires_a_numeric_telegram_chat():
    from run_local import validate_telegram_target

    assert validate_telegram_target("telegram:-100123:4478") == "telegram:-100123:4478"
    assert validate_telegram_target("") == ""
    with pytest.raises(RuntimeError, match="Telegram target"):
        validate_telegram_target("https://example.com/")
