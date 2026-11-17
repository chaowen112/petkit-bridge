"""Connect Mac-native YOLO to the HA-managed PetKit RTSP stream."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
STREAM_PREFIX = "petkit_"


def load_local_env(path: Path = ENV_FILE) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError("Create yolo-viewer/.env with the go2rtc connection settings first")
    os.chmod(path, 0o600)
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def validate_url(value: str, scheme: str, variable: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != scheme or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise RuntimeError(f"{variable} must be a {scheme} URL without credentials/query")
    return value.rstrip("/")


def validate_telegram_target(value: str) -> str:
    """Accept only an explicit Telegram chat or forum-topic target."""
    value = value.strip()
    if not value:
        return ""
    if not re.fullmatch(r"telegram:-?\d+(?::\d+)?", value):
        raise RuntimeError("Telegram target must be specified as telegram:<chat_id>[:<thread_id>]")
    return value


def api_get(api_base_url: str, path: str, username: str, password: str):
    credentials = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    request = Request(f"{api_base_url.rstrip('/')}{path}", headers={"Authorization": f"Basic {credentials}", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("go2rtc rejected API credentials; check the private .env") from None
        raise RuntimeError(f"go2rtc API returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise RuntimeError("Cannot reach the authenticated go2rtc API") from None


def find_petkit_stream(streams: dict, wanted: str = "") -> str:
    names = sorted(name for name in streams if isinstance(name, str) and name.startswith(STREAM_PREFIX))
    if wanted:
        if wanted not in names:
            raise RuntimeError("YOLO_STREAM_NAME is not registered by Home Assistant")
        return wanted
    if len(names) != 1:
        raise RuntimeError(f"Expected one PetKit stream, found {len(names)}; set YOLO_STREAM_NAME locally")
    return names[0]


def wait_for_petkit_stream(api_url: str, username: str, password: str, wanted: str = "", timeout: int = 90) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        streams = api_get(api_url, "/api/streams", username, password)
        names = [name for name in streams if isinstance(name, str) and name.startswith(STREAM_PREFIX)]
        if wanted and wanted in names:
            return wanted
        if wanted and names and wanted not in names:
            raise RuntimeError("YOLO_STREAM_NAME is not in the registered PetKit streams")
        if not wanted and len(names) == 1:
            return names[0]
        if len(names) > 1 and not wanted:
            raise RuntimeError("Multiple PetKit streams registered; set YOLO_STREAM_NAME locally")
        time.sleep(2)
    raise RuntimeError("No PetKit stream registered; check HA go2rtc config and restart")


def make_rtsp_url(base_url: str, username: str, password: str, stream_name: str) -> str:
    parts = urlsplit(base_url)
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parts.netloc}"
    path = parts.path.rstrip("/") + "/" + quote(stream_name, safe="_")
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def main() -> int:
    try:
        values = load_local_env()
        api_url = validate_url(values.get("GO2RTC_API_URL", ""), "http", "GO2RTC_API_URL")
        rtsp_base = validate_url(values.get("GO2RTC_RTSP_BASE_URL", ""), "rtsp", "GO2RTC_RTSP_BASE_URL")
        alert_target = validate_telegram_target(values.get("YOLO_CAT_ALERT_TARGET", ""))
        username = values.get("GO2RTC_API_USERNAME", "")
        password = values.get("GO2RTC_API_PASSWORD", "")
        if not username or not password:
            raise RuntimeError("Set GO2RTC_API_USERNAME and GO2RTC_API_PASSWORD locally")
        api_get(api_url, "/api", username, password)
        stream_name = wait_for_petkit_stream(api_url, username, password, values.get("YOLO_STREAM_NAME", ""))
        rtsp_url = values.get("YOLO_RTSP_URL") or make_rtsp_url(rtsp_base, username, password, stream_name)
    except RuntimeError as exc:
        print(f"Setup blocked: {exc}", file=sys.stderr)
        return 2

    viewer_token = values.get("YOLO_VIEWER_TOKEN") or secrets.token_urlsafe(32)
    viewer_host = values.get("YOLO_VIEWER_HOST", "127.0.0.1")
    link_host = values.get("YOLO_VIEWER_LINK_HOST", "localhost")
    port = int(values.get("YOLO_VIEWER_PORT", "8765"))
    os.environ["YOLO_VIEWER_TOKEN"] = viewer_token
    os.environ["YOLO_RTSP_URL"] = rtsp_url
    os.environ["YOLO_MODEL"] = values.get("YOLO_MODEL", "yolo11n.pt")
    os.environ["YOLO_INFERENCE_FPS"] = values.get("YOLO_INFERENCE_FPS", "2")
    os.environ["YOLO_CAT_ALERT_TARGET"] = alert_target
    os.environ["YOLO_CAT_ALERT_MIN_CONFIDENCE"] = values.get("YOLO_CAT_ALERT_MIN_CONFIDENCE", "0.3")
    os.environ["YOLO_CAT_ALERT_COOLDOWN_SECONDS"] = values.get("YOLO_CAT_ALERT_COOLDOWN_SECONDS", "10")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

    print(f"Viewer link (home LAN only): http://{link_host}:{port}/{viewer_token}", flush=True)
    import uvicorn
    uvicorn.run("app:app", host=viewer_host, port=port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
