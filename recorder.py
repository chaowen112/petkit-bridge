"""Local, segmented NAS recorder for the PETKIT YumShare Solo camera."""

import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

LOG = logging.getLogger("petkit-recorder")
STREAM_NAME = "petkit_yumshare_solo"


def select_d4h_device(devices, states, configured_id=None):
    """Select the one camera-capable YumShare Solo (device type d4h)."""
    device_map = {
        str(device.get("id")): device
        for device in devices
        if device.get("id") is not None
    }
    type_map = {
        str(device_id): str(state.get("device_type", "")).lower()
        for device_id, state in states.items()
        if isinstance(state, dict)
    }
    candidates = [
        device_id
        for device_id, device in device_map.items()
        if device.get("camera") is True and type_map.get(device_id) == "d4h"
    ]

    if configured_id:
        selected = str(configured_id)
        if selected not in candidates:
            raise ValueError("Configured device is not a YumShare Solo camera")
        return selected
    if not candidates:
        raise ValueError("No YumShare Solo camera found")
    if len(candidates) > 1:
        raise ValueError("Multiple YumShare Solo cameras found; set PETKIT_DEVICE_ID")
    return candidates[0]


def build_whep_source(bridge_url, device_id, token):
    """Create go2rtc's upstream WHEP URI without logging its token."""
    base = bridge_url.rstrip("/")
    safe_id = quote(str(device_id), safe="")
    safe_token = quote(str(token), safe="")
    return (
        "webrtc:"
        + base
        + "/device/"
        + safe_id
        + "/whep?token="
        + safe_token
        + "#format=whep"
    )


def go2rtc_registration_url(api_url, stream_name, source):
    query = urlencode({"name": stream_name, "src": source})
    return api_url.rstrip("/") + "/api/streams?" + query


def build_ffmpeg_command(rtsp_url, output_pattern, segment_seconds, record_audio=False):
    if int(segment_seconds) <= 0:
        raise ValueError("segment_seconds must be positive")
    if "%" not in output_pattern:
        raise ValueError("output_pattern must contain a strftime placeholder")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-n",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        "15000000",
        "-i",
        rtsp_url,
        "-map",
        "0:v:0",
    ]
    if record_audio:
        command.extend(["-map", "0:a?"])
    command.extend(
        [
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(int(segment_seconds)),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            "-segment_format",
            "matroska",
            output_pattern,
        ]
    )
    return command


def parse_retention_days(value):
    """Enforce the user's hard 30-day maximum; blank defaults to 30."""
    if value is None or str(value).strip() == "":
        return 30
    try:
        days = int(value)
    except (TypeError, ValueError):
        raise ValueError("PETKIT_RECORDING_RETENTION_DAYS must be from 1 to 30") from None
    if days < 1 or days > 30:
        raise ValueError("PETKIT_RECORDING_RETENTION_DAYS must be from 1 to 30")
    return days


def cleanup_old_recordings(directory, retention_days, now=None):
    """Delete expired recorder segments; retention is always capped at 30 days."""
    retention_days = parse_retention_days(retention_days)

    root = Path(directory)
    if not root.is_dir():
        return 0
    current_time = time.time() if now is None else float(now)
    cutoff = current_time - retention_days * 86400
    removed = 0
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        if not (path.name.startswith("yumshare-solo-") and path.suffix == ".mkv"):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _json_request(url, token=None, timeout=8):
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Auth-Token"] = token
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError("local service returned HTTP " + str(exc.code)) from None
    except URLError:
        raise RuntimeError("local service is not ready") from None
    except TimeoutError:
        raise RuntimeError("local service timed out") from None


def _wait_for_bridge(bridge_url, stop_event):
    last_notice = 0.0
    while not stop_event.is_set():
        try:
            status = _json_request(bridge_url.rstrip("/") + "/healthz?strict=1")
            if status.get("status") == "ok":
                return True
        except (RuntimeError, ValueError):
            pass
        now = time.monotonic()
        if now - last_notice >= 60:
            LOG.info("Waiting for PETKIT bridge/cloud session readiness")
            last_notice = now
        stop_event.wait(5)
    return False


def _wait_for_go2rtc(api_url, stop_event):
    last_notice = 0.0
    while not stop_event.is_set():
        try:
            _json_request(api_url.rstrip("/") + "/api")
            return True
        except (RuntimeError, ValueError):
            now = time.monotonic()
            if now - last_notice >= 60:
                LOG.info("Waiting for local go2rtc API")
                last_notice = now
            stop_event.wait(5)
    return False


def _discover_camera(bridge_url, token, configured_id=None):
    base = bridge_url.rstrip("/")
    devices = _json_request(base + "/devices", token=token)
    if not isinstance(devices, list):
        raise ValueError("bridge returned an invalid device list")
    states = {}
    for device in devices:
        device_id = device.get("id")
        if device_id is None:
            continue
        if configured_id and str(device_id) != str(configured_id):
            continue
        states[str(device_id)] = _json_request(
            base + "/device/" + quote(str(device_id), safe="") + "/state",
            token=token,
        )
    selected = select_d4h_device(devices, states, configured_id)
    selected_device = next(device for device in devices if str(device.get("id")) == selected)
    return selected, selected_device.get("name") or "YumShare Solo"


def _register_go2rtc_stream(api_url, stream_name, source):
    streams = _json_request(api_url.rstrip("/") + "/api/streams")
    method = "PATCH" if stream_name in streams else "PUT"
    request = Request(
        go2rtc_registration_url(api_url, stream_name, source),
        method=method,
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=10):
            pass
    except HTTPError as exc:
        raise RuntimeError("go2rtc stream registration returned HTTP " + str(exc.code)) from None
    except (URLError, TimeoutError):
        raise RuntimeError("go2rtc stream registration failed") from None


def _terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_recorder():
    token = os.environ.get("BRIDGE_TOKEN", "").strip()
    if not token:
        raise ValueError("BRIDGE_TOKEN is required for local bridge access")

    bridge_url = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8787")
    go2rtc_api = os.environ.get("GO2RTC_API_URL", "http://127.0.0.1:1984")
    rtsp_url = os.environ.get(
        "GO2RTC_RTSP_URL", "rtsp://127.0.0.1:8554/petkit_yumshare_solo"
    )
    device_id = os.environ.get("PETKIT_DEVICE_ID", "").strip() or None
    recordings_dir = Path(os.environ.get("RECORDINGS_DIR", "/recordings"))
    segment_seconds = int(os.environ.get("PETKIT_RECORDING_SEGMENT_SECONDS", "300"))
    retention_days = parse_retention_days(
        os.environ.get("PETKIT_RECORDING_RETENTION_DAYS")
    )
    record_audio = os.environ.get("PETKIT_RECORDING_AUDIO", "0").lower() in {
        "1", "true", "yes", "on"
    }
    if segment_seconds <= 0:
        raise ValueError("PETKIT_RECORDING_SEGMENT_SECONDS must be positive")

    recordings_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    next_cleanup = 0.0
    while not stop_event.is_set():
        process = None
        try:
            if not _wait_for_bridge(bridge_url, stop_event):
                break
            if not _wait_for_go2rtc(go2rtc_api, stop_event):
                break
            selected_id, _name = _discover_camera(bridge_url, token, device_id)
            source = build_whep_source(bridge_url, selected_id, token)
            _register_go2rtc_stream(go2rtc_api, STREAM_NAME, source)
            command = build_ffmpeg_command(
                rtsp_url,
                str(recordings_dir / "yumshare-solo-%Y%m%d-%H%M%S.mkv"),
                segment_seconds,
                record_audio=record_audio,
            )
            LOG.info(
                "Recording YumShare Solo locally: segment=%ss audio=%s retention_days=%s",
                segment_seconds,
                record_audio,
                retention_days,
            )
            process = subprocess.Popen(command)
            next_cleanup = 0.0
            while not stop_event.is_set() and process.poll() is None:
                now = time.monotonic()
                if retention_days and now >= next_cleanup:
                    removed = cleanup_old_recordings(recordings_dir, retention_days)
                    if removed:
                        LOG.info("Expired local recording segments removed: %s", removed)
                    next_cleanup = now + 3600
                stop_event.wait(2)
            if stop_event.is_set():
                _terminate_process(process)
                break
            LOG.warning("ffmpeg recorder exited with status %s; retrying", process.returncode)
        except (RuntimeError, ValueError, OSError) as exc:
            # Do not log request URLs or API bodies; they may contain the bridge token.
            LOG.warning("Recorder setup/recovery failed: %s", str(exc))
        finally:
            if process is not None and stop_event.is_set():
                _terminate_process(process)
        stop_event.wait(10)


def main():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        _run_recorder()
    except (ValueError, OSError) as exc:
        LOG.error("Recorder configuration failed: %s", str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
