# PetKit YOLO viewer — task template

## Objective
Serve an annotated, browser-viewable MJPEG stream from a PetKit bridge RTSP source using local YOLO inference on macOS (MPS when available, CPU fallback).

## Constraints
- Runs on Mac with uv-managed Python 3.11; never modifies or starts bridge/Home Assistant services.
- Configure RTSP URL through `YOLO_RTSP_URL`; never check in credentials.
- Access by unguessable `YOLO_VIEWER_TOKEN` path; default bind address is Ethernet `192.168.0.10` (override only deliberately). No unauthenticated health, UI, or stream.
- Logs/status redact RTSP username, password, query, and fragment. No fake feed when unconfigured.

## Implementation
- FastAPI UI, token-gated health and MJPEG routes; RTSP reconnect loop; inference rate limit; YOLO model/device status.
- YOLO package downloads the configured model on first inference, so first use needs internet access.

## Tests
```sh
uv run pytest -q
```
Tests cover route auth, source masking, and disconnected/unconfigured status. Runtime inference separately exercised with an openly available image when feasible; that is not a PetKit stream test.

## Deliverables
`app.py`, `pyproject.toml`, `uv.lock`, `.python-version`, tests, `.env.example`, and this run guide.

## Local run
1. `cd /Users/chaowen/go/src/petkit-bridge/yolo-viewer`
2. `cp .env.example .env`, then set `YOLO_RTSP_URL` to the bridge's RTSP URL and set `YOLO_VIEWER_TOKEN` to a generated random token. Never commit `.env`.
3. `set -a; source .env; set +a`
4. `uv sync`
5. `uv run uvicorn app:app --host 192.168.0.10 --port 8765 --no-access-log`
6. Visit `http://192.168.0.10:8765/$YOLO_VIEWER_TOKEN`; status is at the same path plus `/health`.

Before a configured bridge URL is available, health reports `configured:false`, stream returns 503, and the UI displays setup-required. Avoid exposing the LAN port via router/public proxy. This prototype's token is a URL path credential: don't share it with untrusted clients.
