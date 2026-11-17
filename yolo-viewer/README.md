# PetKit YOLO viewer

Mac-native YOLO inference and a browser view for a PetKit camera already integrated with Home Assistant. Uses Apple MPS when available; CPU fallback otherwise.

## Data path

```text
HA PetKit integration --signed WHEP--> lightweight go2rtc relay on HA's LAN
      --> authenticated RTSP --> YOLO/MPS on Mac --> token-protected MJPEG page
```

Only the small stream relay runs beside HA; YOLO inference stays on the Mac. The HA PetKit integration registers its signed WHEP source with go2rtc, so the viewer needs no HA token or second PetKit login.

## One-time setup

1. Run a small go2rtc service on the same trusted LAN network as HA (a macvlan network works well). Require Basic Auth for both the API and RTSP; do not expose either port to the WAN. Make the API and RTSP ports reachable only on the private LAN.
2. In Home Assistant's `configuration.yaml`, point the `go2rtc:` integration at the relay's LAN IP and set the matching API credentials:

   ```yaml
   go2rtc:
     url: http://<GO2RTC_LAN_IP>:1984
     username: "<GO2RTC_API_USERNAME>"
     password: "<GO2RTC_API_PASSWORD>"
   ```

   Back up the file and restart HA after this change. HA will register the PetKit stream with go2rtc using its signed WHEP URL.
3. Copy `.env.example` to `.env`; set `GO2RTC_API_URL`, `GO2RTC_RTSP_BASE_URL`, API username/password, and both viewer host fields to the Mac's LAN IP if you want phone access. Keep `.env` private; `run_local.py` enforces mode `0600`.
4. Install dependencies once: `uv sync`.
5. Start the viewer: `uv run python run_local.py`. It verifies the authenticated go2rtc API, detects the HA-registered `petkit_…` stream, and prints a link.
6. Open the link from a trusted device on the home LAN. The viewer requires an unguessable path token; do not publish it through the router or an unauthenticated public proxy.

The RTSP password stays in the local `.env`; `run_local.py` builds the URL in memory, and the status endpoint redacts user/password/query data. The Mac performs inference and serves the annotated stream; go2rtc only relays the PetKit feed. The relay's API and RTSP endpoints must both require Basic Auth.

If HA has multiple PetKit cameras, set `YOLO_STREAM_NAME` to the intended stream name. If no PetKit stream appears, check that HA's `go2rtc:` URL and credentials match the relay, and that the relay can reach HA on its LAN address. The first YOLO inference downloads `yolo11n.pt` if it is not cached and needs internet access.

## Validation

```sh
uv run pytest -q
```

Tests cover viewer authorization, source redaction, disconnected behavior, go2rtc stream selection, URL validation, and RTSP credential encoding. Static-image MPS inference has been verified; that is not proof that the live PetKit stream works. The live link is valid only after HA registers the stream and the Mac viewer receives real RTSP frames.
