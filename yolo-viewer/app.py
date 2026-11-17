from __future__ import annotations

import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import math
import shutil
import subprocess
import secrets
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import cv2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL", "yolo11n.pt")
INFERENCE_FPS = max(0.1, float(os.getenv("YOLO_INFERENCE_FPS", "2")))
ALERT_DIR = Path(__file__).resolve().parent / ".runtime"
IGNORED_DETECTION_CLASSES = frozenset({"bowl"})


def cat_confidence(result: Any, threshold: float) -> float | None:
    """Return the strongest cat detection at or above threshold."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    class_ids: Any = getattr(boxes, "cls", None)
    confidences: Any = getattr(boxes, "conf", None)
    if class_ids is None or confidences is None:
        return None
    if hasattr(class_ids, "tolist"):
        class_ids = class_ids.tolist()
    if hasattr(confidences, "tolist"):
        confidences = confidences.tolist()
    names = getattr(result, "names", {})
    strongest = None
    for class_id, confidence in zip(class_ids, confidences):
        if hasattr(class_id, "item"):
            class_id = class_id.item()
        if hasattr(confidence, "item"):
            confidence = confidence.item()
        try:
            class_id = int(class_id)
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if isinstance(names, dict):
            name = names.get(class_id, names.get(str(class_id)))
        else:
            try:
                name = names[class_id]
            except (IndexError, KeyError, TypeError):
                name = None
        if name == "cat" and confidence >= threshold:
            strongest = confidence if strongest is None else max(strongest, confidence)
    return strongest


def predict_without_classes(model: Any, frame: Any, device: str,
                            excluded_names=IGNORED_DETECTION_CLASSES):
    """Run inference without returning detections for ignored class labels."""
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    excluded = {str(name).strip().casefold() for name in excluded_names}
    classes = []
    for class_id, name in items:
        if str(name).strip().casefold() in excluded:
            continue
        try:
            classes.append(int(class_id))
        except (TypeError, ValueError):
            continue
    return model.predict(frame, device=device, verbose=False, classes=classes)[0]


def redact_source(url: str) -> str:
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "[configured]"


class Viewer:
    def __init__(self, source_url: str):
        self.source_url = source_url
        self.model = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.alert_target = os.getenv("YOLO_CAT_ALERT_TARGET", "").strip()
        try:
            confidence = float(os.getenv("YOLO_CAT_ALERT_MIN_CONFIDENCE", "0.6"))
        except ValueError:
            confidence = 0.6
        self.cat_alert_min_confidence = confidence if 0.0 <= confidence <= 1.0 else 0.6
        try:
            cooldown = float(os.getenv("YOLO_CAT_ALERT_COOLDOWN_SECONDS", "10"))
        except ValueError:
            cooldown = 10.0
        self.cat_alert_cooldown_seconds = max(10.0, cooldown) if math.isfinite(cooldown) else 10.0
        self._alert_lock = threading.Lock()
        self.alert_dir = ALERT_DIR
        self.alert_state_path = self.alert_dir / "cat_alert_last_sent"
        self._last_alert_time = None
        self.last_cat_alert = None
        self.cat_alert_error = None
        self.frame = None
        self.connected = False
        self.last_error = None
        self.last_inference = None
        self.running = False
        self.thread = None

    def start(self):
        if self.source_url:
            self.running = True
            self.thread = threading.Thread(target=self._capture, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _capture(self):
        while self.running:
            cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.connected = False
                self.last_error = "Unable to open RTSP source; retrying"
                cap.release()
                time.sleep(3)
                continue
            self.connected = True
            self.last_error = None
            last = 0.0
            try:
                while self.running:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    now = time.monotonic()
                    if now - last >= 1.0 / INFERENCE_FPS:
                        try:
                            if self.model is None:
                                self.model = YOLO(MODEL_NAME)
                            result = predict_without_classes(self.model, frame, self.device)
                            frame = result.plot()
                            self._maybe_send_cat_alert(result, frame)
                            self.last_inference = time.time()
                        except Exception as exc:
                            self.last_error = f"Inference failed ({type(exc).__name__})"
                            cv2.putText(frame, "YOLO inference unavailable", (12, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        self.frame = frame
                        last = now
            finally:
                self.connected = False
                cap.release()
            time.sleep(1)

    def _claim_cat_alert_slot(self, now: float | None = None) -> bool:
        """Enforce a hard one-photo-per-ten-seconds minimum cooldown."""
        if not self.alert_target:
            return False
        now = time.time() if now is None else now
        with self._alert_lock:
            try:
                previous = float(self.alert_state_path.read_text(encoding="ascii").strip())
            except FileNotFoundError:
                previous = self._last_alert_time
            except (OSError, ValueError):
                self.cat_alert_error = "Could not read cat alert cooldown"
                return False
            if previous is not None and now - previous < self.cat_alert_cooldown_seconds:
                return False
            temporary_path = None
            try:
                self.alert_state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(self.alert_state_path.parent, 0o700)
                fd, temporary_name = tempfile.mkstemp(
                    prefix=".cat-alert-state-", dir=self.alert_state_path.parent
                )
                temporary_path = Path(temporary_name)
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="ascii") as state_file:
                    state_file.write(f"{now:.6f}\n")
                    state_file.flush()
                    os.fsync(state_file.fileno())
                os.replace(temporary_path, self.alert_state_path)
                self._last_alert_time = now
                return True
            except OSError:
                self.cat_alert_error = "Could not persist cat alert cooldown"
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                return False

    def _maybe_send_cat_alert(self, result: Any, annotated_frame, now: float | None = None):
        confidence = cat_confidence(result, self.cat_alert_min_confidence)
        if confidence is None or not self._claim_cat_alert_slot(now):
            return
        try:
            threading.Thread(
                target=self._send_cat_photo,
                args=(annotated_frame.copy(), confidence),
                name="petkit-cat-alert",
                daemon=True,
            ).start()
        except RuntimeError:
            self.cat_alert_error = "Telegram photo send failed"

    def _send_cat_photo(self, annotated_frame, confidence: float) -> None:
        """Send one annotated JPEG through Hermes' configured Telegram transport."""
        image_path = None
        try:
            ok, encoded = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                self.cat_alert_error = "Could not encode cat snapshot"
                return
            self.alert_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.alert_dir, 0o700)
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix="petkit-cat-", suffix=".jpg", dir=self.alert_dir, delete=False
            ) as image_file:
                os.chmod(image_file.name, 0o600)
                image_file.write(encoded.tobytes())
                image_path = Path(image_file.name)
            hermes = shutil.which("hermes")
            if not hermes:
                fallback = Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
                hermes = str(fallback) if fallback.is_file() else None
            if not hermes:
                self.cat_alert_error = "Hermes send CLI unavailable"
                return
            caption = f"YOLO detected cat ({confidence:.0%})\nMEDIA:{image_path}"
            result = subprocess.run(
                [hermes, "send", "--to", self.alert_target, "--quiet", caption],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                self.last_cat_alert = time.time()
                self.cat_alert_error = None
            else:
                self.cat_alert_error = "Telegram photo send failed"
        except (OSError, subprocess.SubprocessError):
            self.cat_alert_error = "Telegram photo send failed"
        finally:
            if image_path is not None:
                try:
                    image_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def status(self):
        return {"configured": bool(self.source_url), "source": redact_source(self.source_url),
                "connected": self.connected, "model": MODEL_NAME, "device": self.device,
                "last_inference": self.last_inference, "error": self.last_error,
                "cat_alert_enabled": bool(self.alert_target),
                "cat_alert_min_confidence": self.cat_alert_min_confidence,
                "cat_alert_cooldown_seconds": self.cat_alert_cooldown_seconds,
                "ignored_classes": sorted(IGNORED_DETECTION_CLASSES),
                "last_cat_alert": self.last_cat_alert,
                "cat_alert_error": self.cat_alert_error}


PAGE = """<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>PetKit YOLO viewer</title>
<style>body{font:16px system-ui;background:#10151b;color:#eef;margin:2rem auto;max-width:960px;padding:0 1rem}img{width:100%;background:#222;border-radius:8px;min-height:240px;object-fit:contain}code{color:#9de}</style>
<h1>PetKit YOLO live viewer</h1><p id=s>Loading status…</p><img id=feed alt='Annotated RTSP stream' hidden>
<script>const token=location.pathname.split('/')[1], feed=document.querySelector('#feed'), status=document.querySelector('#s');
async function refresh(){try{let r=await fetch('/'+token+'/health');if(!r.ok)throw Error();let d=await r.json();status.textContent=d.error?(d.connected?'Connected · YOLO error: '+d.error:'Disconnected · '+d.error):(d.connected?'Connected · '+d.model+' · '+d.device:(d.configured?'Disconnected · waiting for RTSP frames':'Setup required · set YOLO_RTSP_URL'));if(d.connected){if(feed.hidden){feed.src='/'+token+'/stream';feed.hidden=false}}else{feed.hidden=true;feed.removeAttribute('src')}}catch{status.textContent='Viewer unavailable or access token invalid'}setTimeout(refresh,2000)}refresh();</script>"""


def create_app(token: str | None = None, source_url: str | None = None) -> FastAPI:
    token = token or os.getenv("YOLO_VIEWER_TOKEN") or secrets.token_urlsafe(32)
    source_url = os.getenv("YOLO_RTSP_URL", "") if source_url is None else source_url
    viewer = Viewer(source_url)

    @asynccontextmanager
    async def lifespan(app):
        viewer.start()
        yield
        viewer.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.viewer, app.state.token = viewer, token

    def authorize(path_token):
        if not secrets.compare_digest(path_token, token):
            raise HTTPException(404)

    @app.get("/{path_token}/health")
    def health(path_token: str):
        authorize(path_token)
        return viewer.status()

    @app.get("/{path_token}/stream")
    def stream(path_token: str):
        authorize(path_token)
        if not viewer.source_url:
            raise HTTPException(503, "YOLO_RTSP_URL is not configured")

        def frames():
            while True:
                frame = viewer.frame
                if frame is not None:
                    ok, data = cv2.imencode(".jpg", frame)
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data.tobytes() + b"\r\n"
                time.sleep(0.1)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/{path_token}", response_class=HTMLResponse)
    def index(path_token: str):
        authorize(path_token)
        return PAGE

    return app


app = create_app()
