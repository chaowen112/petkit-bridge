from __future__ import annotations

import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import secrets
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

import cv2
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL", "yolo11n.pt")
INFERENCE_FPS = max(0.1, float(os.getenv("YOLO_INFERENCE_FPS", "2")))


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
                            result = self.model.predict(frame, device=self.device, verbose=False)[0]
                            frame = result.plot()
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

    def status(self):
        return {"configured": bool(self.source_url), "source": redact_source(self.source_url),
                "connected": self.connected, "model": MODEL_NAME, "device": self.device,
                "last_inference": self.last_inference, "error": self.last_error}


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
