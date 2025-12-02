"""server_v2.py
FastAPI server v2 skeleton:
- WebSocket endpoints for attendees and operator
- /audio_chunk POST endpoint for recorder
- /start_session, /end_session endpoints
- /start_recorder, /stop_recorder endpoints (spawn/kill recorder subprocess)

Integrate your existing transcription/translation/TTS calls where marked.
"""

import asyncio
import base64
import json
import os
import signal
import subprocess
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles #REMOVE

app = FastAPI()

# Mount the "static" folder at root  REMOVE LATER
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected websockets
attendees: List[WebSocket] = []
operators: List[WebSocket] = []

# Session / recorder process state
session_active = False
recorder_proc: Optional[subprocess.Popen] = None

class RecorderControl(BaseModel):
    device: Optional[str] = None
    samplerate: Optional[int] = 16000

@app.post("/start_session")
async def start_session():
    global session_active
    session_active = True
    return {"status": "session_started"}

@app.post("/end_session")
async def end_session():
    global session_active
    session_active = False
    # Optionally stop recorder if running
    await stop_recorder_internal()
    return {"status": "session_ended"}

@app.post("/start_recorder")
async def start_recorder(ctrl: RecorderControl):
    """Spawn recorder_v2.py as a subprocess. Pass device and samplerate as args."""
    global recorder_proc
    if recorder_proc is not None and recorder_proc.poll() is None:
        return {"status": "recorder_already_running"}

    cmd = ["python", "recorder_v2.py"]
    if ctrl.device:
        cmd += ["--device", str(ctrl.device)]
    if ctrl.samplerate:
        cmd += ["--samplerate", str(ctrl.samplerate)]

    recorder_proc = subprocess.Popen(cmd)
    return {"status": "recorder_started", "pid": recorder_proc.pid}

async def stop_recorder_internal():
    global recorder_proc
    if recorder_proc is None:
        return {"status": "no_recorder"}
    try:
        recorder_proc.terminate()
        try:
            recorder_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            recorder_proc.kill()
    except Exception as e:
        print("Error stopping recorder:", e)
    recorder_proc = None
    return {"status": "recorder_stopped"}

@app.post("/stop_recorder")
async def stop_recorder():
    return await stop_recorder_internal()

@app.websocket("/ws/attendee")
async def ws_attendee(ws: WebSocket):
    await ws.accept()
    attendees.append(ws)
    try:
        while True:
            # Currently we don't expect to receive messages, but could handle ping or language selection
            msg = await ws.receive_text()
            # Optionally handle messages from attendees (e.g., selected language)
    except WebSocketDisconnect:
        attendees.remove(ws)

@app.websocket("/ws/operator")
async def ws_operator(ws: WebSocket):
    await ws.accept()
    operators.append(ws)
    try:
        while True:
            msg = await ws.receive_text()
            # Operator may send control messages in future
    except WebSocketDisconnect:
        operators.remove(ws)

@app.post("/audio_chunk")
async def receive_audio_chunk(request: Request):
    """Endpoint for recorder to POST raw audio bytes (WAV/PCM) or a base64 payload.
    Expected: binary body or JSON {"audio_base64": "...", "duration": 0.5}
    """
    if not session_active:
        return {"status": "session_inactive"}

    content_type = request.headers.get("content-type", "")
    data = await request.body()

    # Support either raw bytes or JSON with base64
    audio_bytes = None
    try:
        if content_type.startswith("application/json"):
            payload = await request.json()
            b64 = payload.get("audio_base64")
            audio_bytes = base64.b64decode(b64)
        else:
            audio_bytes = data
    except Exception as e:
        print("Failed to parse chunk:", e)
        return {"status": "bad_request", "error": str(e)}

    # TODO: send audio_bytes to your transcription engine (Whisper/OpenAI) and obtain english_text
    # Placeholder - replace with your ASR call
    english_text = "[ASR placeholder text]"

    # TODO: translate english_text to target languages (e.g., via OpenAI translate or other)
    translations = {"en": english_text, "es": "[Spanish placeholder]", "zh": "[Chinese placeholder]"}

    # TODO: generate TTS for translations and either stream audio chunks or send base64 audio to clients
    # Placeholder: no audio content currently

    # Broadcast english text to operators
    await broadcast_to_operators({"type": "english_text", "text": english_text})

    # Broadcast translations to attendees
    await broadcast_to_attendees({"type": "translation", "text": translations})

    return {"status": "ok"}

async def broadcast_to_attendees(message: dict):
    payload = json.dumps(message)
    to_remove = []
    for ws in attendees:
        try:
            await ws.send_text(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in attendees:
            attendees.remove(ws)

async def broadcast_to_operators(message: dict):
    payload = json.dumps(message)
    to_remove = []
    for ws in operators:
        try:
            await ws.send_text(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in operators:
            operators.remove(ws)

async def broadcast_status():
    """Send session/recorder status to all attendees."""
    while True:
        message = {
            "type": "status",
            "session_active": session_active,
            "recorder_running": recorder_proc is not None and recorder_proc.poll() is None
        }
        await broadcast_to_attendees(message)
        await asyncio.sleep(2)  # update every 2 seconds

# Run this in background at server startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_status())

            

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_v2:app", host="0.0.0.0", port=8000, reload=False)
