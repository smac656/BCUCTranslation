import os
import json
import base64
import asyncio
import logging
import subprocess
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from websockets import connect as ws_connect

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_v4")

# ------------------------------------------------------------------------------
# Environment and constants
# ------------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")

# IMPORTANT: Correct transcription-capable realtime model
REALTIME_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"

OPENAI_REALTIME_URL = (
    f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
)

# ------------------------------------------------------------------------------
# FastAPI setup
# ------------------------------------------------------------------------------
app = FastAPI()

# Serve /static/* from ./static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global flags
session_active = False
recorder_process: Optional[subprocess.Popen] = None

# Operator WebSocket clients
operator_websockets = set()

# ------------------------------------------------------------------------------
# Realtime WebSocket client
# ------------------------------------------------------------------------------
class RealtimeTranscriber:
    def __init__(self):
        self.ws = None
        self.running = True

    async def connect(self):
        """
        Connect to OpenAI Realtime WebSocket.
        Retries until success.
        """
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        while True:
            try:
                logger.info("Connecting to OpenAI Realtime WS (transcription)…")

                self.ws = await ws_connect(
                    OPENAI_REALTIME_URL,
                    extra_headers=headers,
                    max_size=20_000_000,
                )

                logger.info("Connected to Realtime API.")

                # --------------------------------------------------------------
                # Send session.update to enforce TRANSCRIPTION-ONLY behavior
                # --------------------------------------------------------------
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio"],
                        "response_modalities": ["transcript"],
                        "temperature": 0,
                        "tool_choice": "none",

                        "instructions": (
                            "You perform speech-to-text transcription ONLY. "
                            "You MUST NOT greet, reply, comment, ask, or output assistant text. "
                            "Return ONLY transcript deltas of spoken audio."
                        ),

                        "input_audio_format": "pcm16",
                        "input_audio_mode": "buffer",

                        "input_audio_transcription": {
                            "model": "gpt-4o-mini-transcribe",
                            "language": "en",
                            "enabled": True
                        },

                        "turn_detection": {"type": "none"},
                    }
                }

                await self.ws.send(json.dumps(session_update))
                logger.info("Sent session.update for transcription-only mode.")

                return

            except Exception as e:
                logger.error(f"Realtime connection failed: {e}")
                await asyncio.sleep(3)

    async def send_audio_chunk(self, pcm_bytes: bytes):
        """
        Sends raw PCM16 audio to Realtime input_audio_buffer.
        """
        if not self.ws:
            return

        try:
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_bytes).decode("ascii"),
            }
            await self.ws.send(json.dumps(event))
        except Exception as e:
            logger.error(f"WS send failure: {e}")

    async def commit(self):
        """
        Commit buffered audio (end of utterance).
        """
        if not self.ws:
            return

        try:
            event = {"type": "input_audio_buffer.commit"}
            await self.ws.send(json.dumps(event))
        except Exception as e:
            logger.error(f"WS commit failure: {e}")

    async def receive_loop(self):
        """
        Reads realtime transcription events.
        Only processes transcript deltas.
        """
        if not self.ws:
            return

        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except:
                    continue

                event_type = data.get("type")

                # Only respond to transcript deltas
                if event_type == "response.audio_transcript.delta":
                    delta_text = data.get("delta")
                    if delta_text:
                        logger.info(f"TRANSCRIPT DELTA: {delta_text}")
                        await broadcast_to_operators({"type": "transcript", "delta": delta_text})

                elif event_type == "response.audio_transcript.done":
                    final_text = data.get("transcript", "")
                    logger.info(f"TRANSCRIPT DONE: {final_text}")
                    await broadcast_to_operators({"type": "transcript_done", "text": final_text})

                else:
                    # Ignore assistant messages or other content
                    pass

        except Exception as e:
            logger.error(f"Realtime WS error: {e}")

# Global realtime client
transcriber = RealtimeTranscriber()

# ------------------------------------------------------------------------------
# Broadcast helper
# ------------------------------------------------------------------------------
async def broadcast_to_operators(payload: dict):
    dead = []
    for ws in operator_websockets:
        try:
            await ws.send_json(payload)
        except:
            dead.append(ws)
    for ws in dead:
        operator_websockets.remove(ws)

# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse("static/operator.html")

@app.post("/start_session")
async def start_session():
    global session_active
    session_active = True
    logger.info("Session started via /start_session")

    # Start websocket receive loop
    asyncio.create_task(transcriber.receive_loop())

    return {"status": "ok"}

@app.post("/start_recorder")
async def start_recorder():
    global recorder_process

    if recorder_process and recorder_process.poll() is None:
        return {"status": "already_running"}

    logger.info("Starting recorder_v2.py...")
    recorder_process = subprocess.Popen(["python", "recorder_v2.py"])
    return {"status": "ok"}

@app.post("/stop_recorder")
async def stop_recorder():
    global recorder_process

    if recorder_process and recorder_process.poll() is None:
        logger.info("Stopping recorder...")
        recorder_process.terminate()
        recorder_process = None
        return {"status": "ok"}

    logger.info("Recorder not running when /stop_recorder called.")
    return {"status": "not_running"}

@app.post("/audio_chunk")
async def receive_chunk(request: Request):
    """
    Receives PCM16 audio chunks from recorder_v2.py.
    Sends them to the Realtime API.
    """
    global session_active

    if not session_active:
        return {"status": "ignored", "reason": "no active session"}

    body = await request.body()
    logger.info(f"Received {len(body)} audio bytes (session_active=True)")

    await transcriber.send_audio_chunk(body)
    return {"status": "ok"}

@app.websocket("/ws/operator")
async def ws_operator(ws: WebSocket):
    await ws.accept()
    operator_websockets.add(ws)
    logger.info("Operator connected.")

    try:
        while True:
            await ws.receive_text()  # keep connection open
    except WebSocketDisconnect:
        operator_websockets.remove(ws)
        logger.info("Operator disconnected.")

# ------------------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(transcriber.connect())
