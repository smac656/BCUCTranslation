# server_v4.py — PURE TRANSCRIPTION MODE
import base64
import asyncio
import json
import logging
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
import websockets

import os

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

CHUNK_SAVE_DIR = "./uploaded_audio_chunks"
os.makedirs(CHUNK_SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_v4")

app = FastAPI()

# Allow WS from any origin (adjust later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


# -----------------------------------------------------------------------------
# WEBSOCKET CONNECTION MANAGER
# -----------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.operator_ws: Optional[WebSocket] = None
        self.attendee_ws: Optional[WebSocket] = None

    async def connect_operator(self, websocket: WebSocket):
        await websocket.accept()
        self.operator_ws = websocket
        logger.info("Operator connected.")

    async def connect_attendee(self, websocket: WebSocket):
        await websocket.accept()
        self.attendee_ws = websocket
        logger.info("Attendee connected.")

    async def broadcast_operator(self, message: Dict[str, Any]):
        if self.operator_ws and self.operator_ws.client_state == WebSocketState.CONNECTED:
            await self.operator_ws.send_json(message)

    async def broadcast_attendee(self, message: Dict[str, Any]):
        if self.attendee_ws and self.attendee_ws.client_state == WebSocketState.CONNECTED:
            await self.attendee_ws.send_json(message)


manager = ConnectionManager()


# -----------------------------------------------------------------------------
# REALTIME TRANSCRIBER — TRANSCRIPTION ONLY MODE
# -----------------------------------------------------------------------------
class RealtimeTranscriber:
    """
    Handles Realtime API WebSocket session in **transcription-only mode**.
    """

    def __init__(self):
        self.ws = None

    async def connect(self):
        logger.info("Connecting to OpenAI Realtime WS (transcription-only)…")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        self.ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            extra_headers=headers,
            max_size=20_000_000,
        )
        logger.info("Connected to Realtime API.")

        # Tell OpenAI we want ONLY transcription — no text/audio generation
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["audio"],
                "response_mode": "none",   # important
                "instructions": "Transcribe spoken audio only. Do not generate responses.",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 300,
                    "prefix_padding_ms": 150,
                    "create_response": True,
                },
            },
        }

        await self.ws.send(json.dumps(session_update))
        logger.info("Sent session.update for transcription-only mode.")

    async def send_audio_bytes(self, pcm_bytes: bytes):
        """
        Append base64 PCM16 audio to the Realtime buffer,
        then commit so the model processes it.
        """
        if not self.ws:
            return

        event_append = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm_bytes).decode("ascii"),
        }
        await self.ws.send(json.dumps(event_append))

        # Always commit immediately — chunk boundaries come from recorder
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def listen_loop(self):
        """
        Handle transcription events only.
        """
        while True:
            try:
                raw = await self.ws.recv()
            except Exception as e:
                logger.error(f"Realtime WS error: {e}")
                break

            logger.debug(f"Realtime raw event: {raw}")

            try:
                event = json.loads(raw)
            except:
                continue

            etype = event.get("type")
            logger.info(f"Realtime event: {etype}")

            # We ONLY care about:
            # response.audio_transcript.delta
            # response.audio_transcript.done
            # Everything else is ignored silently

            if etype == "response.audio_transcript.delta":
                delta = event.get("delta", "")
                logger.info(f"TRANSCRIPT DELTA: {delta}")

                # Send partial transcript to operator only
                await manager.broadcast_operator({
                    "type": "transcript_delta",
                    "text": delta
                })

            elif etype == "response.audio_transcript.done":
                text = event.get("transcript", "")
                logger.info(f"TRANSCRIPT DONE: {text}")

                # Send final text to operator and attendee
                await manager.broadcast_operator({
                    "type": "transcript_done",
                    "text": text
                })
                await manager.broadcast_attendee({
                    "type": "transcript_done",
                    "text": text
                })

            else:
                # Ignore everything else
                pass


transcriber = RealtimeTranscriber()


# -----------------------------------------------------------------------------
# FASTAPI ROUTES
# -----------------------------------------------------------------------------
@app.post("/audio_chunk")
async def receive_audio_chunk(raw_audio: bytes):
    """
    Recorder sends PCM16 bytes here.
    """
    logger.info(f"Received {len(raw_audio)} bytes")

    # Forward to Realtime API
    await transcriber.send_audio_bytes(raw_audio)

    return {"status": "ok"}


@app.websocket("/ws/operator")
async def operator_ws(websocket: WebSocket):
    await manager.connect_operator(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Operator disconnected.")


@app.websocket("/ws/attendee")
async def attendee_ws(websocket: WebSocket):
    await manager.connect_attendee(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Attendee disconnected.")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(transcriber.connect())
    asyncio.create_task(transcriber.listen_loop())


# Simple root endpoint
@app.get("/")
async def root():
    return {"status": "transcription server running"}
