import os
import json
import base64
import asyncio
import logging
import subprocess
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
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
REALTIME_MODEL = "gpt-4o-realtime-preview-2025-01-28"

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
_commit_loop_task = None

# Operator WebSocket clients
operator_websockets = set()


# ------------------------------------------------------------------------------
# Realtime WebSocket client
# ------------------------------------------------------------------------------
class RealtimeTranscriber:
    def __init__(self):
        self.ws = None
        self.running = True
        #self.server_buffer_ms = 0
       

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

                # ----------------------------------------------------------
                # Wait for initial session.created from the server
                # ----------------------------------------------------------
                created_raw = await self.ws.recv()
                try:
                    created = json.loads(created_raw)
                    session_id = created.get("session", {}).get("id", "unknown")
                    logger.info("Realtime session created (id=%s)", session_id)
                except Exception:
                    logger.info("Realtime session created.")

                # ----------------------------------------------------------
                # Send session.update to enforce TRANSCRIPTION-ONLY behavior
                # ----------------------------------------------------------
                session_update = {
                    "type": "session.update",
                    "session": {
                        # We want audio input and text events so we can get
                        # input_audio_transcription.* events.
                        "modalities": ["audio", "text"],

                        # Realtime now enforces a minimum of 0.6
                        "temperature": 0.6,

                        # Tell the server what audio format we are sending
                        "input_audio_format": "pcm16",

                        # Use the built-in transcription engine on input audio
                        "input_audio_transcription": {
                            "model": "gpt-4o-transcribe",
                            "language": "en"
                        },

                        # Let the server detect utterance boundaries, but
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.4,
                            "prefix_padding_ms": 500,
                            "silence_duration_ms": 400,
                            "create_response": False,
                            "interrupt_response": False,
                        },

                        # Strong instruction, but we also enforce this
                        # in code by ignoring response.* events.
                        "instructions": (
                            "You are transcribing live spoken sermons and public readings. "
                            "Transcribe speech verbatim and literally. Expect Christian and biblical terminology."
                            "Do not paraphrase, summarize, or reinterpret. "
                            "Preserve wording as spoken, including religious and formal language. "
                            "Do not correct or simplify phrasing."
                        ),
                    },
                }

                await self.ws.send(json.dumps(session_update))
                logger.info("Sent session.update for transcription-only mode.")

                return

            except Exception as e:
                logger.error(f"Realtime connection failed: {e}")
                await asyncio.sleep(3)

    async def send_audio_chunk(self, pcm_bytes: bytes):
        """
        Sends raw PCM16 audio to Realtime input_audio_buffer and updates
        local pending_audio so the commit loop can know how much audio
        is buffered.
        """
        if not self.ws:
            return

        try:

            event = {
                "type": "input_audio_buffer.append",
                # REQUIRED field name is 'audio' (base64-encoded PCM16)
                "audio": base64.b64encode(pcm_bytes).decode("ascii"),
            }
            await self.ws.send(json.dumps(event))
        except Exception as e:
            logger.error(f"WS send failure: {e}")

    async def periodic_commit_loop(self):
        """
        Periodically commits audio to the Realtime API, but ONLY when
        enough buffered audio exists to avoid empty-commit errors.

        This keeps transcription flowing smoothly without relying entirely
        on server VAD.
        """

        MIN_MS = 120      # minimum audio required before committing
        INTERVAL = 0.5    # seconds between checks (max ~2 commits/sec)

        last_commit = 0.0
        loop = asyncio.get_event_loop()

        while self.running:
            await asyncio.sleep(INTERVAL)

            # Only commit when the server confirms enough buffered audio
            #if self.server_buffer_ms < MIN_MS:
            #    continue


            # --- Commit no faster than 2/sec ---
            now = loop.time()
            if now - last_commit < INTERVAL:
                continue

            try:
                await self.commit()
                last_commit = now
            except Exception as e:
                logger.error("Commit failed: %s", e)
                # do NOT stop the loop — continue running
                continue

    async def commit(self):
        """
        Commit buffered audio (end of utterance) and reset pending counter.
        """
        if not self.ws:
            return

        try:
            event = {"type": "input_audio_buffer.commit"}
            await self.ws.send(json.dumps(event))
            # IMPORTANT: reset local buffer length tracking
            #self.server_buffer_ms = 0
        except Exception as e:
            logger.error(f"WS commit failure: {e}")

    async def receive_loop(self):
        """
        Reads realtime transcription events.
        Only processes transcript deltas and input_audio_transcription events.
        """
        if not self.ws:
            return

        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                event_type = data.get("type")
                
                # Track server-confirmed appended audio
                #if event_type == "input_audio_buffer.appended":
                    # Provided by server: duration of appended audio in ms
                 #   ms = data.get("audio_duration_ms", 0)
                 #   self.server_buffer_ms += ms
                 #   continue
                
                important_types = {
                    "error",
                    "conversation.item.input_audio_transcription.delta",
                    "conversation.item.input_audio_transcription.completed",
                    "response.audio_transcript.delta",
                    "response.audio_transcript.done",
                }

                if event_type not in important_types:
                    # Ignore all the routine chatter from the model
                    continue

                logger.info("EVENT TYPE RECEIVED (important): %s", event_type)

                if event_type == "error":
                    logger.error(
                        "REALTIME SESSION ERROR: %s",
                        json.dumps(data, indent=2),
                    )
                    continue

                # ------------------------------------------------------
                # Correct transcription events:
                # These represent the USER’S spoken audio.
                # ------------------------------------------------------
                if event_type == "conversation.item.input_audio_transcription.delta":
                    delta_text = data.get("transcript", "")
                    if delta_text:
                        logger.info("INPUT TRANSCRIPT DELTA: %s", delta_text)
                        await broadcast_to_operators(
                            {"type": "transcript", "delta": delta_text}
                        )

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    final_text = data.get("transcript", "")
                    logger.info("INPUT TRANSCRIPT DONE: %s", final_text)
                    await broadcast_to_operators(
                        {"type": "transcript_done", "text": final_text}
                    )

                # ------------------------------------------------------
                # Ignore ALL assistant-generated audio/transcript events
                # (we never want the model to "talk back").
                # ------------------------------------------------------
                elif event_type.startswith("response.audio"):
                    logger.debug(
                        "Ignoring assistant audio/transcript event."
                    )
                    continue

                else:
                    # Future hooks if we ever want to log something else
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
        except Exception:
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

@app.post("/stop_session")
async def stop_session():
    global session_active, _commit_loop_task

    session_active = False
    transcriber.running = False

    if _commit_loop_task and not _commit_loop_task.done():
        _commit_loop_task.cancel()
        try:
            await _commit_loop_task
        except Exception:
            pass

    if transcriber.ws:
        await transcriber.ws.close()
        transcriber.ws = None

    logger.info("Session stopped.")

    return {"status": "stopped"}


@app.post("/start_recorder")
async def start_recorder():
    """
    Starts recorder_v2.py and begins the audio commit loop.
    Ensures the loop is not already running.
    """
    global recorder_process, _commit_loop_task

    # 1. Prevent double-start of recorder
    if recorder_process and recorder_process.poll() is None:
        logger.info("Recorder already running.")
        return {"status": "already_running"}

    # 2. Stop any old commit-loop
    if _commit_loop_task and not _commit_loop_task.done():
        logger.info("Cancelling old commit loop...")
        _commit_loop_task.cancel()
        try:
            await _commit_loop_task
        except Exception:
            pass

    # 3. Launch recorder_v2.py using same venv interpreter
    import sys
    venv_python = sys.executable

    logger.info("Starting recorder_v2.py using interpreter: %s", venv_python)
    recorder_process = subprocess.Popen([venv_python, "recorder_v2.py"])

    # 4. Start a new commit loop only AFTER recorder starts
    #logger.info("Starting new audio commit loop...")
    #_commit_loop_task = asyncio.create_task(transcriber.periodic_commit_loop())

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
    # logger.info("Received %d audio bytes (session_active=True)", len(body))

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
