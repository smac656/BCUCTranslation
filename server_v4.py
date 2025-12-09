"""
server_v4.py

Architecture:
- FastAPI server with:
  - /start_session, /end_session, /start_recorder, /stop_recorder
  - /audio_chunk POST endpoint (recorder_v2.py posts chunks here)
  - /ws/operator and /ws/attendee WebSockets
- Uses ONE long-lived OpenAI Realtime WebSocket session for transcription.
- Audio chunks from /audio_chunk are pushed into an internal queue.
- RealtimeTranscriber consumes audio, sends to OpenAI Realtime, and
  yields FINAL transcript segments back to the server.
- Server then runs translate + TTS (same as v3) and broadcasts to attendees.
"""

import sys
import os
import re
import asyncio
import time
import base64
import json
import subprocess
import wave
import logging
from typing import List, Optional

import numpy as np
import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------
# Environment / basic setup
# ---------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGUAGES = [s.strip() for s in os.getenv("LANGUAGES", "zh,es").split(",") if s.strip()]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_v4")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Global state: websockets, session, recorder
# ---------------------------------------------------------
attendees: List[WebSocket] = []
operators: List[WebSocket] = []

session_active = False
recorder_proc: Optional[subprocess.Popen] = None

# ---------------------------------------------------------
# Helper: RMS-based silence filter (like v3)
# ---------------------------------------------------------
def is_chunk_loud_enough(
    wav_path: str,
    rms_threshold: float = 0.004,  # slightly more permissive
    min_duration: float = 0.15,    # allow shorter spoken chunks
) -> bool:
    """
    Returns True if the WAV chunk has RMS above threshold and duration above min_duration.
    Also logs RMS + duration so we can see what's being filtered.
    """
    try:
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            duration = n_frames / float(framerate or 16000)

            frames = wf.readframes(n_frames)
            if not frames:
                logger.info("RMS check: no frames, treating as silence")
                return False

            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                logger.info("RMS check: empty samples, treating as silence")
                return False

            rms = float(np.sqrt(np.mean(samples ** 2)))
            logger.info(
                "RMS check: duration=%.3fs, rms=%.5f (threshold=%.5f)",
                duration,
                rms,
                rms_threshold,
            )

            if duration < min_duration:
                logger.info(
                    "RMS check: duration below min_duration (%.3f < %.3f) -> silence",
                    duration,
                    min_duration,
                )
                return False

            return rms >= rms_threshold
    except Exception as e:
        logger.warning(f"Failed RMS check: {e}; letting chunk through (fail-open).")
        return True  # fail-open if we cannot measure


# ---------------------------------------------------------
# MODE / config (kept simple for v4)
# ---------------------------------------------------------
BALANCED_CONFIG = {
    "max_buffer_chars": 400,  # not heavily used in v4, left for future
}

MODE_PROFILES = {"balanced": BALANCED_CONFIG}
CURRENT_MODE = MODE_PROFILES["balanced"]


# ---------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------
async def broadcast_status():
    payload = json.dumps(
        {
            "type": "status",
            "session_active": session_active,
            "recorder_running": recorder_proc is not None and recorder_proc.poll() is None,
        }
    )

    # operators
    to_remove = []
    for ws in operators:
        try:
            await ws.send_text(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in operators:
            operators.remove(ws)

    # attendees
    to_remove = []
    for ws in attendees:
        try:
            await ws.send_text(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in attendees:
            attendees.remove(ws)


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


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------
class RecorderControl(BaseModel):
    device: Optional[str] = None
    samplerate: Optional[int] = 16000


# ---------------------------------------------------------
# finalize_buffer_with_text
# ---------------------------------------------------------
async def finalize_buffer_with_text(text: str):
    """
    In v4 we don't use a final_queue worker like v3.
    This helper just normalizes and forwards to handle_final_transcript().
    """
    text = (text or "").strip()
    if not text:
        return
    await handle_final_transcript(text)


# ---------------------------------------------------------
# Realtime transcriber wrapper
# ---------------------------------------------------------
class RealtimeTranscriber:
    """
    Handles a persistent WebSocket connection to OpenAI's Realtime API.
    Audio PCM is queued here, then encoded + sent as proper JSON events.
    """

    def __init__(self):
        self.ws = None
        self.running = False
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.loop_task: Optional[asyncio.Task] = None

    async def start(self):
        if self.running:
            return

        self.running = True
        self.loop_task = asyncio.create_task(self._run_loop())
        logger.info("RealtimeTranscriber started")

    async def stop(self):
        self.running = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self.loop_task:
            # give the loop a moment to unwind
            await asyncio.sleep(0.1)
        logger.info("RealtimeTranscriber stopped")

    async def _run_loop(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                max_size=None,
            ) as ws:
                self.ws = ws
                logger.info("Connected to OpenAI Realtime WebSocket")

                # Override default chatty instructions: transcription-only
                try:
                    await ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "instructions": (
                                "You are a live speech transcription engine for church sermons. "
                                "Your ONLY job is to transcribe the user's spoken audio into text. "
                                "Do NOT greet, chat, ask questions, or add commentary. "
                                "Output only the words that are spoken, as clear text."
                            ),
                            # We keep other settings as-is; server VAD is already enabled
                            # and will trigger responses that carry transcription.
                        },
                    }))
                except Exception:
                    logger.exception("Failed to send session.update for transcription-only mode")

                # Start consumer and producer tasks
                sender = asyncio.create_task(self._sender_loop())
                receiver = asyncio.create_task(self._receiver_loop())

                await asyncio.wait(
                    [sender, receiver],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

        except Exception as e:
            logger.error("Error in RealtimeTranscriber main loop", exc_info=e)
        finally:
            self.running = False
            logger.warning("RealtimeTranscriber main loop ended")

    async def _sender_loop(self):
        """
        Correct Realtime API audio sender.

        Sends PCM16 bytes as:
            { "type": "input_audio_buffer.append",
              "audio": "<base64 pcm16>" }

        NOTE:
        - We rely on *server VAD*, so we do NOT send
          input_audio_buffer.commit ourselves. This avoids
          'commit_empty' errors and lets the model decide
          turn boundaries.
        """
        while self.running:
            # Raw PCM16 bytes from /audio_chunk
            pcm = await self.audio_queue.get()
            if not pcm:
                continue

            # Base64 encode
            b64 = base64.b64encode(pcm).decode("ascii")

            payload = {
                "type": "input_audio_buffer.append",
                "audio": b64,
            }

            logger.info("SENDING EVENT: %s", json.dumps(payload)[:200])

            try:
                await self.ws.send(json.dumps(payload))
            except ConnectionClosed:
                logger.warning("Realtime WebSocket closed while sending audio")
                break
            except Exception:
                logger.exception("Error sending audio to Realtime WebSocket")
                break


    async def _receiver_loop(self):
        """
        Receives events from Realtime API and:
          - Logs them for debugging
          - Extracts partial + final text from the *actual* events we see:
              - response.audio_transcript.delta
              - response.audio_transcript.done
          - Sends to operator / translation pipeline

        We also keep support for:
          - conversation.item.input_audio_transcription.delta/completed
          - response.text.delta / response.text.done
          - response.output_text.delta / response.completed
        (for forwards compatibility)
        """
        while self.running:
            try:
                msg = await self.ws.recv()
            except Exception as e:
                logger.warning("Realtime receiver loop error/closed: %r", e)
                break

            # Log raw message (truncated for safety)
            logger.info("Realtime raw event: %s", msg[:800])

            try:
                data = json.loads(msg)
            except Exception:
                logger.exception("Failed to parse realtime JSON")
                continue

            evt_type = data.get("type")
            logger.info("Realtime event type: %s", evt_type)

            # =====================================================
            #  A) NEW: audio transcript events (the ones you see)
            # =====================================================
            if evt_type == "response.audio_transcript.delta":
                # Shape: {"type":"response.audio_transcript.delta", "delta":"Thanks", ...}
                text = data.get("delta") or ""
                logger.info("response.audio_transcript.delta text: %r", text)
                if text:
                    await broadcast_to_operators({"type": "partial", "text": text})
                continue

            if evt_type == "response.audio_transcript.done":
                # Shape: {"type":"response.audio_transcript.done","transcript":"Thanks for your question."}
                transcript = data.get("transcript") or ""
                logger.info("response.audio_transcript.done transcript: %r", transcript)
                if transcript:
                    await broadcast_to_operators(
                        {"type": "english_text", "text": transcript}
                    )
                    await finalize_buffer_with_text(transcript)
                continue

            # =====================================================
            #  B) Transcription-specific conversation.* events
            #     (not currently seen in your logs, but kept)
            # =====================================================
            if evt_type == "conversation.item.input_audio_transcription.delta":
                delta = data.get("delta") or ""
                logger.info("Transcription delta (conversation.*): %r", delta)
                if delta:
                    await broadcast_to_operators({"type": "partial", "text": delta})
                continue

            if evt_type == "conversation.item.input_audio_transcription.completed":
                transcript = data.get("transcript") or ""
                logger.info("Transcription completed (conversation.*): %r", transcript)
                if transcript:
                    await broadcast_to_operators(
                        {"type": "english_text", "text": transcript}
                    )
                    await finalize_buffer_with_text(transcript)
                continue

            # =====================================================
            #  C) Text streaming events (if model ever emits them)
            # =====================================================
            if evt_type == "response.text.delta":
                text = data.get("delta") or ""
                logger.info("response.text.delta text: %r", text)
                if text:
                    await broadcast_to_operators({"type": "partial", "text": text})
                continue

            if evt_type == "response.text.done":
                text = data.get("text") or ""
                logger.info("response.text.done text: %r", text)
                if text:
                    await broadcast_to_operators({"type": "english_text", "text": text})
                    await finalize_buffer_with_text(text)
                continue

            # =====================================================
            #  D) Legacy-ish response.* events (fallback)
            # =====================================================
            if evt_type == "response.output_text.delta":
                delta = data.get("delta")
                if isinstance(delta, dict):
                    text = delta.get("text", "")
                else:
                    text = delta or ""
                logger.info("Legacy partial delta text: %r", text)
                if text:
                    await broadcast_to_operators({"type": "partial", "text": text})
                continue

            if evt_type == "response.completed":
                logger.info("response.completed payload: %s", data)
                response = data.get("response", {}) or {}
                text = response.get("output_text", "")

                if not text:
                    output = response.get("output")
                    text = _extract_text_from_output(output)

                logger.info("response.completed extracted text: %r", text)
                if text:
                    await broadcast_to_operators({"type": "english_text", "text": text})
                    await finalize_buffer_with_text(text)
                continue

            # =====================================================
            #  E) Everything else: just log, no action
            # =====================================================
            logger.info("Unhandled realtime event type: %s", evt_type)


# ---------------------------------------------------------
# Helper to extract text from Realtime "output"
# (VERY approximate; adjust to match actual OpenAI Realtime schema)
# ---------------------------------------------------------
def _extract_text_from_output(output):
    """
    Given event['response']['output'], try to extract text.
    This is approximate. Adjust with real Realtime response shape.
    """
    if not output:
        return ""
    try:
        first = output[0]
        content = first.get("content", [])
        if not content:
            return ""
        # content item might look like {"type":"text","text":"..."}
        for c in content:
            if isinstance(c, dict) and "text" in c:
                return c["text"]
    except Exception:
        logger.exception("Failed to extract text from output")
    return ""


# ---------------------------------------------------------
# Global transcriber instance
# ---------------------------------------------------------
realtime_transcriber = RealtimeTranscriber()


# ---------------------------------------------------------
# When a FINAL transcript is ready, we handle it here:
# - broadcast English to operators
# - translate + TTS to LANGUAGES, broadcast to attendees
# ---------------------------------------------------------
async def handle_final_transcript(text: str):
    text = text.strip()
    if not text:
        return

    # Clean text similar to v3
    text = re.sub(r"\s+", " ", text)

    # Broadcast English to operators
    await broadcast_to_operators({"type": "english_text", "text": text})

    # Translate in parallel
    translations = {}

    async def trans_task(lang):
        return lang, await call_openai_translate(text, lang)

    tasks = [trans_task(lang) for lang in LANGUAGES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
        if isinstance(res, Exception):
            logger.exception("Translation error")
        else:
            lang, tr = res
            translations[lang] = tr

    # TTS in parallel
    tts_audio_base64 = {}

    async def tts_task(lang, txt):
        try:
            audio_b64 = await call_openai_tts(txt)
            return lang, audio_b64
        except Exception:
            logger.exception("TTS error for %s", lang)
            return lang, None

    tts_tasks = [tts_task(lang, translations.get(lang, "")) for lang in translations.keys()]
    tts_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    for res in tts_results:
        if isinstance(res, Exception):
            logger.exception("TTS exception")
        else:
            lang, a = res
            tts_audio_base64[lang] = a

    # Broadcast to attendees
    payload = {
        "type": "translation",
        "text": translations,
        "tts": tts_audio_base64,
        "origin_text": text,
        "final_ts": time.time(),
    }
    await broadcast_to_attendees(payload)


# ---------------------------------------------------------
# Session / Recorder endpoints
# ---------------------------------------------------------
@app.post("/start_session")
async def start_session():
    global session_active
    session_active = True
    await realtime_transcriber.start()
    await broadcast_status()
    return {"status": "session_started"}


@app.post("/end_session")
async def end_session():
    global session_active
    session_active = False
    await stop_recorder_internal()
    await realtime_transcriber.stop()
    await broadcast_status()
    return {"status": "session_ended"}


@app.post("/start_recorder")
async def start_recorder(ctrl: RecorderControl):
    global recorder_proc
    if not session_active:
        return {"status": "error", "detail": "Cannot start recorder: session inactive"}

    if recorder_proc is not None and recorder_proc.poll() is None:
        return {"status": "recorder_already_running"}

    cmd = [sys.executable, "recorder_v2.py"]
    if ctrl.device:
        cmd += ["--device", str(ctrl.device)]
    if ctrl.samplerate:
        cmd += ["--samplerate", str(ctrl.samplerate)]

    recorder_proc = subprocess.Popen(cmd)
    await broadcast_status()
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
    except Exception:
        logger.exception("Error stopping recorder")
    recorder_proc = None
    await broadcast_status()
    return {"status": "recorder_stopped"}


@app.post("/stop_recorder")
async def stop_recorder():
    return await stop_recorder_internal()


# ---------------------------------------------------------
# WebSockets
# ---------------------------------------------------------
@app.websocket("/ws/operator")
async def ws_operator(ws: WebSocket):
    await ws.accept()
    operators.append(ws)

    await ws.send_text(
        json.dumps(
            {
                "type": "status",
                "session_active": session_active,
                "recorder_running": recorder_proc is not None
                and recorder_proc.poll() is None,
            }
        )
    )

    try:
        while True:
            _ = await ws.receive_text()
            # Future: operator commands here
    except WebSocketDisconnect:
        if ws in operators:
            operators.remove(ws)


@app.websocket("/ws/attendee")
async def ws_attendee(ws: WebSocket):
    await ws.accept()
    attendees.append(ws)

    await ws.send_text(
        json.dumps(
            {
                "type": "status",
                "session_active": session_active,
                "recorder_running": recorder_proc is not None
                and recorder_proc.poll() is None,
            }
        )
    )

    try:
        while True:
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        if ws in attendees:
            attendees.remove(ws)


# ---------------------------------------------------------
# /audio_chunk: forwards audio into RealtimeTranscriber
# ---------------------------------------------------------
@app.post("/audio_chunk")
async def receive_audio_chunk(request: Request):
    """
    Recorder posts chunks here (WAV or raw bytes).
    We:
      - ensure session is active
      - optionally apply RMS filter
      - push audio into realtime_transcriber.audio_queue as 16-bit PCM
    """
    global session_active

    if not session_active:
        return {"status": "session_inactive"}

    content_type = request.headers.get("content-type", "")
    raw_body = await request.body()

    audio_bytes = None
    try:
        if content_type.startswith("application/json"):
            payload = await request.json()
            b64 = payload.get("audio_base64")
            audio_bytes = base64.b64decode(b64) if b64 else None
        else:
            audio_bytes = raw_body
    except Exception as e:
        logger.exception("Failed to parse /audio_chunk")
        return {"status": "bad_request", "error": str(e)}

    if not audio_bytes:
        return {"status": "bad_request", "error": "no audio bytes"}

    # --- Convert to WAV on disk so we can reuse is_chunk_loud_enough ---
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    tmp_path = os.path.join(temp_dir, f"chunk_{int(time.time() * 1000)}.wav")

    try:
        if audio_bytes[:4] == b"RIFF":
            # Already WAV
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
        else:
            # Assume raw PCM 16-bit mono @ 16kHz
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
    except Exception as e:
        logger.exception("Failed to write temp WAV for /audio_chunk")
        return {"status": "error", "error": str(e)}

    # --- Silence filter ---
    try:
        if not is_chunk_loud_enough(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return {"status": "ok", "note": "silence_filtered"}
    except Exception:
        # If RMS check fails, fall back to letting chunk through
        logger.exception("RMS check failed in /audio_chunk")

    # --- Extract 16-bit PCM frames from the WAV into memory ---
    try:
        import io

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        # Read frames from in-memory WAV
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
    except Exception:
        logger.exception("Failed to extract PCM frames from WAV in /audio_chunk")
        # fall back: just push raw bytes
        frames = audio_bytes
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # --- Push into realtime_transcriber audio queue ---
    if realtime_transcriber is None or not realtime_transcriber.running:
        logger.warning("Received audio_chunk but RealtimeTranscriber not running")
        # We *drop* the audio in this case; there’s nowhere to send it.
        return {"status": "ok", "note": "transcriber_not_running"}

    try:
        await realtime_transcriber.audio_queue.put(frames)
    except Exception:
        logger.exception("Failed to enqueue audio frames into RealtimeTranscriber")
        return {"status": "error", "error": "queue_failure"}

    return {"status": "ok"}


# ---------------------------------------------------------
# OpenAI translate + TTS (same as v3)
# ---------------------------------------------------------
async def call_openai_translate(text: str, target_language: str) -> str:
    system_prompt = (
        "You are a concise translator. Translate the provided English sermon speech into the target language. "
        "Keep it short, literal, suitable for live captions."
    )
    user_prompt = f"Translate the following into {target_language} concisely:\n\n{text}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""


async def call_openai_tts(text: str) -> Optional[str]:
    if not text:
        return None
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": "gpt-4o-mini-tts", "voice": "alloy", "input": text}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/speech", json=payload, headers=headers
        )
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_v4:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
    )
