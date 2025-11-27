"""
recorder.py - Robust VAD -> upload -> TTS playback recorder

How it works:
- Captures audio frames (30ms) from the mic (mono, int16).
- Uses webrtcvad to detect speech frames.
- Buffers contiguous speech frames; when silence > SILENCE_SECONDS or max chunk length reached,
  saves a WAV and uploads it to SERVER_URL (/transcribe).
- Plays TTS returned by server sequentially (queued).
"""

import os
import time
import uuid
import queue
import threading
import requests
import base64
import sounddevice as sd
import soundfile as sf
import webrtcvad
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ----------------- Config -----------------
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))   # must be 8000,16000,32000,48000
CHANNELS = int(os.getenv("CHANNELS", "1"))            # must be 1 (mono) for webrtcvad
VAD_MODE = int(os.getenv("VAD_MODE", "2"))            # 0..3 (3 = most aggressive)
FRAME_MS = 30                                         # 10,20 or 30 ms only
SILENCE_SECONDS = float(os.getenv("SILENCE_SECONDS", "0.7"))
MAX_CHUNK_SECONDS = float(os.getenv("MAX_CHUNK_SECONDS", "12"))
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Derived
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * CHANNELS * 2  # 2 bytes per sample for int16
MAX_CHUNK_FRAMES = int(MAX_CHUNK_SECONDS * 1000 / FRAME_MS)
MAX_SILENCE_FRAMES = int(SILENCE_SECONDS * 1000 / FRAME_MS)

# ----------------- Globals -----------------
vad = webrtcvad.Vad(VAD_MODE)
audio_queue = queue.Queue()       # raw bytes from callback
upload_queue = queue.Queue()      # filenames to upload (background)
play_queue = queue.Queue()        # tts dicts to play sequentially

# ----------------- Audio callback -----------------
def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback: push raw bytes (int16) to audio_queue."""
    if status:
        print("Audio device status:", status)
    # Ensure mono int16
    audio_queue.put(indata.copy().tobytes())

# ----------------- Uploader (background) -----------------
def uploader_worker():
    while True:
        filename = upload_queue.get()
        if not filename:
            continue
        try:
            with open(filename, "rb") as f:
                files = {"file": f}
                resp = requests.post(f"{SERVER_URL}/transcribe", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                print("Server response: english:", data.get("english"))
                for lang, txt in data.get("translations", {}).items():
                    print(f"  {lang}: {txt}")
                if "tts" in data and isinstance(data["tts"], dict):
                    play_queue.put(data["tts"])
        except Exception as e:
            print("Upload failed:", e)
        finally:
            try:
                os.remove(filename)
            except Exception:
                pass

# ----------------- Player (background) -----------------
def player_worker():
    while True:
        tts_dict = play_queue.get()
        if not tts_dict:
            continue
        # play languages sequentially; you can filter to preferred language(s)
        for lang, b64 in tts_dict.items():
            try:
                audio_bytes = base64.b64decode(b64)
                tmp_path = os.path.join(TEMP_DIR, f"tts_{lang}_{uuid.uuid4().hex[:6]}.wav")
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                data, sr = sf.read(tmp_path, dtype="int16")
                print(f"Playing TTS ({lang})")
                sd.play(data, sr)
                sd.wait()
                os.remove(tmp_path)
            except Exception as e:
                print(f"TTS playback failed ({lang}):", e)

# ----------------- Save WAV helper -----------------
def save_wav_from_frames(frames_bytes: bytes, path: str):
    """frames_bytes is raw int16 bytes concatenated for mono or multi-channel."""
    arr = np.frombuffer(frames_bytes, dtype="int16")
    if CHANNELS > 1:
        arr = arr.reshape(-1, CHANNELS)
    else:
        arr = arr.reshape(-1, 1)
    sf.write(path, arr, SAMPLE_RATE, subtype="PCM_16")

# ----------------- Main VAD loop -----------------
def vad_loop():
    print(f"Starting VAD loop: {SAMPLE_RATE}Hz mono, frame={FRAME_MS}ms, vad_mode={VAD_MODE}")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                            blocksize=FRAME_SAMPLES, callback=audio_callback)
    stream.start()

    # buffers
    speech_frames = []        # list of raw bytes frames (each FRAME_BYTES)
    silence_count = 0
    chunk_frames_count = 0

    try:
        while True:
            try:
                raw = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # split raw into exact FRAME_BYTES frames (should already be that size because blocksize)
            for offset in range(0, len(raw), FRAME_BYTES):
                frame = raw[offset:offset + FRAME_BYTES]
                if len(frame) != FRAME_BYTES:
                    # ignore partial frame
                    continue

                # VAD wants raw bytes (int16 PCM)
                try:
                    is_speech = vad.is_speech(frame, SAMPLE_RATE)
                except Exception as e:
                    print("VAD error on frame:", e)
                    is_speech = False

                # Debug (uncomment for heavy logging)
                # print("VAD:", is_speech)

                if is_speech:
                    speech_frames.append(frame)
                    silence_count = 0
                    chunk_frames_count += 1
                    # If chunk too long, force upload
                    if chunk_frames_count >= MAX_CHUNK_FRAMES:
                        print("Max chunk length reached, finalizing chunk.")
                        finalize_and_queue(speech_frames)
                        speech_frames = []
                        silence_count = 0
                        chunk_frames_count = 0
                else:
                    if speech_frames:
                        silence_count += 1
                        # Keep adding the small silent frame so final audio feels natural
                        speech_frames.append(frame)
                        # If we've seen enough silence frames, finalize chunk
                        if silence_count >= MAX_SILENCE_FRAMES:
                            print("Silence threshold reached, finalizing chunk.")
                            finalize_and_queue(speech_frames)
                            speech_frames = []
                            silence_count = 0
                            chunk_frames_count = 0
                    else:
                        # nothing buffered; just continue
                        pass

    except KeyboardInterrupt:
        print("Stopping VAD loop by user.")
    finally:
        stream.stop()
        stream.close()

# ----------------- Finalize chunk and push to uploader -----------------
def finalize_and_queue(frames_list):
    if not frames_list:
        return
    frames_bytes = b"".join(frames_list)
    filename = os.path.join(TEMP_DIR, f"chunk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav")
    try:
        save_wav_from_frames(frames_bytes, filename)
        print("Saved chunk:", filename)
        upload_queue.put(filename)
    except Exception as e:
        print("Failed to save/upload chunk:", e)
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except:
            pass

# ----------------- Start background workers and VAD -----------------
def main():
    # Start uploader thread
    t_up = threading.Thread(target=uploader_worker, daemon=True)
    t_up.start()
    # Start player thread
    t_play = threading.Thread(target=player_worker, daemon=True)
    t_play.start()

    # Run VAD loop (main thread)
    vad_loop()

if __name__ == "__main__":
    main()
