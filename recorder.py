"""VAD-enabled recorder with real-time TTS playback"""

import os
import time
import uuid
import queue
import sounddevice as sd
import soundfile as sf
import requests
import base64
import webrtcvad

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:8000')
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '16000'))
CHANNELS = int(os.getenv('CHANNELS', '1'))
FRAME_DURATION_MS = 30  # 10, 20, or 30 ms frames for VAD
VAD_MODE = 2  # 0-3, higher is more aggressive
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

vad = webrtcvad.Vad(VAD_MODE)

def record_callback(indata, frames, time_info, status):
    """Callback from sounddevice.InputStream, pushes raw audio into queue"""
    if status:
        print("Status:", status)
    audio_queue.put(bytes(indata))

def frames_from_queue():
    """Yield fixed-size frames from the audio queue"""
    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * CHANNELS * 2  # 16-bit PCM
    buffer = b""
    while True:
        buffer += audio_queue.get()
        while len(buffer) >= frame_size:
            yield buffer[:frame_size]
            buffer = buffer[frame_size:]

def upload_chunk(filename):
    """Upload WAV chunk to server and return JSON response"""
    with open(filename, 'rb') as f:
        files = {'file': f}
        try:
            resp = requests.post(f"{SERVER_URL}/transcribe", files=files, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print("Upload failed:", e)
            return None

def play_tts(tts_dict):
    """Decode base64 TTS audio and play via sounddevice"""
    for lang, b64_audio in tts_dict.items():
        try:
            audio_bytes = base64.b64decode(b64_audio)
            tmp_file = os.path.join(TEMP_DIR, f"tts_{lang}_{uuid.uuid4().hex[:6]}.wav")
            with open(tmp_file, 'wb') as f:
                f.write(audio_bytes)

            data, sr = sf.read(tmp_file, dtype='int16')
            print(f"Playing TTS for {lang}")
            sd.play(data, sr)
            sd.wait()
            os.remove(tmp_file)
        except Exception as e:
            print(f"Failed to play TTS for {lang}:", e)

def main_loop():
    print("Starting VAD recorder...")
    print(f"Listening at {SAMPLE_RATE} Hz, mode={VAD_MODE}")
    print("Microphone active. Speak anytime...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
                        blocksize=int(SAMPLE_RATE * FRAME_DURATION_MS / 1000),
                        callback=record_callback):
        speech_buffer = bytearray()
        for frame in frames_from_queue():
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            # Debug
            print("VAD:", is_speech)

            if is_speech:
                speech_buffer.extend(frame)
            elif speech_buffer:
                # End of speech detected
                filename = os.path.join(TEMP_DIR, f"chunk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav")
                sf.write(filename, 
                         data=bytes_to_numpy(speech_buffer),
                         samplerate=SAMPLE_RATE)
                print("Speech chunk recorded ->", filename)
                
                data = upload_chunk(filename)
                if data:
                    print("English:", data.get('english', ''))
                    for lang, text in data.get('translations', {}).items():
                        print(f"{lang}: {text}")
                    if 'tts' in data:
                        play_tts(data['tts'])

                speech_buffer.clear()
                try:
                    os.remove(filename)
                except Exception:
                    pass

def bytes_to_numpy(raw_bytes):
    import numpy as np
    return np.frombuffer(raw_bytes, dtype='int16').reshape(-1, CHANNELS)

audio_queue = queue.Queue()
if __name__ == "__main__":
    main_loop()
