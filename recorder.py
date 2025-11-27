import os
import time
import uuid
import sounddevice as sd
import soundfile as sf
import requests
import base64
import webrtcvad
import numpy as np
from dotenv import load_dotenv

load_dotenv()
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:8000')
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '16000'))
CHANNELS = int(os.getenv('CHANNELS', '1'))
VAD_MODE = int(os.getenv('VAD_MODE', '2'))  # 0-3

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

vad = webrtcvad.Vad(VAD_MODE)

FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

def frame_generator(audio):
    for start in range(0, len(audio), FRAME_SIZE):
        yield audio[start:start + FRAME_SIZE].tobytes()

def record_and_send_vad():
    print(f"Starting VAD recorder...\nListening at {SAMPLE_RATE} Hz, mode={VAD_MODE}")
    try:
        while True:
            recording = sd.rec(FRAME_SIZE * 100, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
            sd.wait()
            audio = np.array(recording).flatten()
            frames = list(frame_generator(audio))
            speech_frames = [f for f in frames if vad.is_speech(f, SAMPLE_RATE)]
            if speech_frames:
                filename = os.path.join(TEMP_DIR, f"chunk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav")
                sf.write(filename, audio, SAMPLE_RATE)
                print(f"Detected speech, sending chunk: {filename}")
                resp = upload_chunk(filename)
                if resp:
                    print("English:", resp.get('english', ''))
                    for lang, text in resp.get('translations', {}).items():
                        print(f"{lang}: {text}")
                    if 'tts' in resp:
                        play_tts(resp['tts'])
                try:
                    os.remove(filename)
                except:
                    pass
    except KeyboardInterrupt:
        print("Recorder stopped.")

def upload_chunk(filename):
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

if __name__ == '__main__':
    record_and_send_vad()
