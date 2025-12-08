"""
recorder_v2.py
- Proper non-blocking recorder for server_v4.
- Uses a background asyncio loop for HTTP POSTs.
- Sounddevice callback never blocks.
"""
print("🔥 RUNNING NEW RECORDER_V2.PY VERSION")


import argparse
import threading
import asyncio
import queue
import sys
import time
import base64
import sounddevice as sd
import httpx
import numpy as np

SERVER_CHUNK_URL = "http://localhost:8000/audio_chunk"
CHUNK_SECONDS = 1.0
CHANNELS = 1
DTYPE = 'int16'

# Thread-safe queue for PCM blocks
pcm_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status, file=sys.stderr)
    # push a copy of the PCM block
    pcm_queue.put(indata.copy())

# ------------- Async Sender Loop -------------
async def sender_loop():
    async with httpx.AsyncClient(timeout=20) as client:
        buffer = []
        last_send = time.time()

        while True:
            # Pull from thread-safe queue (non-async)
            try:
                block = pcm_queue.get(timeout=0.1)
                buffer.append(block)
            except Exception:
                pass

            elapsed = time.time() - last_send
            if elapsed >= CHUNK_SECONDS and buffer:
                arr = np.concatenate(buffer, axis=0)
                bts = arr.tobytes()

                try:
#                    await client.post(SERVER_CHUNK_URL, content=bts)

                    print("🚀 Sending chunk:", len(bts))
                    resp = await client.post(SERVER_CHUNK_URL, content=bts)
                    print("✔️ Server replied:", resp.status_code, resp.text)

                except Exception as e:
                    print("Send error:", e)

                buffer = []
                last_send = time.time()

            await asyncio.sleep(0.001)

# ------------- Start Async Loop in Background Thread -------------
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sender_loop())

# ------------- Main Entrypoint -------------
def main(args):
    samplerate = args.samplerate
    device = None if args.device == 'default' else (
        int(args.device) if args.device.isdigit() else args.device
    )

    # Create background event loop for async tasks
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=start_async_loop, args=(loop,), daemon=True)
    thread.start()

    # Start microphone
    stream = sd.InputStream(samplerate=samplerate, device=device,
                            channels=CHANNELS, dtype=DTYPE,
                            callback=audio_callback)

    with stream:
        print("Recorder started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping recorder...")
            loop.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="default")
    parser.add_argument("--samplerate", type=int, default=16000)
    args = parser.parse_args()
    main(args)
