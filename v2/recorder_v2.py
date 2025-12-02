"""recorder_v2.py
- Simple recorder that captures audio with sounddevice and POSTs chunks to server_v2 /audio_chunk
- Uses a buffering approach: accumulates frames for CHUNK_SECONDS and sends as raw bytes
- Accepts CLI args: --device (device index or name), --samplerate

Note: replace the lightweight chunking with your webrtcvad logic if desired.
"""

import argparse
import queue
import sys
import time
import sounddevice as sd
import httpx
import base64

SERVER_CHUNK_URL = "http://localhost:8000/audio_chunk"
CHUNK_SECONDS = 0.6
CHANNELS = 1
DTYPE = 'int16'

q = queue.Queue()

def int16_to_bytes(frames):
    return frames.tobytes()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status, file=sys.stderr)
    q.put(indata.copy())

async def send_bytes(bts: bytes):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            # send raw bytes
            resp = await client.post(SERVER_CHUNK_URL, content=bts)
            # print(resp.text)
        except Exception as e:
            print("Failed to send chunk:", e)


def main(args):
    samplerate = args.samplerate
    device = None if args.device == 'default' else int(args.device) if args.device.isdigit() else args.device

    stream = sd.InputStream(samplerate=samplerate, device=device, channels=CHANNELS, dtype=DTYPE, callback=audio_callback)
    with stream:
        print("Recorder started. Press Ctrl+C to stop.")
        buffer = []
        last_send = time.time()
        try:
            while True:
                try:
                    block = q.get(timeout=1)
                    buffer.append(block)
                except Exception:
                    pass

                # send when buffer reaches CHUNK_SECONDS
                elapsed = time.time() - last_send
                if elapsed >= CHUNK_SECONDS and buffer:
                    import numpy as np
                    arr = np.concatenate(buffer, axis=0)
                    bts = int16_to_bytes(arr)
                    # send asynchronously
                    import asyncio
                    asyncio.run(send_bytes(bts))
                    buffer = []
                    last_send = time.time()

        except KeyboardInterrupt:
            print("Recorder stopped by user")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='default', help='Device index or name')
    parser.add_argument('--samplerate', default=16000, type=int)
    args = parser.parse_args()
    main(args)
