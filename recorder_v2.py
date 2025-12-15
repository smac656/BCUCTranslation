"""
recorder_v2.py - Non-blocking microphone recorder for BCUCTranslation.

- Captures PCM16 mono @ 16 kHz from the default input device (or a chosen device).
- Pushes raw PCM blocks into a thread-safe queue from the sounddevice callback.
- A background asyncio task batches ~1s of audio and POSTs it to the FastAPI server
  at /audio_chunk as raw bytes.

This script is intentionally simple and robust: the audio callback never blocks and
network hiccups only affect the sender loop, not recording.
"""

import argparse
import asyncio
import queue
import sys
import threading
import time
from typing import Optional

import httpx
import numpy as np
import sounddevice as sd

SERVER_CHUNK_URL = "http://localhost:8000/audio_chunk"
CHUNK_SECONDS = 0.25
CHANNELS = 1
DTYPE = "int16"

# Thread-safe queue for PCM blocks coming from the audio callback
pcm_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Callback from sounddevice.InputStream.

    Runs on a realtime audio thread. We must not block here.
    """
    if status:
        print("Audio status:", status, file=sys.stderr)

    # Make a copy so sounddevice can safely reuse its buffer
    pcm_queue.put(indata.copy())


async def sender_loop():
    """Async loop that batches PCM blocks and sends them to the server."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        buffer = []
        last_send = time.time()

        while True:
            # Pull PCM frames
            try:
                block = pcm_queue.get(timeout=0.1)
                buffer.append(block)
            except queue.Empty:
                pass

            now = time.time()
            elapsed = now - last_send

            if elapsed >= CHUNK_SECONDS and buffer:

                # Combine blocks
                arr = np.concatenate(buffer, axis=0)

                # Convert to int16 FIRST — must happen before p2p
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)

                # Compute amplitude ONCE — this value is final
                amplitude = int(np.ptp(arr))

##                # Silence gate
##                if amplitude < 600:
##                    buffer.clear()
##                    last_send = now
##                    await asyncio.sleep(0.001)
##                    continue

                # Convert to bytes and send
                bts = arr.tobytes()

                try:
                    resp = await client.post(SERVER_CHUNK_URL, content=bts)
                except Exception as e:
                    print("Send error:", repr(e))

                buffer.clear()
                last_send = now

            await asyncio.sleep(0.001)




def start_async_loop(loop: asyncio.AbstractEventLoop):
    """Run the sender_loop inside its own event loop in a background thread."""
    asyncio.set_event_loop(loop)
    loop.create_task(sender_loop())
    try:
        loop.run_forever()
    finally:
        # Best-effort cleanup
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


def main(args):
    samplerate: int = args.samplerate

    # "default" -> None for sounddevice, digits -> device index, else device name
    if args.device == "default":
        device: Optional[int | str] = None
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device

    # Background event loop for HTTP POSTs
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=start_async_loop, args=(loop,), daemon=True)
    thread.start()

    # Start microphone stream
    stream = sd.InputStream(
        samplerate=samplerate,
        device=device,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
    )

    print("Recorder started. Press Ctrl+C to stop.")
    with stream:
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping recorder...")
        finally:
            # Stop the background loop
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="default")
    parser.add_argument("--samplerate", type=int, default=16000)
    args = parser.parse_args()
    main(args)
