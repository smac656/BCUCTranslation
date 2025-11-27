# README.md (short)

1) Copy config.example.env -> .env and add your OPENAI_API_KEY and optional settings


2) python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


3) Start server:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload


4) On the same machine (or another on the LAN) run recorder.py
python recorder.py
(recorder will capture audio from default input; you can run it on the laptop connected to the mixer output.)


5) On attendee phones open http://<server-ip>:8000/ and select language. Or print a QR code to the URL.


NOTES:
- This approach records short chunks and uploads them. Latency ~2-4s per chunk.
- After you are happy with MVP, I can help migrate to a Realtime WebSocket send approach to reduce latency.
- Tweak CHUNK_SECONDS to trade latency vs stability.