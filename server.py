import os
import asyncio
import json
import base64
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
import httpx
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGUAGES = [s.strip() for s in os.getenv('LANGUAGES', 'zh,es').split(',') if s.strip()]

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

# --- WebSocket manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(data)
            except Exception:
                try:
                    connection.close()
                except Exception:
                    pass

manager = ConnectionManager()

# --- Bible dictionary ---
BIBLE_DICT_PATH = 'bible_dict.json'
if os.path.exists(BIBLE_DICT_PATH):
    with open(BIBLE_DICT_PATH, 'r', encoding='utf-8') as f:
        BIBLE_DICT = json.load(f)
else:
    BIBLE_DICT = {}

def apply_bible_dictionary(text: str) -> str:
    for word, phonetic in BIBLE_DICT.items():
        text = text.replace(word, phonetic)
    return text

# --- Routes ---
@app.get('/')
async def root():
    return {"status": "Server running"}

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post('/transcribe')
async def transcribe_chunk(file: UploadFile = File(...)):
    tmp_dir = "/temp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, 'wb') as f:
        f.write(await file.read())

    # Transcribe
    transcribed_text = await call_openai_transcription(tmp_path)

    # Translate
    translations = {}
    for lang in LANGUAGES:
        translations[lang] = await call_openai_translate(transcribed_text, lang)

    # Generate TTS
    tts_audio_base64 = {}
    for lang, text in translations.items():
        text_for_tts = apply_bible_dictionary(text)
        tts_audio_base64[lang] = await call_openai_tts(text_for_tts)

    # Broadcast
    packet = {
        'type': 'translation_packet',
        'english': transcribed_text,
        'translations': translations,
        'tts': tts_audio_base64
    }
    await manager.broadcast(packet)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return {'status': 'ok', 'english': transcribed_text, 'translations': translations}

# --- OpenAI API calls ---
async def call_openai_transcription(audio_filepath: str) -> str:
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {
            'file': (os.path.basename(audio_filepath), open(audio_filepath, 'rb'), 'audio/wav'),
            'model': (None, 'gpt-4o-mini-transcribe')
        }
        resp = await client.post('https://api.openai.com/v1/audio/transcriptions', files=files, headers=headers)
        resp.raise_for_status()
        return resp.json().get('text', '')

async def call_openai_translate(text: str, target_language: str) -> str:
    system_prompt = (
        "You are a concise translator. Translate the provided English sermon speech "
        "into the target language. Keep it short, literal, suitable for live captions."
    )
    user_prompt = f"Translate the following into {target_language} concisely:\n\n{text}"
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'temperature': 0.0,
        'max_tokens': 400
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post('https://api.openai.com/v1/chat/completions', json=payload, headers=headers)
        resp.raise_for_status()
        choices = resp.json().get('choices', [])
        if choices:
            return choices[0].get('message', {}).get('content', '').strip()
        return ''

async def call_openai_tts(text: str) -> str:
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': 'gpt-4o-mini-tts', 'voice': 'alloy', 'input': text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post('https://api.openai.com/v1/audio/speech', json=payload, headers=headers)
        resp.raise_for_status()
        audio_bytes = resp.content
        return base64.b64encode(audio_bytes).decode('utf-8')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
