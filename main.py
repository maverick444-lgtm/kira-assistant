import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
from llama_cpp import Llama
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import asyncpg
import asyncio
from urllib.parse import urlparse

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=500, detail="Database URL not found in environment variables.")
    parsed_url = urlparse(database_url)
    user = parsed_url.username
    password = parsed_url.password
    host = parsed_url.hostname
    port = parsed_url.port
    database = parsed_url.path[1:]
    conn = await asyncpg.connect(user=user, password=password, database=database, host=host, port=port)
    return conn

class VoiceEngine:
    def __init__(self):
        self.sample_rate = 16000
        self.model = Model("vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)

    def record_audio(self, duration=5):
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        return audio.flatten().astype(np.int16)

    def listen(self):
        audio = self.record_audio()
        if self.recognizer.AcceptWaveform(audio.tobytes()):
            return self.recognizer.Result()
        return self.recognizer.PartialResult()

    def speak(self, text):
        self.tts.tts_to_file(text=text, file_path="response.wav")
        return "response.wav"

@app.post("/register_face")
async def register_face(file: UploadFile = File(...)):
    try:
        file_location = "face.jpg"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        embedding = DeepFace.represent(file_location, model_name="Facenet")[0]["embedding"]
        logger.info("Face registered successfully.")
        return {"message": "Face registered!", "embedding": embedding}
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail="Error registering face.")

@app.post("/authenticate")
async def authenticate(file: UploadFile = File(...)):
    try:
        file_location = "current_face.jpg"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())
        result = DeepFace.verify(file_location, "face.jpg")
        return {"verified": result["verified"]}
    except Exception as e:
        logger.error(f"Error in authentication: {e}")
        raise HTTPException(status_code=500, detail="Error in authentication.")

@app.post("/add_phrase")
async def add_phrase(creole: str, english: str):
    try:
        conn = await get_db_connection()
        await conn.execute("INSERT INTO phrases (creole, english) VALUES ($1, $2)", creole, english)
        await conn.close()
        logger.info(f"Phrase added: {creole} -> {english}")
        return {"message": "Phrase saved!"}
    except Exception as e:
        logger.error(f"Error adding phrase: {e}")
        raise HTTPException(status_code=500, detail="Error saving phrase.")

@app.get("/translate")
async def translate(creole: str):
    try:
        conn = await get_db_connection()
        result = await conn.fetchrow("SELECT english FROM phrases WHERE creole=$1", creole)
        await conn.close()
        if result:
            return {"translation": result['english']}
        else:
            return {"translation": "I'm still learning Creole. Please correct me!"}
    except Exception as e:
        logger.error(f"Error in translation: {e}")
        raise HTTPException(status_code=500, detail="Error in translation.")

@app.get("/ask")
async def ask(question: str):
    try:
        llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        response = llm(question, max_tokens=100)
        return {"answer": response["choices"][0]["text"]}
    except Exception as e:
        logger.error(f"Error in AI response: {e}")
        raise HTTPException(status_code=500, detail="Error generating response.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
