from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
from llama_cpp import Llama
from TTS.api import TTS
import sqlite3
import sounddevice as sd
import numpy as np
import os

app = FastAPI()

# --- Voice Interface ---
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

# --- Facial Recognition ---
@app.post("/register_face")
async def register_face(file: UploadFile = File(...)):
    with open("face.jpg", "wb") as buffer:
        buffer.write(await file.read())
    embedding = DeepFace.represent("face.jpg", model_name="Facenet")[0]["embedding"]
    return {"message": "Face registered!", "embedding": embedding}

@app.post("/authenticate")
async def authenticate(file: UploadFile = File(...)):
    with open("current_face.jpg", "wb") as buffer:
        buffer.write(await file.read())
    result = DeepFace.verify("current_face.jpg", "face.jpg")
    return {"verified": result["verified"]}

# --- Creole Learning System ---
@app.post("/add_phrase")
async def add_phrase(creole: str, english: str):
    conn = sqlite3.connect('creole.db')
    conn.execute("INSERT INTO phrases VALUES (?, ?)", (creole, english))
    conn.commit()
    return {"message": "Phrase saved!"}

@app.get("/translate")
async def translate(creole: str):
    conn = sqlite3.connect('creole.db')
    cursor = conn.execute("SELECT english FROM phrases WHERE creole=?", (creole,))
    result = cursor.fetchone()
    return {"translation": result[0] if result else "I'm still learning Creole. Please correct me!"}

# --- Local AI (Mistral 7B) ---
@app.get("/ask")
async def ask(question: str):
    llm = Llama(model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    response = llm(question, max_tokens=100)
    return {"answer": response["choices"][0]["text"]}

# --- Main Loop ---
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
