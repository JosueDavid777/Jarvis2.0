"""
Asistente de voz con Gemini (Google AI)
Compatible con Python 3.14
Usa gTTS + playsound3 para leer las respuestas
Palabra de activación: "asistente"
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
from playsound3 import playsound
import speech_recognition as sr
import google.generativeai as genai
import edge_tts
import asyncio
from playsound3 import playsound
import os

VOICE = "es-MX-DaliaNeural"
 

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
WAKE_SECONDS = 2
MAX_RECORD_SECONDS = 6
WAKE_WORD = "jarvis"
MODEL_NAME = "models/gemini-2.0-flash"

async def speak_async(text):
    print("Asistente:", text)
    tts = edge_tts.Communicate(text, voice="es-ES-AlvaroNeural")
    await tts.save("voz.mp3")
    playsound("voz.mp3")
    os.remove("voz.mp3")

def speak(text):
    asyncio.run(speak_async(text))
# Clave Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ ERROR: No encontré GEMINI_API_KEY. Configura tu clave primero.")
    sys.exit(1)

# Configurar Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

# ---------------- FUNCIONES ---------------

def normalize_audio(audio):
    """Normaliza audio a int16."""
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return (audio * 32767).astype(np.int16)

def save_wav(path, pcm16, samplerate):
    """Guarda audio en WAV."""
    sf.write(path, pcm16, samplerate, subtype='PCM_16')

def transcribe_google(wav_path: str) -> str:
    """Transcribe audio usando SpeechRecognition (Google)."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="es-ES")
        return text.lower()
    except sr.UnknownValueError:
        print(" No entendí lo que dijiste.")
        return ""
    except sr.RequestError as e:
        print(f"⚠️ Error en SpeechRecognition: {e}")
        return ""

def record(seconds: int) -> np.ndarray:
    """Graba audio con sounddevice."""
    rec = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    return rec

def listen_for_wake():
    print("🎧 Escuchando activación (di 'jarvis')...")
    rec = record(WAKE_SECONDS)
    if np.max(np.abs(rec)) > 28000:
       rec = normalize_audio(rec)
    wav_path = "wake_word.wav"
    save_wav(wav_path, rec, SAMPLE_RATE)
    text = transcribe_google(wav_path)
    return text

def get_gemini_response(prompt: str) -> str:
    """Envía texto a Gemini y devuelve la respuesta."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("⚠️ Error al contactar con Gemini:", e)
        return "Hubo un problema al contactar con Gemini."


def menu():
    print("🎙️ Habla ahora...")
    rec = record(MAX_RECORD_SECONDS)
    if np.max(np.abs(rec)) > 28000:
       rec = normalize_audio(rec)
    wav_path = "command.wav"
    save_wav(wav_path, rec, SAMPLE_RATE)
    text = transcribe_google(wav_path)
    print(f"📝 Texto reconocido: {text}")
    if text == "pregunta":
        listen_and_respond()

def listen_and_respond():
    """Graba el comando, lo transcribe y responde con IA."""
    print("🎙️ Habla ahora...")
    rec = record(MAX_RECORD_SECONDS)
    rec = normalize_audio(rec)
    wav_path = "command.wav"
    save_wav(wav_path, rec, SAMPLE_RATE)
    text = transcribe_google(wav_path)
    response = get_gemini_response(text)
    print(text)
    speak(response)
    if text == "adiós":
        main()
    else:
        listen_and_respond()


# ---------------- MAIN LOOP ----------------
def main():
    while True:
        wake_text = listen_for_wake()
        if wake_text and WAKE_WORD in wake_text:
            speak("Te escucho.")
            menu()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
