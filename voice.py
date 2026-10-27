"""Speech helpers: text-to-speech for questions, speech-to-text for answers."""
import os

import streamlit as st
from openai import OpenAI

TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")


@st.cache_data(show_spinner=False, max_entries=50)
def speak(text: str) -> bytes:
    """Return MP3 bytes of the text spoken aloud."""
    response = OpenAI().audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=text)
    return response.content


def transcribe(audio) -> str:
    """Turn a recorded Streamlit audio file into text."""
    result = OpenAI().audio.transcriptions.create(
        model=STT_MODEL, file=("answer.wav", audio.getvalue(), "audio/wav")
    )
    return result.text.strip()
