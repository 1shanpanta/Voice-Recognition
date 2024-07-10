import pyaudio
import wave
import time
import os
from groq import Groq

from dotenv import load_dotenv
import spacy 


load_dotenv()

nlp = spacy.load("en_core_web_sm")


def record_audio(filename, duration=5, sample_rate=44100, channels=2, chunk=1024):
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

def transcribe_audio(audio_file):
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    model = 'whisper-large-v3'

    # Load the audio file
    with open(audio_file, "rb") as audio:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio
        )

    return transcription.text

if __name__ == "__main__":
    audio_file = record_audio(f"recording_{int(time.time())}.wav", duration=5)
    
    transcription = transcribe_audio(audio_file)
    
    print("Transcription:")
    print(transcription)