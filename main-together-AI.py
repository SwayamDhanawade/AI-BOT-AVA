import subprocess
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write
import platform
import yt_dlp
import vlc
import time
import torch
import os
import sys
from dotenv import load_dotenv
from together import Together

# Load environment variables
load_dotenv()

# Fetch API key from environment variables
api_key = os.getenv("TOGETHER_AI_KEY")
if not api_key:
    print("Error: API key not found. Please set TOGETHER_AI_KEY in .env file.")
    sys.exit(1)

# Initialize Together client with the API key
client = Together(api_key=api_key)

# Initialize the pyttsx3 TTS engine
engine = pyttsx3.init()

# Initialize VLC Player
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()

# Track if music is playing
music_playing = False

def is_gpu_available():
    return torch.cuda.is_available()

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to record audio using sounddevice
def record_audio(duration=5, sample_rate=16000):
    try:
        print("Listening...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        return np.squeeze(audio_data), sample_rate
    except Exception as e:
        print(f"Error recording audio: {e}")
        speak_text("There was an issue recording your voice.")
        return None, None

# Function to convert audio data to WAV format
def audio_to_wav(audio_data, sample_rate):
    if audio_data is None or sample_rate is None:
        return None
    audio_io = io.BytesIO()
    write(audio_io, sample_rate, np.array(audio_data, dtype=np.int16))
    audio_io.seek(0)
    return audio_io

# Function to capture voice input
def get_voice_input():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    audio_data, sample_rate = record_audio(duration=10)
    
    if audio_data is None or sample_rate is None:
        return None
    
    audio_io = audio_to_wav(audio_data, sample_rate)
    if audio_io is None:
        return None
    
    with sr.AudioFile(audio_io) as source:
        audio = recognizer.record(source)

    try:
        print("Recognizing command...")
        command = recognizer.recognize_google(audio)
        print(f"User said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak_text("Sorry, I couldn't understand your voice.")
        return None
    except sr.RequestError:
        speak_text("Sorry, I couldn't reach the speech recognition service.")
        return None

# Function to fetch response using Together AI
def fetch_together_ai_response(query):
    try:
        print(f"Fetching response for query: {query}")

        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": "Sure! Let me help you with that."}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error fetching response: {e}")
        return "An unexpected error occurred while fetching the response."

# Object detection process management
object_detection_process = None
ocr_process = None

# Function to start object detection with GPU support
def start_object_detection():
    global object_detection_process
    script_path = "object_detection.py"

    if object_detection_process is None:
        try:
            cmd = ['python', script_path]
            if is_gpu_available():
                cmd.append("--use-gpu")
            if platform.system() == 'Windows':
                object_detection_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                object_detection_process = subprocess.Popen(cmd)
            speak_text("Object detection started.")
        except Exception as e:
            print(f"Error starting object detection: {e}")
            speak_text("Error starting object detection.")
    else:
        speak_text("Object detection is already running.")

# Function to stop object detection
def stop_object_detection():
    global object_detection_process
    if object_detection_process is not None:
        try:
            object_detection_process.terminate()
            object_detection_process.wait()
            object_detection_process = None
            speak_text("Object detection stopped.")
        except Exception as e:
            print(f"Error stopping object detection: {e}")
            speak_text("Error stopping object detection.")
    else:
        speak_text("Object detection is not running.")

def start_ocr():
    global ocr_process
    script_path = "ocr_script.py"
    if ocr_process is None:
        try:
            cmd = ['python', script_path]
            if is_gpu_available():
                cmd.append("--use-gpu")
            if platform.system() == 'Windows':
                ocr_process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                ocr_process = subprocess.Popen(cmd)
            speak_text("OCR started.")
        except Exception as e:
            print(f"Error starting OCR: {e}")
            speak_text("Error starting OCR.")
    else:
        speak_text("OCR is already running.")

def stop_ocr():
    global ocr_process
    if ocr_process is not None:
        try:
            ocr_process.terminate()
            ocr_process.wait()
            ocr_process = None
            speak_text("OCR stopped.")
        except Exception as e:
            print(f"Error stopping OCR: {e}")
            speak_text("Error stopping OCR.")
    else:
        speak_text("OCR is not running.")

# Function to terminate the entire program
def terminate_program():
    if object_detection_process is not None:
        stop_object_detection()
    if music_playing is not False:
        stop_music()
    speak_text("Terminating the program. Goodbye!")
    sys.exit()

# Music control functions
def play_music(song_name):
    global music_playing
    try:
        speak_text(f"Searching for {song_name} on YouTube.")
        ydl_opts = {"format": "bestaudio/best", "quiet": True, "noplaylist": True, "default_search": "ytsearch1:"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(song_name, download=False)
            if result is None or ('entries' in result and not result["entries"]):
                speak_text("Could not find a playable song.")
                return
            video = result["entries"][0] if "entries" in result else result
            url = video.get("url")
            if not url:
                speak_text("Could not find a valid URL for the song.")
                return
            speak_text(f"Playing {video['title']}.")
            media = vlc_instance.media_new(url)
            player.set_media(media)
            player.play()
            music_playing = True
            print(f"Now Playing: {video['title']}")
    except Exception as e:
        print(f"Error playing music: {e}")
        speak_text("Sorry, I couldn't fetch the song.")

def stop_music():
    global music_playing
    player.stop()
    music_playing = False
    speak_text("Music stopped.")

def handle_chat(query):
    if "start object detection" in query:
        start_object_detection()
        return
    elif "stop object detection" in query:
        stop_object_detection()
        return
    elif "top object detection" in query:
        stop_object_detection()
        return
    if "start ocr" in query:
        start_ocr()
        return
    elif "stop ocr" in query:
        stop_ocr()
        return
    elif "top ocr" in query:
        stop_ocr()
        return    
    elif any(exit_word in query for exit_word in ["terminate", "exit", "bye"]):
        terminate_program()
        return
    elif "play" in query and "music" in query:
        song_name = query.replace("play music", "").strip()  # Extract song name if provided
        if song_name:
            play_music(song_name)
        else:
            speak_text("Please specify a song name.")
        return
    elif "stop music" in query:
        stop_music()
        return
    elif "top music" in query:
        stop_music()
        return
    else:
        answer = fetch_together_ai_response(query)
        print(answer)
        speak_text(answer)

def main():
    global music_playing
    while True:
        user_input = input("Type 'Hey Ava' to start: ").strip().lower()
        if user_input != "hey ava":
            print ("Please Enter Valid Input")
        elif user_input == "hey ava":
            break
    while True:
        if music_playing:
            user_input = input("Music is playing. Type 'Hey Ava' to reactivate listening: ").strip().lower()
            if user_input == "hey ava":
                music_playing = False
        else:
            speak_text("Listening. Say your command.")
            command = get_voice_input()
            if command:
                handle_chat(command)

if __name__ == "__main__":
    main()
