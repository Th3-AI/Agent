import typer
import asyncio
from agent_main import process_user_input, initialize_session
import speech_recognition as sr
import pyttsx3
import threading
import time

app = typer.Typer(help="Multipurpose Agent CLI: Web search, scraping, and more. Use text or voice mode.")

def init_tts():
    """Initialize text-to-speech engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)  # Speed of speech
    return engine

def speak_text(engine, text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen_for_voice():
    """Listen for voice input and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

@app.command()
def text():
    """Run the agent in text mode."""
    print("Starting agent in text mode. Type 'exit' to quit.")
    asyncio.run(initialize_session())
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            break
            
        response = asyncio.run(process_user_input(user_input))
        print(f"\nAgent: {response}")

@app.command()
def voice():
    """Run the agent in voice mode."""
    print("Starting agent in voice mode. Say 'exit' to quit.")
    asyncio.run(initialize_session())
    
    engine = init_tts()
    speak_text(engine, "Voice mode activated. How can I help you?")
    
    while True:
        user_input = listen_for_voice()
        if not user_input:
            continue
            
        if user_input.lower() == 'exit':
            speak_text(engine, "Goodbye!")
            break
            
        response = asyncio.run(process_user_input(user_input, is_voice_mode=True))
        
        # Handle tuple response (display_text, speak_text)
        if isinstance(response, tuple) and len(response) == 2:
            display_text, speech_text = response
            print(f"\nAgent: {display_text}")
            speak_text(engine, speech_text)
        else:
            # Regular single response
            print(f"\nAgent: {response}")
            speak_text(engine, response)

@app.command()
def search(query: str):
    """Quick web search (text mode only)."""
    typer.echo(f"Searching for: {query}")
    # Use the perform_web_search tool directly
    from google.adk.tools import perform_web_search
    result = perform_web_search(query)
    typer.echo(result.get("result") or result.get("error"))

@app.command()
def scrape(url: str):
    """Quickly scrape a URL (text mode only)."""
    typer.echo(f"Scraping: {url}")
    from google.adk.tools import perform_url_scrape
    result = perform_url_scrape(url)
    typer.echo(result.get("text_content") or result.get("error"))

if __name__ == "__main__":
    app() 