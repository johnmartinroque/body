import speech_recognition as sr
import pyttsx3

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

tts_engine.setProperty('rate', 160)

with sr.Microphone() as source:
    print("Please speak something...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        # Speak the recognized text
        tts_engine.say(f"You said: {text}")
        tts_engine.runAndWait()

    except sr.UnknownValueError:
        print("Sorry, could not understand your speech.")
        tts_engine.say("Sorry, I could not understand what you said.")
        tts_engine.runAndWait()
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        tts_engine.say("Sorry, I could not reach the speech service.")
        tts_engine.runAndWait()
