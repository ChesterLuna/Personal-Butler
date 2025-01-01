import speech_recognition as sr


# The listener module of the Butler. Allows it to listen to what people are saying.

class Listener:

    def __init__(self):
        l = sr.Recognizer()
        mic = sr.Microphone() # Use default microphone

        # calibrate
        with mic as source:
            l.adjust_for_ambient_noise(source)

def main():
    ls = Listener()

if __name__ == "__main__":
    main()
