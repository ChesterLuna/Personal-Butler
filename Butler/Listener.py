import speech_recognition as sr


# The listener module of the Butler. Allows it to listen to what people are saying.

class Listener:

    def __init__(self):
        self.l = sr.Recognizer()
        self.mic = sr.Microphone() # Use default microphone

        # calibrate
        with self.mic as source:
            print("adjusting")
            self.l.adjust_for_ambient_noise(source)

    
    def listen_once(self, timeout = None):
        dialogue = None
        print("listening")

        with self.mic as source:
            audio = self.l.listen(source,timeout)
            dialogue = self.l.recognize_google(audio)

        return(dialogue)
    

def main():
    ls = Listener()

    print(ls.listen_once())


if __name__ == "__main__":
    main()
