import Recognizer as rn
import Speech

def main():
    # Receive information from Recognizer
    # If a face is found, label it and return the name
    # Use the Speech to greet the person using the label
    faceFound = None
    speaker = Speech.Speaker()
    while faceFound is not False:
        faceFound = rn.recognizeFace()
        faceFile, name, data = faceFound

        speaker.greet(name)



if __name__ == "__main__":
    main()
