import Recognizer as rn
import Speech

def main():
    # Receive information from Recognizer
    # If a face is found, label it and return the name
    # Use the Speech to greet the person using the label
    peopleGreeted = []
    faceFiles = None
    speaker = Speech.Speaker()
    recognizer = rn.Recognizer()
    while faceFiles is not False:
        facesFound = recognizer.recognize_faces()
        faceFiles, name, data = facesFound
        
        if(faceFiles is False):
            break

        if(len(faceFiles) != 0):
            if(name[0] not in peopleGreeted):
                speaker.greet(name[0])
                peopleGreeted.append(name[0])
    recognizer.stop_recognizer



if __name__ == "__main__":
    main()
