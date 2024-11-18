import cv2
import os 
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt

# The code that makes the Butler recognize the users faces and allows it to greet them and respond to their comands.
# By Chester A. Perez Luna
# 
# Used code snippets from:
# 
# https://www.datacamp.com/tutorial/face-detection-python-opencv
# https://medium.com/@siucy814/training-a-facial-recognition-model-by-opencv-e4717d86b7ec


# Makes a classifier that recognizes faces
CC_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set the path to get the users pictures
current_path = os.getcwd()
path = current_path + "\\recognized_faces"

# Get the users names by the name of their respective folder
labels = os.listdir(path)
name = {}

# How much confidence does the Butler need to know before recognizing a face.
# The confidence is a measure of how different the face is to the prediction.
# The smallest the number, the more confident it needs to be of the face.
confidence_level = 50

def getImagesAndLabels(path):
    faceSamples=[]
    ids = []
    
    folders = labels

    n = 0
    for folder in folders:
        n += 1
        folderPath = os.path.join(path, folder)
        name[n] = folder

        imagePaths = [os.path.join(folderPath,f) for f in os.listdir(folderPath)]
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') #Luminance  ==> greystyle
            img_numpy = np.array(PIL_img,'uint8')
            #print(PIL_img)
            #.show()
            #print(len(img_numpy)
            id = n
            faces = CC_classifier.detectMultiScale(img_numpy)
            #print(id)
            # print(faces)
            if len(faces) > 1:
                print("The following image detect more than 1 face", imagePath)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
                #print(ids)
            
    return faceSamples,ids

faces,ids = getImagesAndLabels(path)


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = CC_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    return faces, gray_image


def train_classifier(faces, faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(ids))

# Save the model
facerecognizer = train_classifier(faces, ids)
# facerecognizer.save(path + '\\mytrainer.yml')


def main():

    # Get the video
    video_stream = cv2.VideoCapture(0)

    while True:

        result, video_frame = video_stream.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        faces, gray_image = detect_bounding_box(video_frame)

        for face in faces:
            (x,y,w,h) = face
            # print(x,y,w,h)
            roi_gray = gray_image[y:y+h, x:x+h]
            label, confidence = face_recognizer.predict(roi_gray)
            
            # print("confidence:", confidence)
            # print("label:", label)
            
            predicted_name = name[label]
            # gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)

            setBoundingBox(face, video_frame, predicted_name, confidence)
        cv2.imshow("A Butler should know your face", video_frame) 

        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

    video_stream.release()
    cv2.destroyAllWindows()

def setBoundingBox(face, vid, label: str, confidence: float):
    (x,y,w,h) = face
    point1 = (x, y)
    point2 = (x + w, y + h)
    color = (0, 255, 0)
    thickness = 4

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4

    text = label + " " + str(confidence)

    # Add bounding box around face
    cv2.rectangle(vid, point1, point2, color, thickness)

    # Add label to face
    if(confidence <= confidence_level):
        cv2.putText(vid, text, (x, y - 5), font_face, scale, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(vid, "Unknown user", (x, y - 5), font_face, scale, color, 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()


