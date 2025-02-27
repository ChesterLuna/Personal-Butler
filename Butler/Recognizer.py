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

class Recognizer:
    def __init__(self, confidence_level = 50, training_CL = 10):
        # Makes a classifier that recognizes faces
        self.CC_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Set the path to get the users pictures
        self.current_path = os.getcwd()
        self.path = self.current_path + "\\recognized_faces"

        # Get the users names by the name of their respective folder
        self.labels = os.listdir(self.path)
        self.name = {}

        # How much confidence does the Butler need to know before recognizing a face.
        # The confidence is a measure of how different the face is to the prediction.
        # The smallest the number, the more confident it needs to be of the face.
        self.confidence_level = confidence_level
        self.training_CL = training_CL

        self.faces, self.ids = self.get_images_and_labels(self.path)
        # self.face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        # self.face_recognizer.train(self.faces, np.array(self.ids))

        # Save the model
        self.facerecognizer = self.train_classifier(self.faces, self.ids)
        # self.facerecognizer.save(path + '\\mytrainer.yml')
        
        # Get the video
        self.video_stream = cv2.VideoCapture(0)

    # Detects the cut faces of the pictures given and their labels.
    # Detects at most one face per image, if there are multiple it chooses the one with the max confidence value.
    # Returns the faces and their labels for training.
    def get_images_and_labels(self, path):
        faceSamples=[]
        ids = []
        
        folders = self.labels

        n = 0
        for folder in folders:
            n += 1
            folderPath = os.path.join(path, folder)
            self.name[n] = folder

            imagePaths = [os.path.join(folderPath,f) for f in os.listdir(folderPath)]
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L') #Luminance  ==> greystyle
                img_numpy = np.array(PIL_img,'uint8')

                id = n
                faces, confidences = self.CC_classifier.detectMultiScale2(img_numpy)
                #print(id)
                # print(faces)
                if len(faces) > 1:
                    print("The following image detect more than 1 face", imagePath)
                if len(faces) > 0:
                    ind_max = np.argmax(confidences)

                    face = faces[ind_max]
                    confidence = confidences[ind_max]

                    if(confidence >= self.training_CL):
                        (x,y,w,h) = face
                        # plt.imshow(img_numpy[y:y+h,x:x+w], interpolation='nearest')
                        # plt.annotate(confidence,(0,0))
                        # plt.show()
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id)
                        #print(ids)
                
        return faceSamples,ids

    def main(self):

        while True:

            self.recognize_faces() 


        self.stop_recognizer()

    def stop_recognizer(self):
        self.video_stream.release()
        cv2.destroyAllWindows()

    def recognize_faces(self):
        video_frame, faces, gray_image = self.detect_faces()

        detected_names = self.show_bounded_faces(video_frame, faces, gray_image)

        if self.check_key('e'):
            return False, None, None
        return detected_names, detected_names, None

    def detect_faces(self):
        result, video_frame = self.read_video_frame()  # read frames from the video

        faces, gray_image = self.detect_bounding_box(video_frame)
        return video_frame,faces,gray_image

    def show_bounded_faces(self, video_frame, faces, gray_image) -> list:
        detected_names = self.set_bounded_faces(video_frame, faces, gray_image)
        cv2.imshow("A Butler should know your face", video_frame)
        return detected_names

    # Returns the frame from the camera
    def read_video_frame(self):
        result, video_frame = self.video_stream.read()  # read frames from the video
        if result is False:
            return False, None  # Return false if the frame is not read successfully
        return result, video_frame

    # Detects the faces from the camera.
    # Returns the (x,y,w,h) of each face and the image from the camera
    def detect_bounding_box(self,vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

        faces = self.CC_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        return faces, gray_image

    def train_classifier(self, faces, faceID):
        face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(faceID))
        return face_recognizer

    def set_bounded_faces(self, video_frame, faces, gray_image) -> list:
        detected_names = []

        for face in faces:
            label, confidence = self.get_label_confidence(face, gray_image)    
            # print("confidence:", confidence)
            # print("label:", label)
            
            predicted_name = self.name[label]
            # gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)

            if(confidence <= self.confidence_level):
                self.set_bounding_box(video_frame, face, predicted_name, confidence)
                detected_names.append(predicted_name)
            else:
                self.set_default_bounding_box(video_frame, face, predicted_name, confidence)
        return detected_names

    # From the face given, it tries to recognize a face from its training.
    # Returns the name of the user and the confidence that it was predicted correctly
    # The smaller the confidence, the biggest the probability. It works as a loss value.
    def get_label_confidence(self, face, gray_image):
        (x,y,w,h) = face
        # print(x,y,w,h)
        roi_gray = gray_image[y:y+h, x:x+h]
        label, confidence = self.facerecognizer.predict(roi_gray)
        return label, confidence

    def set_bounding_box(self, vid, face, label: str, confidence: float):
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
        cv2.putText(vid, text, (x, y - 5), font_face, scale, color, 1, cv2.LINE_AA)

    def set_default_bounding_box(self, vid, face, label: str, confidence: float):
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

        cv2.putText(vid, "Unknown user", (x, y - 5), font_face, scale, color, 1, cv2.LINE_AA)

    def check_key(self, char):
        return cv2.waitKey(1) & 0xFF == ord(char)


    # if __name__ == "__main__":
    #     main()


