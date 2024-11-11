import cv2

# WIP I would like this to recognize me and my friends so I would like to see if works for them.
CC_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():

    # Following tut: https://www.datacamp.com/tutorial/face-detection-python-opencv

    # Get the video
    video_stream = cv2.VideoCapture(0)

    while True:

        result, video_frame = video_stream.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfully

        faces = detect_bounding_box(video_frame)

        cv2.imshow("A Butler should know your face", video_frame) 

        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

    video_stream.release()
    cv2.destroyAllWindows()




def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = CC_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

if __name__ == "__main__":
    main()


