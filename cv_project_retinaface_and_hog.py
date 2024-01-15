#RetinaFace implementation for image

import cv2
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Loading the image
img = cv2.imread("img2.jpg")

# Detecting faces
faces = RetinaFace.detect_faces(img_path="img2.jpg")

if faces:
    for face in faces.values():
        # Getting the coordinates of the bounding box
        facial_area = face['facial_area']
        x, y, w, h = facial_area

        # Drawing a rectangle around each face
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

# Converting color from BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Displaying the image
plt.imshow(img_rgb)
plt.show()

#RetinaFace implementation for video

import cv2
from retinaface import RetinaFace

cap = cv2.VideoCapture('vid2.mp4')

# Preparing to write the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('retina_output_vid2.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detecting faces
    faces = RetinaFace.detect_faces(frame)

    if faces:
        for face in faces.values():
            # Getting the coordinates of the bounding box
            facial_area = face['facial_area']
            x, y, w, h = facial_area

            # Drawing a rectangle around each face
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    # output
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

# HOG implementation for image

import cv2
import dlib
from google.colab.patches import cv2_imshow

def detect_faces_in_image(image_path):
    # Loading the detector
    detector = dlib.get_frontal_face_detector()

    # Loading the image
    img = cv2.imread(image_path)

    # Converting to grayscale (needed for face detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = detector(gray)

    # Drawing bounding boxes
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

    # Displaying the output
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# implementation / function call
detect_faces_in_image("img10.png")

# HOG implementation for video

import cv2
import dlib

def detect_faces_in_video(video_path, output_path):
    # Loading the detector
    detector = dlib.get_frontal_face_detector()

    # Opening the video
    cap = cv2.VideoCapture(video_path)

    # Video writer
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converting to grayscale and resizing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Upsampling image
        upscaled_gray = cv2.pyrUp(gray)

        # Detecting faces
        faces = detector(upscaled_gray, 1)

        for face in faces:
            # Scaling to the original frame size
            x, y = int(face.left()/2), int(face.top()/2)
            x1, y1 = int(face.right()/2), int(face.bottom()/2)
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# function call
detect_faces_in_video("vid2.mp4", "hog_output_vid2.avi")