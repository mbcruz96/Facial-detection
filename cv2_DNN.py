from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                default='CondensedMovies/data/videos/2018/mkbZvLfg2Yg.mp4',
                help="path to video to apply face detection")
ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Specify the path to your image
image_path = "/home/pyimagesearch/Downloads/img2.jpg"

# Read the image
image = cv2.imread(image_path)
image = imutils.resize(image, width=700)

(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# pass the blob through the network and obtain the detections and predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
    if confidence < args["confidence"]:
        continue
    # compute the (x, y)-coordinates of the bounding box for the object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    # draw the bounding box of the face along with the associated probability
    text = "{:.2f}%".format(confidence * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
    cv2.putText(image, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
"""

#Initialize the FileVideoStream to read video file
vs = FileVideoStream(args["video"]).start()
dlibDict = {}
index = 0

start = time.time()
while True:
    # grab the frame from the file video stream,resize it, and convert it to grayscale
    frame = vs.read()
    if frame is None:
        break

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        dlibDict[index] = (startX, startY, endX, endY)
        index+=1

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        #print("Each Frame coordinate: ",(startX, startY), " and ", (endX, endY))

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()"""
