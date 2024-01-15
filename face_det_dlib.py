from imutils.video import FileVideoStream
import argparse
import imutils
import time
import cv2
import dlib

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY

    # return our bounding box coordinates
    return (startX, startY, w, h)
    

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default='Hollywood2/actioncliptest00142.avi',
	help="path to video to apply face detection")
ap.add_argument("-m", "--model", type=str, default="mmod_human_face_detector.dat",
	help="path to dlib's CNN face detector model")
ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")

args = vars(ap.parse_args())

# Specify the path to your image
image_path = "/home/pyimagesearch/Downloads/img2.jpg"

vs = FileVideoStream(args["video"]).start()

# load dlib's CNN face detector
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1(args["model"])

# Read the image
image = cv2.imread(image_path)
image = imutils.resize(image, width=500)

results = detector(image, args["upsample"])

# convert the resulting dlib rectangle objects to bounding boxes, then ensure 
# the bounding boxes are all within the bounds of the input image
boxes = [convert_and_trim_bb(image, r.rect) for r in results]

# loop over the bounding boxes
for (x, y, w, h) in boxes:
    #draw the bounding box on our image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Frame", image)
cv2.waitKey(0)

"""start = time.time()
while True:
    # grab the frame from the file video stream, resize it, and convert it to grayscale
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector(rgb, args["upsample"])

    # convert the resulting dlib rectangle objects to bounding boxes, then ensure 
    # the bounding boxes are all within the bounds of the input image
    boxes = [convert_and_trim_bb(frame, r.rect) for r in results]

    # loop over the bounding boxes
    for (x, y, w, h) in boxes:
        #draw the bounding box on our image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)            
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start)) """ 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
