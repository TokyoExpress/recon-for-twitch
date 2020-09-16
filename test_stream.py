# USAGE
# python test_stream.py --model output/simple_neural_network.hdf5 --video test.mp4 --pred ow

# import the necessary packages
from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

def image_to_feature_vector(image, size=(160, 90)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
ap.add_argument("-v", "--video", required=True,
	help="path to the video")
ap.add_argument("-p", "--pred", required=True,
	help="prediction")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())

# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["apex", "csgo", "fallguys", "fortnite", "league", "mw", "ow", "val"]

# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])
print("[INFO] initializing video analysis...")

vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)
count = 0
correct = 0

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	else:
		count += 1

	features = image_to_feature_vector(frame) / 255.0
	features = np.array([features])

	# classify the image using our extracted features and pre-trained
	# neural network
	probs = model.predict(features)[0]
	prediction = probs.argmax(axis=0)

	# draw the class and probability on the test image and display it
	# to our screen
	label = "{}: {:.2f}%".format(CLASSES[prediction],
		probs[prediction] * 100)
	if(CLASSES[prediction] == args["pred"]):
		correct += 1
	frac = "accuracy: {:.2f}%".format(correct / count * 100)

	cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)
	cv2.putText(frame, frac, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 255, 0), 3)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()