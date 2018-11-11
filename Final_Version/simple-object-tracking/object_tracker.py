# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

def human_detect (image_provide):
	ret_val = []
	args = {}
	args["prototxt"] = "MobileNetSSD_deploy.prototxt.txt"
	args["model"] = "MobileNetSSD_deploy.caffemodel"
	args["image"] = image_provide
	args["confidence"] = 0.2

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	#	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)
	image = cv2.imread(args["image"])
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	#print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			if CLASSES[idx] == "person":
				#print("{} {} {} {} {:.2f}".format(startX,startY,endX,endY,confidence*100))
				tuple1 = (startX,endX,startY,endY)
				ret_val.append(tuple1)

			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			#print("[INFO] {}".format(label))
			#cv2.ellipse(image,((startX+endX)//2,(startY+endY)//2),((endY-startY)//2,(endX-startX)//2),0,0,360,10,1)
			cv2.rectangle(image, (startX, startY), (endX, endY),
				COLORS[idx], 2)
#			cv2.imshow('pr',image)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(image, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Output", image)
	cv2.waitKey(0)

	return ret_val




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
#print(" starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
vid = raw_input("Enter video source\n")
vidcap = cv2.VideoCapture(vid)
success,image = vidcap.read()
trajectory = []
# loop over the frames from the video stream
while success:
	cv2.imwrite("frame1.jpeg", image)     # save frame as JPEG file
  	success,image = vidcap.read()

	human_detect("frame1.jpeg")
	
	image = imutils.resize(image, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = image.shape[:2]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		#compute centroid trajectory
		trajectory.append((objectID,(centroid[0],centroid[1])))

	# show the output frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()

#pass trajectory to neural network
print(trajectory)

change_in_pos = []
size_ = 0
prev=(0,0)
for ip in trajectory:
	if size_ == 0:
		change_in_pos.append((0,0))
		size_=1
		prev = ip[1]
	else:
		change_in_pos.append(((ip[1][0] - prev[0]),(ip[1][1]-prev[1])))
		prev = ip[1]

print(change_in_pos)
