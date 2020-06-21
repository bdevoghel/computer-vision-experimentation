# tutorial followed here : https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# USAGE
# python detect_faces.py --image img/rooster.jpg --prototxt model/deploy.prototxt.txt
# 												 --model model/res10_300x300_ssd_iter_140000.caffemodel

import numpy as np
import argparse
import cv2


def detect_faces(image, net):
	(h, w) = image.shape[:2]

	# construct input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass blob through the network and obtain the detections and predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence < args['confidence']:
			continue

		# compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype('int')

		print(f"Face detected at ({(startX + endX) / 2}, {(startY + endY) / 2})")

		# draw the bounding box of the face along with the associated probability
		text = '{:.2f}%'.format(confidence * 100)
		y = (startY - 10) if startY - 10 > 10 else (startY + 10)
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output image
	cv2.imshow('Output', image)
	cv2.waitKey(0)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required=True, help="path to input image")
	ap.add_argument('-p', '--prototxt', required=True, help="path to Caffe 'deploy' prototxt file")
	ap.add_argument('-m', '--model', required=True, help="path to Caffe pre-trained model")
	ap.add_argument('-c', '--confidence', type=float, default=0.5, help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

	# load input image
	image = cv2.imread(args['image'])

	detect_faces(image, net)
