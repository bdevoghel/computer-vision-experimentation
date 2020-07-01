# tutorial followed here : https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

# USAGE example
# python scan.py --image ../data/document_scanner/receipt.jpg

import argparse
import imutils
import cv2
from skimage.filters import threshold_local
from pyimagesearch.transform import four_point_transform


def scan(image):
	# compute the ratio of the old height to the new height, clone it, and resize it
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	# convert the image to grayscale, blur it, and find edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	cv2.imshow("Image", image)
	cv2.imshow("Edged", edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our document
		if len(approx) == 4:
			doc_cnt = approx
			break

	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of paper")
	cv2.drawContours(image, [doc_cnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, doc_cnt.reshape(4, 2) * ratio)

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	scanned = warped.copy()
	scanned = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
	T = threshold_local(scanned, 11, offset=10, method="gaussian")
	scanned = (scanned > T).astype("uint8") * 255

	# show the original and scanned img
	print("STEP 3: Apply perspective transform")
	cv2.imshow("Original", imutils.resize(orig, height=650))
	cv2.imshow("Warped", imutils.resize(warped, height=650))
	cv2.imshow("Scanned", imutils.resize(scanned, height=650))
	cv2.waitKey(0)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
	args = vars(ap.parse_args())

	# load image and
	image = cv2.imread(args["image"])

	scan(image)
