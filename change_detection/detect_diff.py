# tutorial followed here : https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

# USAGE
# python detect_diff.py -f ../data/change_detection/sushi1.jpg -s ../data/change_detection/sushi2.jpg

import argparse
import imutils
import cv2
from skimage.metrics import structural_similarity


def detect_diff(image_a, image_b):
    # convert the images to grayscale
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # compute the SSIM between the two images, returning the difference image and the difference score
    # SSIM : Structural Similarity Index
    diff_score, diff_img = structural_similarity(gray_a, gray_b, full=True)  # score of 1 is perfect match
    diff_img = (diff_img * 255).astype('uint8')
    print(f"SSIM: {diff_score}")

    # threshold the difference image
    thresh_img = cv2.threshold(diff_img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find contours to obtain the regions of the two input images that differ
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    size_thresh = min(image_a.shape[:2]) * 0.02
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # drop too small contours
        if w < size_thresh or h < size_thresh:
            continue

        # draw the bounding box on both input images to represent where the two images differ
        cv2.rectangle(image_b, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image_a, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output images
    cv2.imshow("First", image_a)
    cv2.imshow("Second", image_b)
    cv2.imshow("Difference", diff_img)
    cv2.imshow("Threshold", thresh_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--first', required=True, help="first input image")
    ap.add_argument('-s', '--second', required=True, help="second input image")
    args = vars(ap.parse_args())

    # load the two input images
    image_a = cv2.imread(args['first'])
    image_b = cv2.imread(args['second'])

    detect_diff(image_a, image_b)
