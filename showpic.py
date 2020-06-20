import argparse
import cv2
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# if the width is greater than 640 pixels, then resize the image
if image.shape[1] > 640:
    image = imutils.resize(image, width=640)

while True:
    cv2.imshow("foo", image)
    cv2.waitKey(0)
