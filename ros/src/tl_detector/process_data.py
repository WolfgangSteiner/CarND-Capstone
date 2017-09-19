# usage $ python click_and_crop.py --image sample.jpg
# import the necessary packages
import argparse
import cv2
import numpy as np
import glob
import pdb
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
getROI = False
refPt = []

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to Data Folder")
args = vars(ap.parse_args())
 



def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global x_start, y_start, x_end, y_end, cropping, getROI

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		x_start, y_start, x_end, y_end = x, y, x, y
		cropping = True

	elif event == cv2.EVENT_MOUSEMOVE:
		if cropping == True:
			x_end, y_end = x, y

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		x_end, y_end = x, y
		cropping = False
		getROI = True


# files = glob.glob("./data/*.jpg")

files = glob.glob(args["path"]+"/*.jpg")

# pdb.set_trace()


for pic in files:

	image = cv2.imread(pic)
	clone = image.copy()
	 
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)

	# keep looping until the 'q' key is pressed
	while True:

		i = image.copy()

		if not cropping and not getROI:
			cv2.imshow("image", i)

		elif cropping and not getROI:
			cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
			cv2.imshow("image", i)

		elif not cropping and getROI:
			cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
			cv2.imshow("image", image)

		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
			getROI = False
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			image = clone.copy()
			getROI = False
			cv2.destroyAllWindows()
			break

		# check save path(label) by key input
		elif key == ord("n"):
			save_path = "./data2/nothing"
		elif key == ord("a"):
			save_path = "./data2/red"
		elif key == ord("y"):
			save_path = "./data2/yellow"
		elif key == ord("b"):
			save_path = "./data2/green"


# cv2.destroyAllWindows()
# pdb.set_trace()
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	# if save_path == "./data/yellow":
	# 	continue 

	refPt = [(x_start, y_start), (x_end, y_end)]
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
		#resize to 64x64
		resize_roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_LINEAR)
		name = save_path +"/data2_"+ pic[8:]
		print (name)
		cv2.imwrite(name, resize_roi)
		cv2.imshow("ROI", resize_roi)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	
# close all open windows
cv2.destroyAllWindows()