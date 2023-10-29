import json
import cv2
import random
import pickle 
import numpy as np
import skimage.morphology
import ipdb

def get_contour(obstacle_map, outer_only=False):
	map_pickle = obstacle_map
	if outer_only:
		selem = skimage.morphology.square(1)
		dilated = skimage.morphology.binary_dilation(map_pickle, selem)*1.0
		#erode back
		#map_pickle = skimage.morphology.binary_erosion(map_pickle, selem)*1.0
		map_pickle = dilated
		area_threshold = 40000
	else:
		area_threshold = 600
	img_reshaped = map_pickle[:, :, np.newaxis]
	arr_repeated = np.repeat(img_reshaped, 3, axis=2)
	img = ((1-arr_repeated)*255).astype(np.uint8)
	#breakpoint()
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	#
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	#
	cv2.imshow("thresh", thresh)
	#
	mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (5, 5), iterations=3)
	#
	_, contours, _ = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#
	sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
	#
	blank_img = (np.ones(img.shape)*255).astype(np.uint8)
	for ci, c in enumerate(sorted_contours[1:]):
		area = cv2.contourArea(c)
		print("area is ", area)
		if area > area_threshold: #40000: #600: #>20000: #< 20000 and area>150: #> 600:#100: #6000:
			save_c = c
			cv2.drawContours(img, [c], -1, (random.randrange(0, 255), random.randrange(0, 256), random.randrange(0, 255)), 3)
			#blank_img = (np.ones(img.shape)*255).astype(np.uint8)
			#cv2.drawContours(blank_img, [c], -1, (0, 0, 0), thickness=cv2.FILLED)
			cv2.drawContours(blank_img, [c], -1, (ci, ci, ci), 1)
			print("drew ", ci)
			#
	#cv2.imshow("mor_img", mor_img); cv2.waitKey(1)
	#cv2.imshow("img", img); cv2.waitKey(1)
	#cv2.imshow("filled_contour", blank_img); cv2.waitKey(1)
	return blank_img, img 