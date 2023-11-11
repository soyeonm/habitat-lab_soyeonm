import json
import cv2
import random
import pickle 
import skimage.morphology
import os
os.chdir('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm')
from tiffany_scripts.divide_into_rooms_functions import *
from tiffany_scripts.fmm_planner_copied import FMMPlanner
import copy


#Conda env: old_cv2 (conda activate old_cv2)

#numpy
map_pickle = pickle.load(open('/Users/soyeonm/Documents/SocialNavigation/OGN/fbe_maps/sep_18_map_for_wall/success_spot_init/18/fbe_map_numpy.p', 'rb'))


#1-1 Get contour (colored, multiple )
contour_multiple, img = get_contour(map_pickle[0], outer_only=False)
unique_contours = np.unique(contour_multiple)
contour_multiple_dict = {c:contour_multiple==c for c in unique_contours if c!=255}

#1-2 Get contour (outer only)
contour_outer, _ = get_contour(map_pickle[0], outer_only=True)
unique_contours = np.unique(contour_outer)
assert len(unique_contours) ==2 #255 and 0
contour_outer = contour_outer[:, :, 0]==0


#2. Dilate and divide into labels 
#2-1: Dilate with outer contour + drawer + wall only #4 is wall and 15 is drawer
added = ((map_pickle[4] + contour_outer*1.0 + map_pickle[15])>0) * 1.0
selem = skimage.morphology.square(15)
dilated = skimage.morphology.binary_dilation(added, selem) * 1.0

#2-2: Divide into labels 
connected_regions = skimage.morphology.label(1-dilated, connectivity=2)
#Get a dict of connected regions
unique_regions = np.unique(connected_regions)
#Get dict of connected regions
connected_regions_dict = {c: connected_regions==c for c in unique_regions if c!=0}

#3. Manually identify regions that are too big!
#Get closest contours for these regions 
list_of_too_big_regions = [connected_regions_dict[3]] #Manually IDENTIFY THESE!
selem = skimage.morphology.square(20)
leave3mostly_try = added + (((skimage.morphology.binary_dilation(connected_regions==1, selem) + skimage.morphology.binary_dilation(connected_regions==2, selem) + skimage.morphology.binary_dilation(connected_regions==4, selem) + skimage.morphology.binary_dilation(connected_regions==5, selem))>0)*1.0)
#leave3mostly_try = (leave3mostly_try >0) *1.0
#Not sure how this came out! Just get pickle for now 
#leave3mostly = pickle.load(open('/Users/soyeonm/Documents/SocialNavigation/OGN/fbe_maps/sep_18_map_for_wall/success_spot_init/18/leave3mostly.p', 'rb')) 
leave3mostly = leave3mostly_try

#CHANGE THIS TOO!

#4.Divide list_of_too_big_regions into smaller rooms
#Match closest contour multiple
for cr in list_of_too_big_regions:
	traversible =1-leave3mostly
	planner = FMMPlanner(traversible)
	dist_to_contour_dict = {}
	for mc, mc_contour in contour_multiple_dict.items():
		planner.set_multi_goal(mc_contour[:, :, 0])
		dist_to_contour_dict[mc] = copy.deepcopy(planner.fmm_dist)
	#stacked_dist = np.stack((dist_to_contour0, dist_to_contour1, dist_to_contour2, dist_to_contour3, dist_to_contour4, dist_to_contour5, dist_to_contour6))
	stacked_dist = np.stack(list(dist_to_contour_dict.values()))
	dist_argmin =np.argmin(stacked_dist, 0) #*traversible
	#
	unique_room_numbers = np.unique(dist_argmin)
	#splitted_room_dict = {u:dist_argmin==u for u in unique_room_numbers if np.sum(dist_argmin==u)>0}
	splitted_room_dict =  {} 
	for u in unique_room_numbers:
		contor_u_trav  = (dist_argmin == u) * traversible
		if np.sum(contor_u_trav) >0:
			splitted_room_dict[u] = contor_u_trav  #This has value 0, -1, 1?


#Combine connected_regiions_dict except for 3 and combine the dict above
#Just multply with dilated, etc

#Just visualize
import random
room_img = copy.deepcopy(img)
for ci, r in splitted_room_dict.items(): 
	rand_num = random.randrange(0, 255)
	room_img[:, :, 0][np.where(r )] = random.randrange(0, 255)
	room_img[:, :, 1][np.where(r )] = random.randrange(0, 255)
	room_img[:, :, 2][np.where(r )] = random.randrange(0, 255)
cv2.imshow("room img", room_img); cv2.waitKey(1)

#Let's also show all the rooms
for ci, con_region in connected_regions_dict.items():
	if ci !=3 :
		room_img[:, :, 0][np.where(con_region)] = random.randrange(0, 255)
		room_img[:, :, 1][np.where(con_region)] = random.randrange(0, 255)
		room_img[:, :, 2][np.where(con_region)] = random.randrange(0, 255) 

cv2.imshow("room img", room_img); cv2.waitKey(1)
os.makedirs("/Users/soyeonm/Documents/SocialNavigation/figures_and_such")
cv2.imwrite("/Users/soyeonm/Documents/SocialNavigation/figures_and_such/example_room_img_retry_selem20.png", room_img)
#Now save to pickle

#example_room_img is from the pickle
#example_room_img_retry is from the selem
#example_room_img_retry_0_1 is from the selem and then leave3mostly_try >0) *1.0
#example_room_img_retry_selem20 is from 
#############################################
#selem = skimage.morphology.square(20)
#leave3mostly_try = added + (((skimage.morphology.binary_dilation(connected_regions==1, selem) + skimage.morphology.binary_dilation(connected_regions==2, selem) + skimage.morphology.binary_dilation(connected_regions==4, selem) + skimage.morphology.binary_dilation(connected_regions==5, selem))>0)*1.0)
#leave3mostly_try = (leave3mostly_try >0) *1.0
#Not sure how this came out! Just get pickle for now 
#leave3mostly = pickle.load(open('/Users/soyeonm/Documents/SocialNavigation/OGN/fbe_maps/sep_18_map_for_wall/success_spot_init/18/leave3mostly.p', 'rb')) 
#leave3mostly = leave3mostly_try
#############################################

#Let's do this
#all the rooms 
all_the_rooms_dict = {}
counter = 0
for ci, con_region in connected_regions_dict.items():
	if ci ==0:
		cv2.imshow("con region 0", con_region); cv2.waitKey(1)
	if ci !=3 :
		all_the_rooms_dict[counter] = con_region>0
		counter +=1

for ci, r in splitted_room_dict.items(): 
	all_the_rooms_dict[counter] =  r>0 
	counter +=1



pickle.dump(all_the_rooms_dict, open('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/room_divide_fpss/18_rooms.p', 'wb'))


#Just temporary fix for non room places
all_the_rooms_dict.pop(0)
pickle.dump(all_the_rooms_dict, open('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/room_divide_fpss/18_rooms.p', 'wb'))


