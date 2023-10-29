import numpy as np
from glob import glob
import pickle
import numpy

root = 'RERE_panoptic_save_dir'

globs = glob(root + '/*/stats_dict.p')
combined_stats_dict = {}
for g in globs:
	stats_dict = pickle.load(open(g, 'rb'))
	combined_stats_dict[g] = stats_dict

#Get total 

#Get mean 
#'beginning_gt_human_visible', 'middle_gt_human_visible', 'end_gt_human_visible', 'beginning_gt_target1_visible', 'middle_gt_target1_visible', 'end_gt_target1_visible'
key_list = list(combined_stats_dict.keys())
valid_episode_list = np.array([combined_stats_dict[k]['valid_episod'] for k in key_list])
beginning_gt_human_visible_list = np.array([combined_stats_dict[k]['beginning_gt_human_visible'] for k in key_list]) 
middle_gt_human_visible_list = np.array([combined_stats_dict[k]['middle_gt_human_visible'] for k in key_list])
end_gt_human_visible_list = np.array([combined_stats_dict[k]['end_gt_human_visible'] for k in key_list])

beginning_gt_targ1_visible_list = np.array([combined_stats_dict[k]['beginning_gt_target1_visible'] for k in key_list])
middle_gt_targ1_visible_list = np.array([combined_stats_dict[k]['middle_gt_target1_visible'] for k in key_list])
end_gt_targ1_visible_list = np.array([combined_stats_dict[k]['end_gt_target1_visible'] for k in key_list])

beggining_human_visible_rate = np.sum(beginning_gt_human_visible_list * valid_episode_list)/ np.sum(valid_episode_list)
middle_human_visible_rate = np.sum(middle_gt_human_visible_list * valid_episode_list)/ np.sum(valid_episode_list)
end_human_visible_rate = np.sum(end_gt_human_visible_list * valid_episode_list)/ np.sum(valid_episode_list)

beggining_targ1_visible_rate = np.sum(beginning_gt_targ1_visible_list * valid_episode_list)/ np.sum(valid_episode_list)
middle_targ1_visible_rate = np.sum(middle_gt_targ1_visible_list * valid_episode_list)/ np.sum(valid_episode_list)
end_targ1_visible_rate = np.sum(end_gt_targ1_visible_list * valid_episode_list)/ np.sum(valid_episode_list)


beggining_human_visible_nonzero = np.sum((beginning_gt_human_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)
middle_human_visible_nonzero = np.sum((middle_gt_human_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)
end_human_visible_nonzero = np.sum((end_gt_human_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)

beggining_targ1_visible_nonzero = np.sum((beginning_gt_targ1_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)
middle_targ1_visible_nonzero = np.sum((middle_gt_targ1_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)
end_targ1_visible_nonzero = np.sum((end_gt_targ1_visible_list!=0) * valid_episode_list)/ np.sum(valid_episode_list)


print("beggining_human_visible_rate", beggining_human_visible_rate)
print("middle_human_visible_rate", middle_human_visible_rate)
print("end_human_visible_rate", end_human_visible_rate)

print("beggining_targ1_visible_rate", beggining_targ1_visible_rate)
print("middle_targ1_visible_rate", middle_targ1_visible_rate)
print("end_targ1_visible_rate", end_targ1_visible_rate)

print("beggining_human_visible_nonzero", beggining_human_visible_nonzero)
print("middle_human_visible_nonzero", middle_human_visible_nonzero)
print("end_human_visible_nonzero", end_human_visible_nonzero)

print("beggining_targ1_visible_nonzero", beggining_targ1_visible_nonzero)
print("middle_targ1_visible_nonzero", middle_targ1_visible_nonzero)
print("end_targ1_visible_nonzero", end_targ1_visible_nonzero)

#only nonzero

nz_beggining_human_visible_rate = np.sum(beginning_gt_human_visible_list * (beginning_gt_human_visible_list!=0))/ np.sum(beginning_gt_human_visible_list!=0)
nz_middle_human_visible_rate = np.sum(middle_gt_human_visible_list * (middle_gt_human_visible_list!=0))/ np.sum(middle_gt_human_visible_list!=0)
nz_end_human_visible_rate = np.sum(end_gt_human_visible_list * (end_gt_human_visible_list!=0))/ np.sum(end_gt_human_visible_list!=0)

nz_beggining_targ1_visible_rate = np.sum(beginning_gt_targ1_visible_list * (beginning_gt_targ1_visible_list!=0))/ np.sum(beginning_gt_targ1_visible_list!=0)
nz_middle_targ1_visible_rate = np.sum(middle_gt_targ1_visible_list * (middle_gt_targ1_visible_list!=0))/ np.sum(middle_gt_targ1_visible_list!=0)
nz_end_targ1_visible_rate = np.sum(end_gt_targ1_visible_list * (end_gt_targ1_visible_list!=0))/ np.sum(end_gt_targ1_visible_list!=0)


print("nz beggining_human_visible_rate", nz_beggining_human_visible_rate)
print("nz middle_human_visible_rate", nz_middle_human_visible_rate)
print("nz end_human_visible_rate", nz_end_human_visible_rate)

print("nz beggining_targ1_visible_rate", nz_beggining_targ1_visible_rate)
print("nz middle_targ1_visible_rate", nz_middle_targ1_visible_rate)
print("nz end_targ1_visible_rate", nz_end_targ1_visible_rate)


#Look at the frames that the human was visible
#Get targ_viz and hum_viz

targ_viz_folders = glob(root + '/*/targ_viz')
hum_viz_folders = glob(root + '/*/hum_viz')

import subprocess

#move to outer directory
import os
for tf in targ_viz_folders:
	tf_replace = tf.replace('RERE_panoptic_save_dir', 'targ_viz_RERE_panoptic_save_dir')
	if not (os.path.exists(tf_replace)):
		os.makedirs(tf_replace)
	subprocess.call(["cp", "-r", tf, tf_replace])


for hf in hum_viz_folders:
	hf_replace = hf.replace('RERE_panoptic_save_dir', 'hum_viz_RERE_panoptic_save_dir')
	if not (os.path.exists(hf_replace)):
		os.makedirs(hf_replace)
	subprocess.call(["cp", "-r", hf, hf_replace])


#Look at the frames that the object was visible 


combined_stats_dict['/Users/soyeonm/Documents/SocialNavigation/old1_PICK_panoptic_save_dir/77/stats_dict.p'].keys()

for g in globs:
	stats_dict = pickle.load(open(g, 'rb'))
	print("valid episode ", stats_dict['valid_episod'])