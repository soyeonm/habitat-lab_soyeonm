import csv

#Just use the same one in fpss folder

file_path = "data/fpss/fphab/semantics/fpmodels.csv"
#file_path = 'fpModels-v0.2.3.csv' #'/Users/soyeonm/Downloads/fpModels-v0.2.3.csv'
data_dict = {}
col_indices_2_keys = {}
#searh by id 
with open(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Assuming the first row contains the headers
    headers = next(reader)
    col_indices_2_keys = {i: k for i, k in enumerate(headers)}
    for row in reader:
    	cur_id = None
    	for ri, r in enumerate(row):
    		#The first one is id 
    		if ri ==0:
    			cur_id = r
    			data_dict[cur_id] = {}
    		#col_indices_2_keys[ri]
    		#Add the rest 
    		cur_key = col_indices_2_keys[ri]
    		data_dict[cur_id][cur_key] = r
        # Assuming the first column contains unique keys
        # key = row[0]
        
        # # Store the remaining values as a list in the dictionary
        # values = row[1:]
        
        # data_dict[key] = values

#Let's keep this dict in a pickle
import pickle
#pickle.dump(data_dict, open('', 'wb'))
#not yet

main_categories = {} #length is 86
for k, entry in data_dict.items():
	mc = entry['main_category']
	if not (mc in main_categories):
		main_categories[mc] = len(main_categories)
	#if not (mc in main_categories):

import json
lexicon = json.load(open("data/fpss/fphab/semantics/object_semantic_id_mapping.json", 'r'))


for l in lexicon: #length is 28
	assert l in main_categories

for m in main_categories: #
	if not m in lexicon:
		print(m)

#Just add person for now in the lexicon -> Just add 102 actually 
#What is the person in the main_categories? Different from glb?
#

#Now let's add everything for the objects
objects_folder = 'data/fpss/fphab/objects' #'/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/fpss/fphab/objects'
from glob import glob
object_jsons = glob(objects_folder + '/*/*.json')
#Maybe throw away all the openings 

#Put in "'semantic_id'" keys to all the loaded jsons 
#For now, just do the ones in the main_categories lexicon 
updated_count = 0
for i, object_json_file_name in enumerate(object_jsons):
	if i%100 ==0:
		print(i)
	#Just do the ones in data_dict
	json_key = object_json_file_name.split('/')[-1].replace('.object_config.json', '') 
	if json_key in data_dict: #Most are here, only 1 is not in here
		cur_data_dict = data_dict[json_key]
		loaded_json = json.load(open(object_json_file_name, 'r'))
		#Just do the ones in lexicon for now
		#Actually do all the categories
		if cur_data_dict['main_category'] in lexicon: #main_categories: #lexicon:
			#if cur_data_dict['main_category'] != '':
			#loaded_json['semantic_id'] = main_categories[cur_data_dict['main_category']]
			loaded_json['semantic_id'] = lexicon[cur_data_dict['main_category']]
			updated_count +=1 #7491 when lexicon
			#Save back to json 
			json.dump(loaded_json, open(object_json_file_name,'w'))



	#break
	#assert json_key in data_dict

#Now convert these into lexicon 
#lexicon_to_save = {}
# for k, v in main_categories.items():
# 	if k!= '':
#main_categories.pop('')
#just write into json 
#json.dump(main_categories, open('data/fpss/fphab/semantics/object_semantic_id_mapping_main_categories.json', 'w'))


#YCB, Berkeley amazon, google

import csv

file_path = "/Users/soyeonm/Documents/object_annotated_categories.csv"


handle_2_category = {}
all_categories = {}
#search by id 
with open(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Assuming the first row contains the headers
    headers = next(reader)
    for row in reader:
    	#breakpoint()
    	handle_2_category[row[0]] = row[1]
    	if not (row[1]) in all_categories:
    		all_categories[row[1]] = len(all_categories)

#pickle
import pickle
pickle.dump(handle_2_category, open('ycb_ba_g_handle_2_category.p', 'wb'))
pickle.dump(all_categories, open('ycb_ba_g_all_categories.p', 'wb'))
