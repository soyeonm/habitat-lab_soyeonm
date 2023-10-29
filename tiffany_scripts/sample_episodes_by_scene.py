import json
a = json.load(open('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/datasets/floorplanner/rearrange/scratch/train/largetrain_15s_15000epi_2obj.json', 'r'))
a.keys()
dict_keys(['episodes', 'config'])
a['config']
a['episodes'][0]
{'episode_id': '0', 'scene_id': 'data/fpss/fphab/scenes-uncluttered/102344250.scene_instance.json', 'scene_dataset_config': 'data/fpss/fphab/fphab-uncluttered.scene_dataset_config.json', 'additional_obj_config_paths': ['data/objects/ycb/configs/'], 'start_position': [0, 0, 0], 'start_rotation': [0, 0, 0, 1], 'info': {'object_labels': {'014_lemon_:0000': 'any_targets|0', '005_tomato_soup_can_:0000': 'any_targets|1'}}, 'ao_states': {}, 'rigid_objs': [['014_lemon.object_config.json', [[-0.20107, 0.03078, 0.97909, -18.33013], [0.10313, 0.99462, -0.01009, 0.93156], [-0.97413, 0.09894, -0.20317, -9.89595], [0.0, 0.0, 0.0, 1.0]]], ['005_tomato_soup_can.object_config.json', [[0.99422, -0.0035, -0.10731, -9.10242], [0.00329, 0.99999, -0.00212, 0.78535], [0.10731, 0.00175, 0.99422, -4.60911], [0.0, 0.0, 0.0, 1.0]]], ['026_sponge.object_config.json', [[-0.39562, 0.10158, -0.91278, -15.78521], [0.05826, 0.99464, 0.08543, 0.47264], [0.91657, -0.01938, -0.39941, -2.73432], [0.0, 0.0, 0.0, 1.0]]]], 'targets': {'014_lemon_:0000': [[-0.76695, 0.0, -0.64171, -4.07266], [0.0, 1.0, 0.0, 0.5284], [0.64171, 0.0, -0.76695, -4.09917], [0.0, 0.0, 0.0, 1.0]], '005_tomato_soup_can_:0000': [[0.66277, 0.0, -0.74882, -7.01594], [0.0, 1.0, 0.0, 0.95765], [0.74882, 0.0, 0.66277, -1.35844], [0.0, 0.0, 0.0, 1.0]]}, 'markers': [], 'target_receptacles': [['c2d198ebe2cd8c70b0886e60d98b327f1934aa0e_:0003', None], ['a7931cc380b0cb2b5d31a212280182a458701032_:0000', None]], 'goal_receptacles': [['7e7974b180d5ce8ef51e76350668b14b47568f7e_:0000', None], ['c2c17462cc5cba5908d12060aaa28efbff652888_:0000', None]], 'name_to_receptacle': {'014_lemon_:0000': 'c2d198ebe2cd8c70b0886e60d98b327f1934aa0e_:0003|receptacle_mesh_c2d198ebe2cd8c70b0886e60d98b327f1934aa0e.0000', '005_tomato_soup_can_:0000': 'a7931cc380b0cb2b5d31a212280182a458701032_:0000|receptacle_mesh_a7931cc380b0cb2b5d31a212280182a458701032.0000', '026_sponge_:0000': '50ff6446cdafa03172df41c1244dfac0f8e6bcc8_:0000|receptacle_mesh_50ff6446cdafa03172df41c1244dfac0f8e6bcc8.0000'}}

scene_ids = {}
for i, e_ep in enumerate(a['episodes']):
    scene_id = e_ep['scene_id']
    if not(scene_id in scene_ids):
            scene_ids[scene_id] = []
    scene_ids[scene_id].append((i, e_ep['episode_id']))


scene_ids.keys()
dict_keys(['data/fpss/fphab/scenes-uncluttered/102344250.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/104862639_172226823.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/102344280.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/102816216.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/103997424_171030444.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/103997460_171030507.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/103997919_171031233.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/104348463_171513588.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/104862621_172226772.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/106366173_174226431.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/106878945_174887058.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/108736851_177263586.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/orig_108736872_177263607.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/107734479_176000442.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/108736872_177263607.scene_instance.json', 'data/fpss/fphab/scenes-uncluttered/104348361_171513414.scene_instance.json'])
for k in scene_ids:
    print(len(scene_ids[k]))


#Just pick 50 each
#Make one version
#Pick 100 each
#Make another version

import random
sampled_50_json_dict = {}
sampled_50_json_dict['config'] = a['config']
sampled_50_json_dict['episodes'] = []
episodes_sampled = []
sampled_indices_50_for_each_scene = {}
for k, v in scene_ids.items():
    random.seed(0)
    sampled_50 = random.sample(scene_ids[k], 20)
    for s in sampled_50:
            #sampled_50_json_dict['episodes'].append(a['episodes'][s[0]])
            episodes_sampled.append(a['episodes'][s[0]])

random.seed(1)
random.shuffle(episodes_sampled)
sampled_50_json_dict['episodes']  = episodes_sampled

json.dump(sampled_50_json_dict, open('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/datasets/floorplanner/rearrange/scratch/train/sampled_20_largetrain_15s_15000epi_2obj.json', 'w'))


json.dump(sampled_50_json_dict, open('/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/datasets/floorplanner/rearrange/scratch/train/sampled_50_largetrain_15s_15000epi_2obj.json', 'w'))
gzip /Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/datasets/floorplanner/rearrange/scratch/train/sampled_50_largetrain_15s_15000epi_2obj.json