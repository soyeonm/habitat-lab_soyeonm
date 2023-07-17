from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import torch
import sys
import os
Detic_directory = os.environ["Detic_directory"]

import os
sys.path.insert(0, os.path.join(Detic_directory, 'third_party/CenterNet2/'))
from centernet.config import add_centernet_config
sys.path.insert(0, Detic_directory)
from detic.config import add_detic_config

def setup_cfg():
    config_file = os.path.join(Detic_directory,'configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    opts = ['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']
    confidence_threshold = 0.5
    pred_all_class = False

    cfg = get_cfg()
    if not(torch.cuda.is_available()): #args.cpu or not(torch.device.availble()):
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)#args.config_file)
    cfg.merge_from_list(opts)#args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold #args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold #args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold #args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not pred_all_class: #args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg

from argparse import Namespace
detic_args = Namespace(vocabulary = 'coco')


