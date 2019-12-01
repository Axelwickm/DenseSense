# Import caffe2 and Detectron
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config

import DenseSense.algorithms.Algorithm


config_fpath = "./models/densepose_rcnn_R_50_FPN_s1x.yaml"
model_fpath = "./models/R_50_FPN_s1x.pkl"

cfg = get_cfg()
cfg.NUM_GPUS = 1
add_densepose_config(cfg)
cfg.merge_from_file(config_fpath)
cfg.MODEL.WEIGHTS = model_fpath
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()

class DenseposeExtractor(DenseSense.algorithms.Algorithm.Algorithm):

    def __init__(self):
        self.predictor = DefaultPredictor(cfg)
        return

    def extract(self, image):
        with torch.no_grad():
            ret = self.predictor(image)["instances"].to("cpu")
            boxes = ret.get("pred_boxes")
            bodies = ret.get("pred_densepose")

        return tuple([boxes, bodies])
    
    def train(self, saveModel):
        raise Exception("Densepose algorithm cannot be trained from within DenseSense")