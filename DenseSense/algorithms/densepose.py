# Import caffe2 and Detectron
from numpy import np

from caffe2.python import workspace

import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.core.config import (assert_and_infer_cfg, cfg, merge_cfg_from_file)
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging

import .Algorithm

# Configure caffe2

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Init densepose model
merge_cfg_from_file("/pose_detection_payload/DensePose_ResNet50_FPN_s1x-e2e.yaml")
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg("/shared/DensePose_ResNet50_FPN_s1x-e2e.pkl")


class DenseposeExtractor(Algorithm):

    def __init__(self):
        return

    def extract(self, image):
        with c2_utils.NamedCudaScope(0):
            boxes, segms, keyps, bodys = infer_engine.im_detect_all(
                model, image, None)
            
            if isinstance(boxes, list):
                box_list = [b for b in boxes if len(b) > 0]
                if 0 < len(box_list):
                    boxes = np.concatenate(box_list)
                else:
                    boxes = None

            if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < 0.7:
                return (np.empty(0), []) # Failed, return nothing
            
            return tuple([boxes, bodys[1]])
    
    def train(self, saveModel):
        raise Exception("Densepose algorithm cannot be trained")