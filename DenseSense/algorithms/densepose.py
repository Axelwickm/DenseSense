import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config

from DenseSense.algorithms.Algorithm import Algorithm
from DenseSense.Person import Person


# TODO: set these paths outside this file
config_fpath = "./models/densepose_rcnn_R_50_FPN_s1x.yaml"
model_fpath = "./models/R_50_FPN_s1x.pkl"

cfg = get_cfg()
cfg.NUM_GPUS = 1
add_densepose_config(cfg)
cfg.merge_from_file(config_fpath)
cfg.MODEL.WEIGHTS = model_fpath
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()


class DenseposeExtractor(Algorithm):
    def __init__(self):
        super().__init__()
        self.predictor = DefaultPredictor(cfg)

    def extract(self, image):
        with torch.no_grad():
            ret = self.predictor(image)["instances"].to("cpu")

        boxes = ret.get("pred_boxes")
        bodies = ret.get("pred_densepose")

        people = []
        for i in range(len(boxes)):
            # Create new person
            person = Person()
            person.bounds = boxes.tensor[i].numpy()

            S = bodies.S[i]
            I = bodies.I[i]

            # Merge S and I
            person.S = S.argmax(dim=0).cpu().numpy()         # Most activated segment (0 is background)
            mask = (person.S > 0)
            person.I = I.argmax(dim=0).cpu().numpy() * mask  # Most activated body part

            # Merge U and V
            Un = bodies.U[i].cpu().numpy().astype(np.float32)
            person.U = np.zeros(person.S.shape, dtype=np.float32)
            Vn = bodies.U[i].cpu().numpy().astype(np.float32)
            person.V = np.zeros(person.S.shape, dtype=np.float32)
            for partId in range(Un.shape[0]):
                person.U[person.I == partId] = Un[partId][person.I == partId].clip(0, 1)
                person.V[person.I == partId] = Vn[partId][person.I == partId].clip(0, 1)

            people.append(person)

        return people

    def train(self, saveModel):
        raise Exception("DensePose algorithm cannot be trained from within DenseSense")

    def renderDebug(self, image, people):
        image = image.astype(np.uint8)

        for person in people:
            # Draw bounding box rectangle
            bnds = person.bounds.astype(np.uint32)
            image = cv2.rectangle(image, (bnds[0], bnds[1]),
                                         (bnds[2], bnds[3]),
                                         (100, 100, 100), 2)

            # Get color of body parts using cv2 color map
            matrix = (person.I*(255/25)).astype(np.uint8)
            matrix = cv2.applyColorMap(matrix, cv2.COLORMAP_PARULA)

            # Generate mask
            mask3 = np.zeros(matrix.shape, dtype=np.uint8)
            mask3[person.S != 0] = 1

            # Apply UVs
            matrix = matrix.astype(np.int16)
            matrix[:, :, 0] += (person.U*200-100).astype(np.int8)
            matrix[:, :, 1] += (person.V*200-100).astype(np.int8)
            matrix = np.clip(matrix, 0, 255).astype(np.uint8)

            # Resize matrix and mask
            dims = (bnds[2]-bnds[0], bnds[3]-bnds[1])
            matrix = cv2.resize(matrix, dims, interpolation=cv2.INTER_AREA)
            mask3 = cv2.resize(mask3, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            alpha = 0.3
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            matrix = np.where(mask3, matrix*alpha + overlap*(1.0-alpha), overlap)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = matrix

        return image
