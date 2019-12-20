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
            bounds = boxes.tensor[i].numpy()
            person = Person()
            person.bounds = bounds
            person.S = bodies.S[i]
            person.I = bodies.I[i]
            person.U = bodies.U[i]
            person.V = bodies.V[i]

            person.S_ind = person.S.argmax(dim=0).cpu().numpy()  # Most activated segment (0 is background)
            person.I_ind = person.I.argmax(dim=0).cpu().numpy() * (person.S_ind > 0)  # Most activated body part

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
                                         (100, 100, 100), 1)
            # Get color of bodyparts using cv2 color map
            matrix = (person.I_ind*(255/25)).astype(np.uint8)
            matrix = cv2.applyColorMap(matrix, cv2.COLORMAP_PARULA)

            # Generate mask
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[person.S_ind < 0] = 1

            # Resize matrix and mask
            dims = (bnds[2]-bnds[0], bnds[3]-bnds[1])
            matrix = cv2.resize(matrix, dims, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            alpha = 0.2
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            matrix = matrix*alpha + overlap*(1.0-alpha)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = matrix

        for person in people:
            break
            print(image.shape)
            print("I", person.I.shape)
            print("U", person.U.shape)
            print("V", person.V.shape)
            print("S", person.S.shape)
            personOverlay = np.dstack((50 + person.I * 8, person.U * 255, person.V * 255))
            personOverlayGuide = np.dstack((person.I, person.I, person.I))
            print("Person Overlay", personOverlay.shape)
            print("Person Overlay Guide", personOverlayGuide.shape)
            print("Unique S:")
            for k in person.S:
                print(k.shape)
            print(person.S[0, :, :])
            image = np.where(personOverlayGuide, personOverlay, image)

        return image
