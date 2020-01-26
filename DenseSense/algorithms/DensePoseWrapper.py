from DenseSense.Person import Person
from DenseSense.algorithms.Algorithm import Algorithm

import cv2
import numpy as np
import os
import psutil

import torch
from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

# TODO: set these paths outside this file
config_fpath = "./models/densepose_rcnn_R_50_FPN_s1x.yaml"
model_fpath = "./models/R_50_FPN_s1x.pkl"

cfg = get_cfg()
cfg.NUM_GPUS = 1
add_densepose_config(cfg)
cfg.merge_from_file(config_fpath)
cfg.MODEL.WEIGHTS = model_fpath
cfg.MODEL.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg.freeze()

print("DensePose running on: ", cfg.MODEL.DEVICE)


class DensePoseWrapper(Algorithm):
    def __init__(self, maxImageDim=320):
        super().__init__()
        self.predictor = DefaultPredictor(cfg)
        self.AREA_THRESHOLD = 40 * 40
        self.MaxImageDim = maxImageDim

    def extract(self, image):
        # Potentially scale down image to limit memory usage
        oDim = image.shape  # old dims
        image = self._rescaleImage(image)
        nDim = image.shape  # new dims

        # Do inference
        with torch.no_grad():
            process = psutil.Process(os.getpid())
            #print("Memory usage before DensePose: {0:.2f} GB".format(process.memory_info().rss / 1e9))
            ret = self.predictor(image)["instances"].to("cpu")

        # Do post processing and compile results into list
        boxes = ret.get("pred_boxes")
        bodies = ret.get("pred_densepose")
        people = []
        for i in range(len(boxes)):
            # Transform internal bounds to original bounds
            bounds = boxes.tensor[i].numpy()
            bounds[::2] = bounds[::2] / nDim[1] * oDim[1]
            bounds[1::2] = bounds[1::2] / nDim[0] * oDim[0]
            bounds = bounds.astype(np.int32)

            # Filter depending on area
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            if area < self.AREA_THRESHOLD:
                continue

            # Create new person
            person = Person()
            person.bounds = bounds

            S = bodies.S[i]
            I = bodies.I[i]

            # Merge S and I
            person.S = S.argmax(dim=0).cpu().numpy()  # Most activated segment (0 is background)
            mask = (person.S > 0)
            person.I = I.argmax(dim=0).cpu().numpy() * mask  # Most activated body part

            # Merge U and V
            Un = bodies.U[i].cpu().numpy().astype(np.float32)
            person.U = np.zeros(person.S.shape, dtype=np.float32)
            Vn = bodies.V[i].cpu().numpy().astype(np.float32)
            person.V = np.zeros(person.S.shape, dtype=np.float32)
            for partId in range(Un.shape[0]):
                person.U[person.I == partId] = Un[partId][person.I == partId].clip(0, 1)
                person.V[person.I == partId] = Vn[partId][person.I == partId].clip(0, 1)

            people.append(person)

        return people

    def train(self, saveModel):
        raise Exception("DensePose algorithm cannot be trained from within DenseSense")

    def _rescaleImage(self, image):
        IS = image.shape
        aspect = image.shape[1] / image.shape[0]
        if self.MaxImageDim < IS[0] and IS[1] < IS[0]:
            dims = (self.MaxImageDim, int(self.MaxImageDim * aspect))
            dims = (dims[1], dims[0])
            image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
        elif self.MaxImageDim < IS[1] and IS[0] < IS[1]:
            dims = (int(self.MaxImageDim / aspect), self.MaxImageDim)
            dims = (dims[1], dims[0])
            image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
        return image

    def renderDebug(self, image, people):
        image = image.astype(np.uint8)

        for person in people:
            # Draw bounding box rectangle
            bnds = person.bounds.astype(np.uint32)
            image -= cv2.rectangle(np.zeros_like(image), (bnds[0], bnds[1]),
                                   (bnds[2], bnds[3]),
                                   (20, 20, 20), 1)

            # Get color of body parts using cv2 color map
            matrix = (person.I * (255 / 25)).astype(np.uint8)
            matrix = cv2.applyColorMap(matrix, cv2.COLORMAP_PARULA)

            # Generate mask
            mask3 = np.zeros(matrix.shape, dtype=np.uint8)
            mask3[person.S != 0] = 1

            # Apply UVs
            matrix = matrix.astype(np.int16)
            matrix[:, :, 0] += (person.U * 200 - 100).astype(np.int8)
            matrix[:, :, 1] += (person.V * 200 - 100).astype(np.int8)
            matrix = np.clip(matrix, 0, 255).astype(np.uint8)

            # Resize matrix and mask
            dims = (bnds[2] - bnds[0], bnds[3] - bnds[1])
            matrix = cv2.resize(matrix, dims, interpolation=cv2.INTER_AREA)
            mask3 = cv2.resize(mask3, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            alpha = 0.3
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            matrix = np.where(mask3, matrix * alpha + overlap * (1.0 - alpha), overlap)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = matrix

        return image
