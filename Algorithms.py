#!/usr/bin/env python2

import time
import json
from collections import deque
import heapq
import threading

import numpy as np

from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color

import cv2
import csv

from scipy.interpolate import griddata
from scipy.spatial import qhull
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from sklearn import cluster

# Import caffe2 and Detectron

from caffe2.python import workspace

import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
from detectron.core.config import (assert_and_infer_cfg, cfg, merge_cfg_from_file)
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Init densepose model
merge_cfg_from_file("/pose_detection_payload/DensePose_ResNet50_FPN_s1x-e2e.yaml")
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg("/shared/DensePose_ResNet50_FPN_s1x-e2e.pkl")


# Import PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("PyTorch running on "+str(device))

firstOpen = True
def writeTrainingData(x, y):
    global firstOpen
    if firstOpen:
        firstOpen = False
        open("/shared/trainingData.dat", "w").close()
    with open("/shared/trainingData.dat", "a") as f:
        f.write("\n"+str(x)+" "+str(y))


# Algorithm
class Algorithm(object): # Abstract class
    name = "base"
    outputFeatureType = ""

    debugOutput = None
    debugOutputShape = None

    def __init__(self, outputFeatureType):
        self.outputFeatureType = outputFeatureType

    def extract():
        pass # To be overridden

    def train():
        pass # To be overridden


class Densepose(Algorithm):
    name = "densepose"

    def __init__(self):
        Algorithm.__init__(self, "densepose")

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


class People_Extractor(Algorithm):
    name = "people"

    model = None
    modanetGenerator = None
    db = None
    epoch = 0

    processedImagesThisEpoch = 0
    vectors = [[[] for j in range(25)] for i in range(25)]
    bodyPartCounts = []
    assigments = np.empty(1)


    def __init__(self, model=None, db = None):
        Algorithm.__init__(self, "densepose")
        
        if model is not None:
            with open(model, "r") as modelFile:
                self.model = json.load(modelFile)
            
            mx = np.zeros((25, 25))
            for x in range(25):
                for y in range(25):
                    if self.model["distributionsX"][x][y] == []:
                        mx[x][y] = 0
                    else:
                        mx[x][y] = np.mean(self.model["distributionsX"][x][y])
            self.model["averagesX"] = np.array(mx)

            my = np.zeros((25, 25))
            for x in range(25):
                for y in range(25):
                    if self.model["distributionsY"][x][y] == []:
                        my[x][y] = 0
                    else:
                        my[x][y] = np.mean(self.model["distributionsY"][x][y])
            self.model["averagesY"] = np.array(my)

        self.modanetGenerator = self.createModanetGenerator()
        if db is not None:
            self.db = db

    def createModanetGenerator(self):
        self.db.getData("modanet", "")
        modanet = self.db.getAllLoaded()["modanet"]

        for key, annotations in modanet.iteritems():
            yield key, annotations
        
        # Reached end of dataset. Restart
        self.epoch += 1
        self.processedImagesThisEpoch = 0
        self.modanetGenerator = self.createModanetGenerator()
        yield self.modanetGenerator.next() # Passes on the torch as a last act, then dies. How tragic and beautiful :,)

    def extract(self, boxes, bodys, image, training=False):
        # Merge into one inds
        mergedIUVs = np.zeros((3, image.shape[0], image.shape[1]), dtype=np.float)
        if boxes.shape[0] == 0:
            return [], mergedIUVs

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(areas) 
        for i in sorted_inds:
            bbox = boxes[i, :4]
            IUVs = bodys[i]
            x1 = int(bbox[0])
            x2 = int(bbox[0] + IUVs.shape[2])
            y1 = int(bbox[1])
            y2 = int(bbox[1] + IUVs.shape[1])
            mergedIUVs[:, y1:y2, x1:x2] = IUVs
        
        mergedInds = mergedIUVs[0].astype(np.uint8)
        
        
        # Find the contours of all the body parts
        COMs = []
        shift = 2
        minArea = 8*8
        bodyparts = []
        indexRegion = [None]
        
        t1 = time.time()

        def range_overlap(a_min, a_max, b_min, b_max):
                return (a_min <= b_max) and (b_min <= a_max)

        for ind in range(1, 25):
            # Extract contours from this bodypart
            regionWhere = np.where(mergedInds == ind)
            if len(regionWhere[0]) == 0:
                indexRegion.append(None)
                continue
            
            region = np.zeros_like(mergedInds, np.uint8)
            region[regionWhere] = 255
            region = cv2.Canny(region, 100, 101)
            kernel = np.ones((5,5),np.uint8)
            
            iterations = 5
            dialted = cv2.dilate(region, kernel, iterations=iterations)

            indexRegion.append(dialted)

            contour, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            
            # Turn these to bounding boxes
            contourNP = np.asarray(contour)
            cb = np.zeros((len(contour), 4))
            for j in range(len(contour)):
                cb[j] = (
                    np.amin(contourNP[j][:, 0, 0]),
                    np.amin(contourNP[j][:, 0, 1]),
                    np.amax(contourNP[j][:, 0, 0]),
                    np.amax(contourNP[j][:, 0, 1])
                )

            cb = cb.tolist()
            
            # Merge overlaping bounds
            j = 0
            while j < len(contour):
                # Delete too small
                if (cb[j][2] - cb[j][0])*(cb[j][3] - cb[j][1]) < minArea:
                    del contour[j]
                    del cb[j]
                    if len(contour) == 0:
                        break
                    continue

                k = j+1
                while k < len(contour):
                    if k != j and range_overlap(cb[j][0], cb[j][2], cb[k][0], cb[k][2]) \
                                and range_overlap(cb[j][1], cb[j][3], cb[k][1], cb[k][3]):
                        # Boxes overlap, merge them!
                        contour[j] = np.append(contour[j], contour[k], axis=0) # FIXME: do this properly
                        del contour[k]
                        del cb[k]

                        c_np = np.asarray(contour[j])
                        cb[j] = (
                            np.amin(c_np[:, 0, 0]),
                            np.amin(c_np[:, 0, 1]),
                            np.amax(c_np[:, 0, 0]),
                            np.amax(c_np[:, 0, 1])
                        )
                        k = j+1
                    k += 1              
                j += 1
            contour = np.asarray(contour)
            for j in range(len(contour)):
                if len(contour[j]) != 0:
                    bodyparts.append((ind, cb[j]))


        # Function for merging parts (disjoint set)
        self.assigments = np.arange(-len(bodyparts), 0, dtype=np.int32)

        def find(i):
            if self.assigments[i] != i:
                self.assigments[i] = find(self.assigments[i])
            return self.assigments[i]


        # Find overlap between bodyparts
        
        overlaps = np.zeros((len(bodyparts), len(bodyparts)))
        for i in range(len(bodyparts)):
            for j in range(i+1, len(bodyparts)):
                ind_i, b_i = bodyparts[i]
                ind_j, b_j = bodyparts[j]
                if range_overlap(b_j[0], b_j[2], b_i[0], b_i[2]) \
                    and range_overlap(b_j[1], b_j[3], b_i[1], b_i[3]):
                        xMin = int(min(b_j[0], b_i[0]))
                        yMin = int(min(b_j[1], b_i[1]))
                        xMax = int(max(b_j[2], b_i[2]))
                        yMax = int(min(b_j[3], b_i[3]))

                        r_i = np.asarray(indexRegion[ind_i][yMin:yMax, xMin:xMax], np.bool)
                        r_j = np.asarray(indexRegion[ind_j][yMin:yMax, xMin:xMax], np.bool)
                        
                        overlap = np.sum(np.bitwise_and(r_i, r_j))
                        if overlap > 2:
                            if self.assigments[i] < 0:
                                self.assigments[i] = i
                            iRoot = find(self.assigments[i])
                            jRoot = find(self.assigments[j])
                            if iRoot != jRoot:
                                self.assigments[jRoot] = iRoot

        
        for i in range(len(self.assigments)):
            find(i) # Loop through one last time to make sure the tree is flat (so not a tree?)

        if training:
            self.bodyPartCounts.append(partCounts)

        t2 = time.time()

        # Save data
        people = []
        #self.debugOutput = {}
        j = 0
        bodyparts = np.asarray(bodyparts, object)

        for i in np.unique(self.assigments):
            parts = bodyparts[self.assigments == i]
            partBounds = parts[:, 1]
            savedParts = {}
            for p in parts:
                savedParts[int(p[0])] = np.asarray(p[1][0], np.float)

            xMin = 1000000
            yMin = 1000000
            xMax = 0
            yMax = 0
            for pb in partBounds:
                pb = np.array(pb)
                xMin = min(xMin, pb[0])
                yMin = min(yMin, pb[1])
                xMax = max(xMax, pb[2])
                yMax = max(yMax, pb[3])

            
            person = {
                "id" : j,
                "index" : j,
                "bounds": np.array([
                    xMin, yMin,
                    xMax, yMax
                ], dtype=np.float),
                "bodyparts": savedParts
            }

            people.append(person)
            #self.debugOutput[j] = centers[self.assigments == i]
            j += 1

        #print(people)
        return people, mergedIUVs

    def train(self, saveModel):
        return


class People_Tracker(Algorithm):
    name = "tracker"

    trackedObjects = []
    seenPeople = 0
    frame = 0
    lastFrameTime = 0
    def __init__(self):
        Algorithm.__init__(self, "tracker")
        self.lastFrameTime = time.time()
    
    def extract(self, people):
        self.frame += 1
        nowTime = time.time()
        dt = self.lastFrameTime - nowTime
        self.lastFrameTime = nowTime

        distThreshold = 500

        delete = 0.3
        hide = 0.5
        minFrames = 3
        maxPersistanceBufferSize = 15


        # Find optimal tracker <-> person configuration
        status = [-1 for _ in xrange(len(people))]
        associations = [(-1, 10000) for _ in xrange(len(self.trackedObjects))]

        def match(person):
            # Convert to center and dim coords
            bounds = person["bounds"]
            dims = np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]])
            center = np.array([bounds[0]+dims[0]/2, bounds[1]+dims[1]/2])
            
            # Find the closest viable tracked object
            displacedPerson = None
            closest = 1000000
            cInd = -1
            for i in xrange(len(self.trackedObjects)):
                predictedCenter = np.squeeze(self.trackedObjects[i][11][:2])
                dist = np.linalg.norm(predictedCenter-center)
                sizeDiff = np.linalg.norm(dims-np.squeeze(self.trackedObjects[i][11][4:]))
                #dist += sizeDiff/5
                if dist < closest:
                    # Is this the closest not reserved?
                    if associations[i][0] == -1 or dist < associations[i][1]:
                        if associations[i][0] != -1:
                            displacedPerson = associations[i][0]
                        else:
                            displacedPerson = None
                        closest = dist
                        cInd = i

            # If this was close enough to be this object
            if closest < distThreshold:
                return cInd, closest, displacedPerson
            return None, None, None

        
        i = 0
        while i < len(people):
            if status[i] != -1:
                i += 1
                continue
            person = people[i]
            ind, dist, displaced = match( person)
            if ind is not None:
                associations[ind] = (i, dist)
                status[i] = ind
                if displaced is not None:
                    # A person (guaranteed lower index) has been displaced. Reconsider this person
                    status[displaced] = -1
                    i = displaced
                    continue
            i += 1

        
        for i in range(len(people)):
            person = people[i]
            bounds = person["bounds"]
            dims = np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]])
            center = np.array([bounds[0]+dims[0]/2, bounds[1]+dims[1]/2])

            if status[i] != -1:
                cInd = status[i]
                person["id"] = self.trackedObjects[cInd][0]
                oldPosition = self.trackedObjects[cInd][1]
                self.trackedObjects[cInd][1] = center 
                self.trackedObjects[cInd][4] = person["index"]
                self.trackedObjects[cInd][5] = self.frame
                self.trackedObjects[cInd][6] = person
                self.trackedObjects[cInd][7].append(1)
                self.trackedObjects[cInd][9] = nowTime

                self.trackedObjects[cInd][10].transitionMatrix[0, 2] = dt*0.4
                self.trackedObjects[cInd][10].transitionMatrix[1, 3] = dt*0.4
                self.trackedObjects[cInd][10].correct(np.concatenate([center, dims]))
                self.trackedObjects[cInd][11] = self.trackedObjects[cInd][10].predict()
                
            else:
                # This must be a new object
                person["id"] = self.seenPeople
                t = cv2.CV_64F 
                s = 6 # x, y, vx, vy, wx, w, h
                m = 4 # x, y, w, h 
                kalman = cv2.KalmanFilter(s, m, 0, type=t)

                kalman.transitionMatrix= np.eye(s)
                kalman.measurementMatrix = np.zeros((m, s))
                kalman.measurementMatrix[0, 0] = 1
                kalman.measurementMatrix[1, 1] = 1
                kalman.measurementMatrix[2, 4] = 1
                kalman.measurementMatrix[3, 5] = 1

                kalman.processNoiseCov = 1e-2 * np.eye(s)
                kalman.processNoiseCov[2, 2] = 5
                kalman.processNoiseCov[3, 3] = 5

                kalman.measurementNoiseCov = 1e-1 * np.eye(m)

                kalman.statePre = np.array([center[0], center[1], 0, 0, dims[0], dims[1]], np.float)
                

                self.trackedObjects.append([ # TODO: use something different than a list for this
                    self.seenPeople, # 0 ID
                    center,          # 1 Center
                    0,               # 2 Times seen
                    self.frame,      # 3 First seen
                    person["index"], # 4 Raw index
                    self.frame,      # 5 Last seen
                    person,          # 6 Person object
                    deque([1]),      # 7 Persistance buffer
                    False,           # 8 Is visible
                    nowTime,         # 9 Last seen time
                    kalman,          # 10 Velocity kalman-filter
                    kalman.statePre  # 11 Prediction new frame
                ])

                self.seenPeople += 1

        toRemove = []
        i = 0
        while i < len(self.trackedObjects):
            deltaLastSeen = nowTime - self.trackedObjects[i][9]
            if i < len(associations):
                if associations[i][0] == -1:
                    self.trackedObjects[i][7].append(0)

            if maxPersistanceBufferSize < len(self.trackedObjects[i][7]):
                self.trackedObjects[i][7].popleft()
            persistance = float(sum(self.trackedObjects[i][7]))/len(self.trackedObjects[i][7])

            # If should be removed and is seen
            if hide < persistance and minFrames < len(self.trackedObjects[i][7]):
                self.trackedObjects[i][8] = True
            else:
                self.trackedObjects[i][8] = False
                if self.trackedObjects[i][5] == self.frame:
                    toRemove.append(self.trackedObjects[i][4]) 
            
            if persistance < delete:
                del self.trackedObjects[i]
                continue
            
            # If shouldn't be removed and not seen
            if self.trackedObjects[i][5] != self.frame and self.trackedObjects[i][8]:
                self.trackedObjects[i][6]["index"] = len(people)
                people.append(self.trackedObjects[i][6])
            
            i += 1
        
        for index in sorted(toRemove, reverse=True):
            del people[index]
        
        ghostBounds = []
        keptAliveIndex = []
        predictions = []
        for to in self.trackedObjects:
            predictions.append((to[0], np.squeeze(to[11][:2]), to[8]))
            if not to[8]:
                ghostBounds.append(to[6]["bounds"])
            elif to[8] and to[5] != self.frame:
                keptAliveIndex.append(to[6]["index"])
        
        print("Tracking: "+str(len(self.trackedObjects))+" ("+str(len(ghostBounds))+" ghosts)")
        
        self.debugOutput = (
            np.array(ghostBounds),
            keptAliveIndex,
            predictions
        )


class UV_Extractor(Algorithm):
    name = "uv"
    def __init__(self, db = None):
        Algorithm.__init__(self, "densepose")
    
    def extract(self, people, mergedIUVs, image, threshold=100):
        resolution = 64
        I, U, V = mergedIUVs[0].astype(np.uint8), mergedIUVs[1], mergedIUVs[2]

        peopleTexture = []
        for person in people:
            bbox = person["bounds"]
            area = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0])
            if area < threshold:
                peopleTexture.append(None)
                continue

            personTexture = np.zeros((25, resolution, resolution, 3), dtype=np.uint8)
            
            for partIndex in xrange(1, 25):
                box = np.asarray(person["bounds"], np.int32)

                x,y = np.where(I[box[1]:box[3], box[0]:box[2]] == partIndex)
                if x.size < 4: # Need at least 4 pixels geo interpolate
                    continue # Did not find this bodypa ge
                
                u = U[box[1]:box[3], box[0]:box[2]][x,y]
                v = V[box[1]:box[3], box[0]:box[2]][x,y]

                # Add box global location
                x += np.floor(box[1]).astype(np.int64) # CHANGE
                y += np.floor(box[0]).astype(np.int64) # CHANGE
                
                pixels = image[x,y]
                
                gx, gy = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]

                # Interpolate values. This can be a bit slow...
                try:
                    # TODO: use cuda accelerated interpolation instead
                    texture = griddata((u, v), pixels, (gx, gy), # Nearest looks weird, but is the consistently much fastest
                        method="nearest", fill_value=0).astype(np.uint8)
                except qhull.QhullError as e:
                    continue
 
                personTexture[partIndex] = texture

            # Put all textures in one square image
            personTexture = np.split(personTexture, 5)
            personTexture = np.concatenate(personTexture, axis=2)
            personTexture = np.concatenate(personTexture, axis=0)

            peopleTexture.append(personTexture)
        
        return np.array(peopleTexture)
    
    def train(self, saveModel):
        raise Exception("UV extraction algorithm cannot be trained")


class DescriptionExtractor(Algorithm):
    name = "description"
    iteration = 0

    availableLabels = {
        3 : "boots",
        4 : "footwear",
        5 : "outer",
        6 : "dress",
        7 : "sunglasses",
        8 : "pants",
        9 : "top",
        10 : "shorts",
        11 : "skirt",
        12 : "headwear",
        13 : "scarfAndTie"
    }

    labelBodyparts = { # https://github.com/facebookresearch/DensePose/issues/64#issuecomment-405608749 PRAISE 
        "boots" : [5, 6],
        "footwear": [5, 6],
        "outer" : [1, 2, 15, 17, 16, 18, 19, 21, 20, 22],
        "dress" : [1, 2],
        "sunglasses" : [],
        "pants" : [7, 9, 8, 10, 11, 13, 12, 14],
        "top" : [1, 2],
        "shorts" : [7, 9, 8, 10],
        "skirt" : [1, 2],
        "headwear" : [23, 24],
        "scarfAndTie" : []
    }

    colors = [
        ((255, 255, 255),  "white"),
        ((210, 209, 218),  "white"),
        ((145, 164, 164),  "white"),
        ((169, 144, 135),  "white"),
        ((197, 175, 177),  "white"),
        ((117, 126, 115),  "white"),
        ((124, 126, 129),  "white"),
        ((0, 0, 0),        "black"),
        ((10, 10, 10),     "black"),
        ((1, 6, 9),        "black"),
        ((5, 10, 6),       "black"),
        ((18, 15, 11),     "black"),
        ((18, 22, 9),      "black"),
        ((16, 16, 14),     "black"),
        ((153, 153, 0),    "yellow"),
        ((144, 115, 99),   "pink"),
        ((207, 185, 174),  "pink"),
        ((206, 191, 131),  "pink"),
        ((208, 179, 54),   "pink"),
        ((202, 19, 43),    "red"),
        ((206, 28, 50),    "red"),
        ((82, 30, 26),     "red"),
        ((156, 47, 35),    "orange"),
        ((126, 78, 47),    "wine red"),
        ((74, 72, 77),     "green"),
        ((31, 38, 38),     "green"),
        ((40, 52, 79),     "green"),
        ((100, 82, 116),   "green"),
        ((8, 17, 55),      "green"),
        ((29, 31, 37),     "dark green"),
        ((46, 46, 36),     "blue"),
        ((29, 78, 60),     "blue"),
        ((74, 97, 85),     "blue"),
        ((60, 68, 67),     "blue"),
        ((181, 195, 232),  "neon blue"),
        ((40, 148, 184),   "bright blue"),
        ((210, 40, 69),    "orange"),
        ((66, 61, 52),     "gray"),
        ((154, 120, 147),  "gray"),
        ((124, 100, 86),   "gray"),
        ((46, 55, 46),     "gray"),
        ((119, 117, 122),  "gray"),
        ((88, 62, 62),     "brown"),
        ((60, 29, 17),     "brown"),
        ((153, 50, 204),   "purple"),
        ((77, 69, 30),     "purple"),
        ((153, 91, 14),    "violet"),
        ((207, 185, 151),  "beige")        
    ]

    colorsHSV = None

    net = None
    criterion = None
    optimizer = None
    modelFile = None

    noActivation = 0.2
    onActivation = 0.8

    debugOutput = 0
    lossAvg = 0

    db = None
    modanetGenerator = None
    epoch = 0
    processedImagesThisEpoch = 0

    class Network(nn.Module):
        def __init__(self, labels): # FIXME: make this work!
            super(DescriptionExtractor.Network, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
            )
            """
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
            """
            self.fc1 = nn.Linear(75843, 750) 
            self.relu1 = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(750, labels)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            out = self.layer1(x)
            #out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out

    def __init__(self, model=None, db = None):
        print("Initiating DescriptionExtractor")
        Algorithm.__init__(self, "dictDescription")

        self.modelFile = model
        # Init classifier
        self.net = DescriptionExtractor.Network(len(self.availableLabels))
        if self.modelFile is not None:
            print("Loading people description file from: "+self.modelFile)
            self.net.load_state_dict(torch.load(model))
        self.net.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0003)

        # Init color lookup KD-tree
        self.colorsHSV = []
        for c in self.colors:
            RGBobj = sRGBColor(c[0][0], c[0][1], c[0][2])
            self.colorsHSV.append(convert_color(RGBobj, HSVColor))

        # Set database and generator for pulling 
        if db is not None:
            self.db = db
            self.modanetGenerator = self.createModanetGenerator()

    def extract(self, peopleTexture, training = False):
        labelsPeople = []
        labelsPeopleVector = [] # For training
        first = True

        # Do label classification
        for personTexture in peopleTexture: # TODO: batch run all people?
            if personTexture is None:
                labelsPeople.append(None)
                labelsPeopleVector.append(None)
                continue
            
            # Run the classification on it
            pyTorchTexture = torch.from_numpy(np.array([np.moveaxis(personTexture.astype(float)/255.0, -1, 0)])).float()
            if training: # Make this a variable if it isn't training
                pyTorchTexture = torch.autograd.Variable(pyTorchTexture)
        
            pyTorchTexture = pyTorchTexture.to(device)
            labelVector = self.net(pyTorchTexture)[0]

            def findColorName(areas):
                areaS = int(personTexture.shape[0]/5)
                Rs, Gs, Bs= [], [], []

                # Pick out colors
                for i in areas:
                    xMin = int((i%5)*areaS)
                    yMin = int(np.floor(i/5)*areaS)
                    for j in range(20):
                        x = np.random.randint(xMin, xMin + areaS)
                        y = np.random.randint(yMin, yMin + areaS)
                        b = personTexture[x, y, 0]
                        g = personTexture[x, y, 1]
                        r = personTexture[x, y, 2]
                        
                        if r != 0 or b != 0 or g != 0:
                            Rs.append(r)
                            Gs.append(g)
                            Bs.append(b)
                
                if len(Rs) + len(Gs) + len(Bs) < 3:
                    return 0
                
                # Find mean color
                r = np.mean(np.array(Rs)).astype(np.uint8)
                g = np.mean(np.array(Gs)).astype(np.uint8)
                b = np.mean(np.array(Bs)).astype(np.uint8)

                # This prints the color colored in the terminal
                RESET = '\033[0m'
                def get_color_escape(r, g, b, background=False):
                    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)
                colorRepr = get_color_escape(r, b, g)+"rgb("+str(r)+", "+str(g)+", "+str(b)+")"+RESET

                # Get nearest color name
                HSVobj = convert_color(sRGBColor(r, g, b), HSVColor)

                nearestIndex = -1
                diffMin = 100000
                for i in xrange(len(self.colorsHSV)):
                    colEntry = self.colorsHSV[i]

                    d = HSVobj.hsv_h - colEntry.hsv_h
                    dh = min(abs(d), 360-abs(d)) / 180.0
                    ds = abs(HSVobj.hsv_s - colEntry.hsv_s)
                    dv = abs(HSVobj.hsv_v - colEntry.hsv_v) / 255.0
                    diff = np.sqrt(dh*dh + ds*ds + dv*dv)
                    if diff < diffMin:
                        diffMin = diff
                        nearestIndex = i

                return { "color":self.colors[nearestIndex][1], "colorDistance":diffMin, "coloredStr":colorRepr }
            
            # Store the data
            if not training:
                labelVectorHost = labelVector.detach().cpu().numpy()
                labels = {}
                for j in range(len(labelVector)):
                    label = self.availableLabels.values()[j]
                    d = (self.onActivation - self.noActivation)/2
                    val = (labelVectorHost[j] - d) / d + 0.5

                    info = { "activation" : min(max(val, 0.0), 1.0) }
                    if 0.7 < val:
                        color = findColorName(self.labelBodyparts[label])
                        if color != 0:
                            info.update(color)
                            #print(color["color"]+"  "+color["coloredStr"])
                    labels[label] = info
                
                labelsPeople.append(labels)
            labelsPeopleVector.append(labelVector)

        if training:
            return labelsPeopleVector
        torch.cuda.empty_cache()        
        return labelsPeople

    def createModanetGenerator(self):
        self.db.getData("modanet", "")
        modanet = self.db.getAllLoaded()["modanet"]

        for key, annotations in modanet.iteritems():
            yield key, annotations
        
        # Reached end of dataset. Restart
        self.epoch += 1
        self.processedImagesThisEpoch = 0
        self.modanetGenerator = self.createModanetGenerator()
        yield self.modanetGenerator.next() # Passes on the torch as a last act, then dies. How tragic and beautiful :,)

    def train(self, saveModel):
        self.iteration += 1

        # Load annotations
        imageID, annotations = self.modanetGenerator.next()

        # Convert annotation labels to vector
        labels = np.full(len(self.availableLabels), self.noActivation)
        
        
        for a in annotations:
            if a["category_id"] in self.availableLabels:
                labelIndex = self.availableLabels.keys().index(a["category_id"])
                labels[labelIndex] = self.onActivation
                
        labels = torch.autograd.Variable(torch.from_numpy(labels)).float().to(device)

        # Get UV texture
        UV_Textures = self.db.getData("UV_Textures", imageID)  

        if len(UV_Textures) == 0:
            self.processedImagesThisEpoch += 1
            return self.train(saveModel) # No person in this image (rare)

        self.debugOutput = UV_Textures[0]

        self.optimizer.zero_grad()

        startTime = time.time()
        output = self.extract(UV_Textures, True) # Got texture from drive
        endTime = time.time()
        output = output[0] # Only 1 person per picture in modanet-dataset
        
        loss_size = self.criterion(output, labels)
        loss_size.backward()
        self.optimizer.step()
        loss_size_detached = loss_size.item()

        self.lossAvg += loss_size_detached
        if self.iteration%6000 == 0:
            writeTrainingData(self.lossAvg/6000.0, loss_size_detached)

        self.processedImagesThisEpoch += 1
        torch.cuda.empty_cache()
        print("TARGET", labels)
        print("GOT   ", output)
        #print("ALLOCATED CUDA ", torch.cuda.memory_allocated(device))
        if saveModel:
            torch.save(self.net.state_dict(), "/shared/trainedDescription.model") # TODO: have this be the self.modelfile instead
        return (self.iteration-1, loss_size_detached, endTime - startTime)


class Action_Classifier(Algorithm):
    name = "actions"

    actions = {
        4  : "dance",
        11 : "sit",
        14 : "walk",
        69 : "hand wave",
        
        12 : "idle", # stand
        17 : "idle", # carry/hold (an object)
        36 : "idle", # lift/pick up
        37 : "idle", # listen
        47 : "idle", # put down
    }

    avaFiltered = {}
    classCursors = {}

    iteration = 0

    db = None
    avaGenerator = None
    epoch = 0
    processedImagesThisEpoch = 0

    def __init__(self, db=None):
        name = "actions"

    class Network(nn.Module):
        def __init__(self, outputs):
            super(Action_Classifier.Network, self).__init__()

            self.lstm = nn.LSTM(25*2, 10)
            # TODO: activation function?
            self.linear = nn.Linear(10, outputs)

        def forward(self, features):
            out = self.lstm(features)
            out = self.linear(out)
            return out


    actions = {
        4  : "dance",
        11 : "sit",
        14 : "walk",
        69 : "hand wave",
        
        12 : "idle", # stand
        17 : "idle", # carry/hold (an object)
        36 : "idle", # lift/pick up
        37 : "idle", # listen
        47 : "idle", # put down
    }

    net = None
    loss_function = None
    optimizer = None
    currentPeople = {}

    classCursors = None
    upcoming = []
    videoBuffer = {}
    lastVideo = None
    recent = deque()

    db = None
    epoch = 0
    iteration = 0

    def __init__(self, db=None):
        Algorithm.__init__(self, "tracker")
        actionIDs = self.actions.keys()
        classCount = len(set(self.actions.values()))

        self.net = Action_Classifier.Network(classCount)
        self.loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)

        if db is not None:
            self.db = db

            # Extract Ava
            db.getData("ava", json.dumps(actionIDs))
            self.ava = db.getAllLoaded()["ava"]

            # Initiate cursors
            self.classCursors = dict(self.actions)
            for key in self.actions.keys():
                self.classCursors[key] = [0, 0]
    

    def extract(self, people, training = False):
        # Person: {"bodyparts":[]}
        labelsPeople = []
        labelsPeopleVector = []

        seenPeople = set()

        # Check if new person, if so, add another instance of Network
        for i in range(len(people)):
            person = people[i]
            seenPeople.append(person["id"])
            if person is None:
                labelsPeople.append(None)
                labelsPeopleVector.append(None)
                continue
            
            #if person["id"] not in self. 

        for key in self.currentPeople:
            if key not in seenPeople:
                del self.seenPeople[key]

        # Run each network for each person
        if training:
            return labelsPeopleVector
        torch.cuda.empty_cache()        
        return labelsPeople

    def download_video(self, key, starttime, endtime):
        print("Downloading", key, starttime, endtime)
        time.sleep(0.2)
        print(self.videoBuffer[key][1] == None)
        video = self.db.getData("youtube", str(key)+"|"+str(starttime)+"|"+str(endtime))
        self.videoBuffer[key][0].set()
        self.videoBuffer[key][1] = video
        print("Finished downloading", key)

    def getNextAndBuffer(self, buffer):
        print("Video buffer keys", self.videoBuffer.keys())
        print("Upcoming", self.upcoming)
        s = ""
        for i in self.upcoming:
            s += self.ava[i[0]]["key"]+", "
        print("Upcoming keys: "+s)
        if self.lastVideo is not None:
            lastKey = self.ava[self.lastVideo[0]]["key"]
            for i, j in self.upcoming:
                if lastKey == self.ava[i]["key"]:
                    break
            else:
                del self.videoBuffer[lastKey]
                print("------------- UNLOAD VIDEO -- "+lastKey+" --- "+str(len(self.videoBuffer))+" videos left ---")

        # What classes should be choosen
        whatClasses = np.random.choice(self.actions.keys(), buffer - len(self.upcoming))
        
        # Move each cursors of chosen classes to next positions,
        # and queue all video downloads
        for c in whatClasses:
            cursor = self.classCursors[c]
            key = self.ava[cursor[0]]["key"]
            starttime = self.ava[cursor[0]]["startTime"]
            endtime = self.ava[cursor[0]]["endTime"]
            self.upcoming.append(cursor[:])
            if key not in self.videoBuffer:
                self.videoBuffer[key] = [threading.Event(), None]
                threading.Thread(target=self.download_video, args=(key, starttime, endtime)).start()
            
            while True:
                cursor[1] += 1
                if len(self.ava[cursor[0]]["people"]) == cursor[1]:
                    cursor[0] = (cursor[0]+1)%len(self.ava)
                    cursor[1] = 0
                person = self.ava[cursor[0]]["people"][cursor[1]]
                if str(c) in person.keys():
                    self.classCursors[c] = cursor
                    break

        # Unload video if it's not appearing again for a while
        current = self.upcoming[0]
        self.upcoming = self.upcoming[1:]

        currentKey = self.ava[current[0]]["key"]
        if self.videoBuffer[currentKey][1] is None:
            self.videoBuffer[currentKey][0].wait() # Wait for video to have been downloaded
        
        if self.videoBuffer[currentKey][1] is False:
            print("Video "+currentKey+" has been removed. Skipping...")
            self.lastVideo = None
            del self.videoBuffer[currentKey] # This video has been removed from YT
            i = 0
            while i < len(self.upcoming):
                ind = self.upcoming[i][0]
                if currentKey == self.ava[i]["key"]:
                    del self.upcoming[ind]
                else:
                    i += 1

            return self.getNextAndBuffer(buffer)
        
        if tuple(current) in self.recent:
            print("Cursor "+str(current)+" was recently processed. Skipping...")
            self.lastVideo = None
            return self.getNextAndBuffer(buffer)
        
        self.recent.append(tuple(current))
        if 15 < len(self.recent):
            self.recent.popleft()
        self.lastVideo = current
        return current, currentKey

    def train(self, saveModel, algorithms):
        self.iteration += 1
        current, key = self.getNextAndBuffer(2)
        video = self.videoBuffer[key]
        person = self.ava[current[0]]["people"][current[1]]
        annotatedFrames = person.items()
        annotatedFrames.sort(key=lambda x: int(x[0]))
        annotatedFrames = filter(lambda x: x[1] != [], annotatedFrames)

        print("Current is", key)
        print("Cursor is", current)

        # TODO: grab the right frames from the video
        for annFrame in annotatedFrames:
            print("Ann frame", annFrame)
            print(video)
            print(len(video))
            for frame in video[int(annFrame[0]):int(annFrame[0])+10]: # Assuming 10 fps
                print("FRAME IN VIDEO!")
                # Extract the bounding box of the image
                lower = frame["bbox"][:2]*frame.shape - np.array([50.0, 50.0])
                upper = frame["bbox"][2:]*frame.shape + np.array([50.0, 50.0])
                print(lower)
                print(upper)
                lower = lower.clip(min=0)
                upper[0] = upper[0].clip(max=frame.shape[1])
                upper[1] = upper[1].clip(max=frame.shape[0])
                frame = frame[int(lower[1]):int(upper[1]), int(lower[0]):int(upper[0])]

                boxes, bodys = algorithms["DP"].extract(frame)
                people, mergedIUVs = algorithms["DE"].extract(boxes, bodys, frame)
                # TODO: Find the biggest one
                person = []
                
                # TODO: give person new id if more than 1 seconds have passed


                # Reformat labels

                # Feed into network
                labels = self.extract(person, True)

                # Train

            
        
        # Process video
        print("PRETEND PROCESSING "+str(key))
        t1 = time.time()
        time.sleep(1)
        t2 = time.time()
        print("DONE PRETENDING")
        """

        # Match how data is passed in during running extraction
        people = []
        lastTimestamp = None
        for frame in frames:
            if lastTimestamp is not None:
                if int(frame[0]) - lastTimestamp > 3: # if more than 3 seconds have passed
                    people
            lastTimestamp = int(frame[0])

        startTime = time.time()
        output = self.extract(people) # Got texture from drive
        endTime = time.time()

        loss_size = self.criterion(output, labels)
        loss_size.backward()
        self.optimizer.step()
        loss_size_detached = loss_size.item()
        """

        return self.iteration, 0.1, (t2-t1) # Loss, time

