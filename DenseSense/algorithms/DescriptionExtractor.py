import time

import DenseSense.algorithms.Algorithm
from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
from DenseSense.algorithms.Sanitizer import Sanitizer
from DenseSense.algorithms.UVMapper import UVMapper
from DenseSense.utils.LMDBHelper import LMDBHelper

from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, HSVColor
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("DescriptionExtractor running on: " + str(device))


class DescriptionExtractor(DenseSense.algorithms.Algorithm.Algorithm):
    iteration = 0

    availableLabels = {
        0: "none",
        1: "short sleeve top",
        2: "long sleeve top",
        3: "short sleeve outwear",
        4: "long sleeve outwear",
        5: "vest",
        6: "sling",
        7: "shorts",
        8: "trousers",
        9: "skirt",
        10: "short sleeve dress",
        11: "long sleeve dress",
        12: "dress vest",
        13: "sling dress"
    }

    #  0 : none
    #  1 : trousers
    #  2 : R hand
    #  3 : L hand
    #  4 : R foot
    #  5 : L foot
    #  6 : R thigh
    #  7 : L thigh
    #  8 : R calf
    #  9 : L calf
    # 10 : L upper arm
    # 11 : R upper arm
    # 12 : L lower arm
    # 13 : R lower arm
    # 14 : head

    labelColorCheck = {
        0: [],
        1: [1, 10, 11],
        2: [1, 10, 11, 12, 13],
        3: [1, 10, 11],
        4: [1, 10, 11, 12, 13],
        5: [1, 10, 11],
        6: [1, 10, 11],
        7: [6, 7],
        8: [6, 7, 8, 9],
        9: [6, 7],
        10: [1, 10, 11],
        11: [1, 10, 11, 12, 13],
        12: [1, 10, 11],
        13: [1, 10, 11]
    }

    colors = [
        ((255, 255, 255), "white"),
        ((210, 209, 218), "white"),
        ((145, 164, 164), "white"),
        ((169, 144, 135), "white"),
        ((197, 175, 177), "white"),
        ((117, 126, 115), "white"),
        ((124, 126, 129), "white"),
        ((0, 0, 0), "black"),
        ((10, 10, 10), "black"),
        ((1, 6, 9), "black"),
        ((5, 10, 6), "black"),
        ((18, 15, 11), "black"),
        ((18, 22, 9), "black"),
        ((16, 16, 14), "black"),
        ((153, 153, 0), "yellow"),
        ((144, 115, 99), "pink"),
        ((207, 185, 174), "pink"),
        ((206, 191, 131), "pink"),
        ((208, 179, 54), "pink"),
        ((202, 19, 43), "red"),
        ((206, 28, 50), "red"),
        ((82, 30, 26), "red"),
        ((156, 47, 35), "orange"),
        ((126, 78, 47), "wine red"),
        ((74, 72, 77), "green"),
        ((31, 38, 38), "green"),
        ((40, 52, 79), "green"),
        ((100, 82, 116), "green"),
        ((8, 17, 55), "green"),
        ((29, 31, 37), "dark green"),
        ((46, 46, 36), "blue"),
        ((29, 78, 60), "blue"),
        ((74, 97, 85), "blue"),
        ((60, 68, 67), "blue"),
        ((181, 195, 232), "neon blue"),
        ((40, 148, 184), "bright blue"),
        ((210, 40, 69), "orange"),
        ((66, 61, 52), "gray"),
        ((154, 120, 147), "gray"),
        ((124, 100, 86), "gray"),
        ((46, 55, 46), "gray"),
        ((119, 117, 122), "gray"),
        ((88, 62, 62), "brown"),
        ((60, 29, 17), "brown"),
        ((153, 50, 204), "purple"),
        ((77, 69, 30), "purple"),
        ((153, 91, 14), "violet"),
        ((207, 185, 151), "beige")
    ]

    colorsHSV = None

    class Network(nn.Module):
        def __init__(self, labels):  # FIXME: make this work!
            super(DescriptionExtractor.Network, self).__init__()
            self.layer1 = nn.Sequential( # Fixme: 3x15 in channels
                nn.Conv2d(in_channels=3*15, out_channels=15, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=15, out_channels=10, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
            )

            self.fc1 = nn.Linear(360, 180)
            self.relu1 = nn.ReLU(inplace=False)
            self.fc2 = nn.Linear(180, labels)

        def forward(self, x):
            batchSize = x.shape[0]
            x = x.view(batchSize, 15*3, 32, 32)
            x = self.layer1(x)
            x = self.layer2(x)
            x = x.view(batchSize, -1)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x

    def __init__(self, model=None, db=None):
        print("Initiating DescriptionExtractor")
        super().__init__()

        self.classifier = DescriptionExtractor.Network(len(self.availableLabels))
        self.modelPath = None
        self._training = False
        self.predictions = []
        self.peopleLabels = []

        # Init color lookup KD-tree
        self.colorsHSV = []
        for c in self.colors:
            RGBobj = sRGBColor(c[0][0], c[0][1], c[0][2])
            self.colorsHSV.append(convert_color(RGBobj, HSVColor))

    def loadModel(self, modelPath):
        self.modelPath = modelPath
        print("Loading DescriptionExtractor file from: " + self.modelPath)
        self.classifier.load_state_dict(torch.load(self.modelPath, map_location=device))
        self.classifier.to(device)

    def saveModel(self, modelPath):
        if modelPath is None:
            print("Don't know where to save model")
        self.modelPath = modelPath
        print("Saving DescriptionExtractor model to: "+self.modelPath)
        torch.save(self.classifier.state_dict(), self.modelPath)

    def _initTraining(self, learningRate, dataset, useDatabase):
        # Dataset is DeepFashion2
        print("Initiating training of DescriptionExtractor")
        print("Loading DeepFashion2")
        from torchvision import transforms
        from torchvision.datasets import CocoDetection

        self.annFile = './annotations/deepfashion2_{}.json'.format(dataset)
        self.cocoImgPath = './data/DeepFashion2/{}'.format(dataset)

        self.useDatabase = useDatabase
        self.dataset = CocoDetection(self.cocoImgPath, self.annFile,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: x.permute(1, 2, 0)),
                                         transforms.Lambda(lambda x: (x * 255).byte().numpy()),
                                         transforms.Lambda(lambda x: x[:, :, ::-1])
                                     ]))

        # Init LMDB_helper
        if useDatabase:
            self.lmdb = LMDBHelper("a")
            self.lmdb.verbose = False

        self.denseposeExtractor = DensePoseWrapper()
        self.sanitizer = Sanitizer()
        self.sanitizer.loadModel("./models/Sanitizer.pth")
        self.uvMapper = UVMapper()

        # PyTorch things
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learningRate, amsgrad=True)
        self.lossFunction = torch.nn.BCEWithLogitsLoss()

    def extract(self, peopleMaps):
        if len(peopleMaps) == 0:
            return []
        self.peopleLabels = []
        determineColorThreshold = 0.3  # FIXME: tune

        # Do label classification
        peopleMapsDevice = torch.Tensor(peopleMaps).to(device)
        self.predictions = self.classifier.forward(peopleMapsDevice)
        self.predictions = self.predictions.sigmoid()
        self.predictions = self.predictions.detach().cpu().numpy()

        # Compile predictions into nice dictionary
        for personIndex, prediction in enumerate(self.predictions):
            labels = {}
            # Some labels might use same areas for determining color
            # This is therefore a lookup table in case value has already been computed
            averages = np.full((peopleMaps.shape[1], 3), -1, dtype=np.int64)
            for i, value in enumerate(prediction):
                if i == 0:  # 0 is None, and not trained on anyways
                    continue
                label = self.availableLabels[i]

                info = {"activation": value}
                if determineColorThreshold < value:
                    # If certainty is above threshold, take the time to calculate the average color
                    averageOfAreas = np.zeros(3, dtype=np.int64)
                    relevantAreas = torch.as_tensor(self.labelColorCheck[i], dtype=torch.int64).to(device)
                    nonBlackAreas = 0
                    for areaIndex in relevantAreas:
                        if (averages[areaIndex] == -1).all():
                            # Calculate average
                            relevantPixels = peopleMapsDevice[personIndex, areaIndex, :, :]
                            relevantPixels = relevantPixels[torch.sum(relevantPixels, axis=2) != 0]
                            if relevantPixels.shape[0] == 0:
                                # All black
                                averages[areaIndex] = np.zeros(3)
                                continue
                            average = relevantPixels.mean(axis=0).cpu().numpy().astype(np.uint8)
                            averages[areaIndex] = average

                        nonBlackAreas += 1
                        averageOfAreas += averages[areaIndex]

                    averageOfAreas = (averageOfAreas/float(nonBlackAreas)).astype(np.uint8)
                    info.update(self._findColorName(averageOfAreas))

                labels[label] = info

            self.peopleLabels.append(labels)

        return self.peopleLabels

    def train(self, epochs=100, learningRate=0.005, dataset="Coco",
              useDatabase=True, printUpdateEvery=40,
              visualize=False, tensorboard=False):
        self._training = True
        self._initTraining(learningRate, dataset, useDatabase)

        # Deal with tensorboard
        if tensorboard or type(tensorboard) == str:
            from torch.utils.tensorboard import SummaryWriter

            if type(tensorboard) == str:
                writer = SummaryWriter("./data/tensorboard/"+tensorboard)
            else:
                writer = SummaryWriter("./data/tensorboard/")
            tensorboard = True

        def findBestROI(ROIs, label):
            bestMatch = 0
            bestIndex = -1
            for i, ROI in enumerate(ROIs):
                lbox = np.array(label["bbox"])
                larea = lbox[2:]-lbox[:2]
                larea = larea[0]*larea[1]
                rbox = ROI.bounds
                rarea = rbox[2:] - rbox[:2]
                rarea = rarea[0] * rarea[1]

                SI = np.maximum(0, np.minimum(lbox[2], rbox[2]) - np.maximum(lbox[0], rbox[0])) * \
                     np.maximum(0, np.minimum(lbox[3], rbox[3]) - np.maximum(lbox[1], rbox[1]))
                SU = larea + rarea - SI
                overlap = SI/SU
                if bestMatch < overlap and SU != 0:
                    bestMatch = overlap
                    bestIndex = i
            return bestIndex

        Iterations = len(self.dataset)

        print("Starting training")
        for epoch in range(epochs):
            epochLoss = np.float64(0)
            for i in range(Iterations):
                ROIs, peopleTextures, labels = self._load(i)

                # Figure out what ROI belongs to what label
                groundtruth = np.zeros((len(ROIs), 14), dtype=np.float32)
                for label in labels:
                    mostMatching = findBestROI(ROIs, label)
                    if mostMatching != -1:
                        groundtruth[mostMatching][label["category_id"]] = 1

                # Most items in this dataset will be bypassed because no people were found or overlapping with gt
                if len(ROIs) == 0 or not np.any(groundtruth != 0):
                    continue

                groundtruth = torch.from_numpy(groundtruth).to(device)

                # Apply noise to peopleTextures
                noise = np.random.randn(*peopleTextures.shape)*5
                peopleTextures = peopleTextures.astype(np.int32) + noise.astype(np.int32)
                peopleTextures = np.clip(peopleTextures, 0, 255)
                peopleTextures = peopleTextures.astype(np.uint8)

                peopleTextures = torch.Tensor(peopleTextures).to(device)
                predictions = self.classifier.forward(peopleTextures)
                print(groundtruth)
                print(predictions)
                print("\n")

                lossSize = self.lossFunction(predictions, groundtruth)
                lossSize.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lossSize = lossSize.cpu().item()

                epochLoss += lossSize / Iterations
                if (i - 1) % printUpdateEvery == 0:
                    print("Iteration {} / {}, epoch {} / {}".format(i, Iterations, epoch, epochs))
                    print("Loss size: {}\n".format(lossSize / printUpdateEvery))

                if tensorboard:
                    absI = i + epoch * Iterations
                    writer.add_scalar("Loss size", lossSize, absI)

            print("Finished epoch {} / {}. Loss size:".format(epoch, epochs, epochLoss))
            self.saveModel(self.modelPath)

        self._training = False

    def getLabelImage(self):
        images = []
        for personLabel in self.peopleLabels:
            # Sort labels by score
            labels = sorted(list(personLabel.items()), key=lambda x: x[1]["activation"], reverse=True)

            # Create image
            image = np.zeros((160, 210, 3))
            for i, label in enumerate(labels):
                name, classification = label
                text = "{0:4d}%   {1}".format(
                    int(classification["activation"]*100), name)
                color = (255, 255, 255)
                if classification["activation"] < 0.75:  # FIXME: magic number, tune
                    color = (128, 128, 128)
                image = cv2.putText(image, text, (0, 12+12*i), cv2.FONT_HERSHEY_DUPLEX, .3,
                                    color, 1, cv2.LINE_AA)

                # Add color
                if "bestMatch" in classification:
                    colorText = classification["bestMatch"][1]
                    colorTextColor = classification["color"]
                    colorTextColor = (int(colorTextColor[0]), int(colorTextColor[1]), int(colorTextColor[2]))
                    image = cv2.putText(image, colorText, (150, 12+12*i), cv2.FONT_HERSHEY_DUPLEX, .3,
                                        colorTextColor, 1, cv2.LINE_AA)

            images.append(image.astype(np.uint8))

        return images

    def _load(self, index):
        cocoImage = self.dataset[index]
        ROIs = None
        if self.useDatabase:
            ROIs = self.lmdb.get(DensePoseWrapper, "deepfashion2" + str(index))
        if ROIs is None:
            ROIs = self.denseposeExtractor.extract(cocoImage[0])
            ROIs = self.sanitizer.extract(ROIs)
            if self.useDatabase:
                self.lmdb.save(DensePoseWrapper, "deepfashion2" + str(index), ROIs)

        peopleTextures = None
        if self.useDatabase:
            peopleTextures = self.lmdb.get(UVMapper, "deepfashion2" + str(index))
        if peopleTextures is None:
            peopleTextures = self.uvMapper.extract(ROIs, cocoImage[0])
            if self.useDatabase:
                self.lmdb.save(UVMapper, "deepfashion2" + str(index), peopleTextures)

        return ROIs, peopleTextures, cocoImage[1]

    def _findColorName(self, color):
        b = color[0]
        g = color[1]
        r = color[2]

        # This prints the color colored in the terminal
        colorRepr = '\033[{};2;{};{};{}m'.format(38, r, g, b) \
                    + "rgb("+str(r)+", "+str(g)+", "+str(b)+")"+'\033[0m'

        # Get nearest color name
        HSVobj = convert_color(sRGBColor(r, g, b), HSVColor)

        nearestIndex = -1
        diffMin = 100000
        for i in range(len(self.colorsHSV)):
            colEntry = self.colorsHSV[i]

            d = HSVobj.hsv_h - colEntry.hsv_h
            dh = min(abs(d), 360 - abs(d)) / 180.0
            ds = abs(HSVobj.hsv_s - colEntry.hsv_s)
            dv = abs(HSVobj.hsv_v - colEntry.hsv_v) / 255.0
            diff = np.sqrt(dh * dh + ds * ds + dv * dv)
            if diff < diffMin:
                diffMin = diff
                nearestIndex = i

        return {
            "color": tuple(color),
            "colorDistance": diffMin,
            "coloredStr": colorRepr,
            "bestMatch": self.colors[nearestIndex]
        }
