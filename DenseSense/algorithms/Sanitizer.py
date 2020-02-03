import time

import DenseSense.algorithms.Algorithm
from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
from DenseSense.utils.LMDBHelper import LMDBHelper

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

topDir = os.path.realpath(os.path.dirname(__file__)+"/../..")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("Sanitizer running on: " + str(device))


class Sanitizer(DenseSense.algorithms.Algorithm.Algorithm):

    # UNet, inspired by https://github.com/usuyama/pytorch-unet/
    # But with a fully connected layer in the middle
    class MaskGenerator(nn.Module):
        def __init__(self):
            super(Sanitizer.MaskGenerator, self).__init__()

            self.dconv1 = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=2),
                nn.LeakyReLU(inplace=True),
            )

            self.dconv2 = nn.Sequential(
                nn.Conv2d(8, 4, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

            self.dconv3 = nn.Sequential(
                nn.Conv2d(3, 1, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

            self.fcImg = nn.Linear(14*14*2+2, 14*14)

            self.maxpool = nn.MaxPool2d(2)
            self.upsample1 = nn.Upsample(size=(29, 29), mode="bilinear")
            self.upsample2 = nn.Upsample(size=(56, 56), mode="bilinear")

            self.sigmoid = nn.Sigmoid()
            self.leakyReLU = nn.LeakyReLU()

        def forward(self, people):
            if len(people) == 0:
                return np.array([]), torch.Tensor([]).to(device)

            # Send data to device
            S = torch.Tensor(len(people), 1, 56, 56)
            b = torch.Tensor(len(people), 2)
            for i in range(len(people)):
                person = people[i]
                S[i][0] = torch.from_numpy(person.S)
                bnds = person.bounds
                area = np.power(np.sqrt((bnds[2] - bnds[0]) * (bnds[3] - bnds[1])), 0.2)
                if bnds[3] == bnds[1]:
                    aspect = 0
                else:
                    aspect = (bnds[2] - bnds[0]) / (bnds[3] - bnds[1])
                b[i] = torch.Tensor([area, aspect])

            S = S.to(device)
            b = b.to(device)
            batchSize = S.shape[0]

            # Normalize input
            x = S.clone()
            x[0 < x] = x[0 < x] / 15.0 * 0.2 + 0.8

            # Convolutions
            x = self.dconv1(x)  # 1 -> 8, 56x56 -> 58x58
            x = self.maxpool(x)     # 58x58 -> 29x29
            conv = self.dconv2(x)  # 8 -> 4
            x = self.maxpool(conv[:, :2])     # 29x29 -> 14x14

            # Fully connected layer
            x = x.view(batchSize, 14*14*2)
            x = torch.cat([x, b], dim=1)  # Image and bbox info
            x = self.fcImg(x)
            x = self.leakyReLU(x)
            x = x.view(batchSize, 1, 14, 14)

            # Merge fully connected with past convolution calculation
            x = self.upsample1(x)  # 14x14 -> 29x29
            x = torch.cat([x, conv[:, 2:]], dim=1)
            x = self.dconv3(x)     # 3 -> 1
            x = self.sigmoid(x)
            x = self.upsample2(x)  # 29x29 -> 56x56

            return x, S

    def __init__(self):
        super().__init__()

        # Generate and maybe load mask generator model()
        self.maskGenerator = Sanitizer.MaskGenerator()
        self.modelPath = None

        self._training = False
        self._trainingInitiated = False
        self._ROI_masks = torch.Tensor()
        self._ROIs = torch.Tensor()
        self._ROI_bounds = np.array([])
        self._overlappingROIs = np.array([])
        self._overlappingROIsValues = np.array([])

    def loadModel(self, modelPath):
        self.modelPath = modelPath
        print("Loading Sanitizer MaskGenerator file from: " + self.modelPath)
        self.maskGenerator.load_state_dict(torch.load(self.modelPath, map_location=device))
        self.maskGenerator.to(device)

    def saveModel(self, modelPath):
        if modelPath is None:
            print("Don't know where to save model")
        self.modelPath = modelPath
        print("Saving Sanitizer MaskGenerator model to: "+self.modelPath)
        torch.save(self.maskGenerator.state_dict(), self.modelPath)

    def _initTraining(self, learningRate, dataset, useDatabase):
        # Dataset is COCO
        print("Initiating training of Sanitizer MaskGenerator")
        print("Loading COCO")
        from pycocotools.coco import COCO
        from os import path

        # TODO: support other data sets than Coco
        annFile = topDir+'/annotations/instances_{}.json'.format(dataset)
        self.cocoPath = topDir+'/data/{}'.format(dataset)

        self.coco = COCO(annFile)
        self.personCatID = self.coco.getCatIds(catNms=['person'])[0]
        self.cocoImageIds = self.coco.getImgIds(catIds=self.personCatID)

        def isNotCrowd(imgId):
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=self.personCatID, iscrowd=False)
            annotation = self.coco.loadAnns(annIds)[0]
            return not annotation["iscrowd"]

        self.cocoImageIds = list(filter(isNotCrowd, self.cocoImageIds))
        self.cocoOnDisk = path.exists(self.cocoPath)

        print("Coco dataset size: {}".format(len(self.cocoImageIds)))
        print("Coco images found on disk:", self.cocoOnDisk)

        # Init LMDB_helper
        if useDatabase:
            self.lmdb = LMDBHelper("a")
            self.lmdb.verbose = False

        # Init loss function and optimizer
        self.optimizer = torch.optim.Adam(self.maskGenerator.parameters(), lr=learningRate, amsgrad=True)
        self.lossFunction = torch.nn.MSELoss()

        # Init DensePose extractor
        self.denseposeExtractor = DensePoseWrapper()

    def extract(self, people):
        # Generate masks for all ROIs (people) using neural network model
        with torch.no_grad():
            self._generateMasks(people)

            if len(self._ROI_masks) == 0:
                return people

            # Multiply masks with with segmentation mask from DensePose
            masked = self._ROI_masks*self._ROIs

            # Find overlapping ROIs
            overlaps, overlapLow, overlapHigh = self._overlappingMatrix(
                self._ROI_bounds.astype(np.int32),
                self._ROI_bounds.astype(np.int32)
            )
            overlaps[np.triu_indices(overlaps.shape[0])] = False
            overlapsInds = np.array(list(zip(*np.where(overlaps))))
            overlapsCorr = np.full_like(overlaps, 0, dtype=np.float)

            # Find correlations between overlapping ROIs
            if overlapsInds.shape[0] != 0:
                for a, b in overlapsInds:  # For every overlap
                    # Extract part that overlaps from mask and make sizes match to smallest dim
                    xCoords = np.array([overlapLow[0][a, b], overlapHigh[0][a, b]])
                    yCoords = np.array([overlapLow[1][a, b], overlapHigh[1][a, b]])
                    aMask = self._getTransformedROI(masked[a, 0], self._ROI_bounds[a], xCoords, yCoords)
                    bMask = self._getTransformedROI(masked[b, 0], self._ROI_bounds[b], xCoords, yCoords)
                    aArea = aMask.shape[0]*aMask.shape[1]
                    bArea = bMask.shape[0]*bMask.shape[1]

                    # Scale down the biggest one
                    if aArea < bArea:
                        bMask = bMask.unsqueeze(0)
                        bMask = F.adaptive_avg_pool2d(bMask, aMask.shape)[0]
                    else:
                        aMask = aMask.unsqueeze(0)
                        aMask = F.adaptive_avg_pool2d(aMask, bMask.shape)[0]

                    # Calculate correlation
                    aMean = aMask.mean()
                    bMean = bMask.mean()
                    correlation = torch.sum((aMask-aMean)*(bMask-bMean))/(aMask.shape[0]*aMask.shape[1]-1)
                    overlapsCorr[a, b] = correlation

            # Find best disjoint sets of overlapping ROIs
            threshold = 0.06  # Must be above 0

            goodCorrelations = np.argwhere(threshold < overlapsCorr)
            sortedCorrelations = overlapsCorr[goodCorrelations[:, 0], goodCorrelations[:, 1]].argsort()
            goodCorrelations = goodCorrelations[sortedCorrelations]
            overlapsCorr += overlapsCorr.T
            coupled = {}

            def getBiPotential(a, diff):
                potential = 0
                for bOther in np.argwhere(overlapsCorr[diff] != 0):
                    bOther = bOther[0]
                    if bOther in coupled[a][0]:
                        potential += overlapsCorr[a, bOther]
                return potential

            for a, b in goodCorrelations:
                aIn = a in coupled
                bIn = b in coupled
                if aIn:
                    if bIn:
                        potential = overlapsCorr[a, b]
                        for diff in coupled[b][0]:
                            potential += getBiPotential(a, diff)
                        if 0 < potential:
                            coupled[a][0].update(coupled[b][0])
                            for diff in coupled[b][0]:
                                coupled[diff] = coupled[a]
                                coupled[a][1] += potential
                    else:
                        potential = overlapsCorr[a, b] + getBiPotential(a, b)
                        if 0 < potential:
                            coupled[a][0].add(b)
                            coupled[a][1] += potential
                            coupled[b] = coupled[a]
                elif bIn:
                    potential = overlapsCorr[b, a] + getBiPotential(b, a)
                    if 0 < potential:
                        coupled[b][0].add(a)
                        coupled[b][1] += potential
                        coupled[a] = coupled[b]
                else:
                    n = [{a, b}, overlapsCorr[a, b]]
                    coupled[a] = n
                    coupled[b] = n

            # Update all people data their data.
            ActiveThreshold = 0.2  # FIXME: magic number
            newPeople = []
            skip = set()
            for i, person in enumerate(people):
                if i not in skip:
                    if i in coupled:
                        # Merge all coupled into one person
                        instances = list(coupled[i][0])
                        for j in instances:
                            skip.add(j)
                        instances = list(map(lambda i: people[i], instances))
                        instances[0].merge(instances[1:])
                        newPeople.append(instances[0])
                    else:
                        # Lonely ROIs are kept alive if it is at least 20 % active
                        active = torch.mean(masked[i])
                        if ActiveThreshold < active:
                            newPeople.append(person)

            return newPeople

    def train(self, epochs=100, learningRate=0.005, dataset="Coco",
              useDatabase=True, printUpdateEvery=40,
              visualize=False, tensorboard=False):

        self._training = True
        if not self._trainingInitiated:
            self._initTraining(learningRate, dataset, useDatabase)

        if tensorboard or type(tensorboard) == str:
            from torch.utils.tensorboard import SummaryWriter

            if type(tensorboard) == str:
                writer = SummaryWriter(topDir+"/data/tensorboard/"+tensorboard)
            else:
                writer = SummaryWriter(topDir+"/data/tensorboard/")
            tensorboard = True

            # dummy_input = torch.Tensor(5, 1, 56, 56)
            # writer.add_graph(self.maskGenerator, dummy_input)
            # writer.close()

        Iterations = len(self.cocoImageIds)

        meanPixels = []

        print("Starting training")

        for epoch in range(epochs):
            epochLoss = np.float64(0)
            interestingImage = None
            interestingMeasure = -100000
            for i in range(Iterations):

                # Load instance of COCO dataset
                cocoImage, image = self._getCocoImage(i)
                if image is None:  # FIXME
                    print("Image is None??? Skipping.", i)
                    print(cocoImage)
                    continue

                # Get annotation
                annIds = self.coco.getAnnIds(imgIds=cocoImage["id"], catIds=self.personCatID, iscrowd=False)
                annotation = self.coco.loadAnns(annIds)

                # Draw each person in annotation to separate mask
                segs = []
                seg_bounds = []
                for person in annotation:
                    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
                    for s in person["segmentation"]:
                        s = np.reshape(np.array(s, dtype=np.int32), (-2, 2))
                        cv2.fillPoly(mask, [s], 1)
                    segs.append(mask)
                    bbox = person["bbox"]
                    seg_bounds.append(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]))

                seg_bounds = np.array(seg_bounds, dtype=np.int32)

                # Get DensePose data from DB or Extractor
                generated = False
                ROIs = None
                if useDatabase:
                    ROIs = self.lmdb.get(DensePoseWrapper, "coco" + str(cocoImage["id"]))
                if ROIs is None:
                    ROIs = self.denseposeExtractor.extract(image)
                    generated = True
                if useDatabase and generated:
                    self.lmdb.save(DensePoseWrapper, "coco" + str(cocoImage["id"]), ROIs)

                # Run prediction
                self._generateMasks(ROIs)

                if len(self._ROI_masks) == 0:
                    continue

                if tensorboard:
                    means = [torch.mean(ROI).detach().cpu().numpy() for ROI in self._ROI_masks]
                    meanPixels.append(sum(means)/len(means))

                # Find overlaps between bboxes of segs and ROIs
                overlaps, overlapLow, overlapHigh = self._overlappingMatrix(
                    seg_bounds.astype(np.int32),
                    self._ROI_bounds.astype(np.int32)
                )

                overlapsInds = np.array(list(zip(*np.where(overlaps))))
                if overlapsInds.shape[0] == 0:
                    continue

                # Get average value where there is overlap between COCO-mask for each person and predictions for
                contentAverage = {}
                for a, b in overlapsInds:  # For every overlap
                    xCoords = np.array([overlapLow[0][a, b], overlapHigh[0][a, b]])
                    yCoords = np.array([overlapLow[1][a, b], overlapHigh[1][a, b]])
                    ROI_mask = self._getTransformedROI(self._ROI_masks[a, 0], self._ROI_bounds[a], xCoords, yCoords)

                    # Segmentation overlap area
                    segOverlap = segs[b][yCoords[0]:yCoords[1], xCoords[0]:xCoords[1]]

                    # Transform segmentation
                    segOverlap = cv2.resize(segOverlap, (ROI_mask.shape[1], ROI_mask.shape[0]),
                                            interpolation=cv2.INTER_AREA)

                    # Calculate sum of product of the ROI mask and segment overlap
                    segOverlap = torch.from_numpy(segOverlap).float().to(device)
                    avgVariable = torch.sum(ROI_mask * segOverlap)

                    # Store this sum
                    if str(a) not in contentAverage:
                        contentAverage[str(a)] = []

                    contentAverage[str(a)].append((avgVariable, segOverlap, ROI_mask))

                self._overlappingROIs = np.unique(overlapsInds[:, 0])

                # Choose which segment each ROI should be compared with
                losses = []
                for j in range(len(self._overlappingROIs)):  # For every ROI with overlap
                    a = self._overlappingROIs[j]

                    AL = list(contentAverage[str(a)])
                    AV = np.array([float(x[0].cpu()) for x in AL])

                    ind = AV.argmax()
                    lossSize = self.lossFunction(AL[ind][2], AL[ind][1])
                    losses.append(lossSize)

                # Modify weights
                losses = torch.stack(losses)
                lossSize = torch.sum(losses)
                lossSize.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lossSize = lossSize.cpu().item()

                epochLoss += lossSize/Iterations
                if (i-1) % printUpdateEvery == 0:
                    print("Iteration {} / {}, epoch {} / {}".format(i, Iterations, epoch, epochs))
                    print("Loss size: {}\n".format(lossSize / printUpdateEvery))
                    if tensorboard:
                        absI = i + epoch * Iterations
                        writer.add_scalar("Loss size", lossSize, absI)
                        writer.add_histogram("Mean ROI pixel value", np.array(meanPixels), absI)
                        meanPixels = []

                if tensorboard:
                    interestingness = np.random.random()  # just choose a random one
                    if interestingMeasure < interestingness:
                        interestingImage = self.renderDebug(image.copy())
                        interestingMeasure = interestingness

                # Show visualization
                if visualize:
                    image = self.renderDebug(image)
                    plt.ion()
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.draw()
                    plt.pause(4)

            print("Finished epoch {} / {}. Loss size:".format(epoch, epochs, epochLoss))
            if tensorboard:
                writer.add_scalar("epoch loss size", epochLoss, Iterations*epoch)
                if interestingImage is not None:
                    interestingImage = cv2.cvtColor(interestingImage, cv2.COLOR_BGR2RGB)
                    interestingImage = torch.from_numpy(interestingImage).permute(2, 0, 1)
                    writer.add_image("interesting image", interestingImage, Iterations*epoch)
            self.saveModel(self.modelPath)

        self._training = False

    def _generateMasks(self, ROIs):
        self._ROI_masks, self._ROIs = self.maskGenerator.forward(ROIs)
        self._ROIs[self._ROIs != 0] = 1
        self._ROI_bounds = np.zeros((len(ROIs), 4), dtype=np.int32)
        for i in range(len(ROIs)):
            self._ROI_bounds[i] = np.array(ROIs[i].bounds, dtype=np.int32)
            ROIs[i].A = torch.round(self._ROI_masks[i, 0]).cpu().numpy()

    def _getCocoImage(self, index):
        if self.cocoOnDisk:
            # Load image from disk
            cocoImage = self.coco.loadImgs(self.cocoImageIds[index])[0]
            image = cv2.imread(self.cocoPath + "/" + cocoImage["file_name"])
            return cocoImage, image
        else:
            raise FileNotFoundError("COCO image cant be found on disk")

    @staticmethod
    def _overlappingMatrix(a, b):
        xo_high = np.minimum(a[:, 2], b[:, None, 2])
        xo_low = np.maximum(a[:, 0], b[:, None, 0])
        xo = xo_high - xo_low

        yo_high = np.minimum(a[:, 3], b[:, None, 3])
        yo_low = np.maximum(a[:, 1], b[:, None, 1])
        yo = yo_high - yo_low

        overlappingMask = np.logical_and((0 < xo), (0 < yo))
        return overlappingMask, (xo_low, yo_low), (xo_low + xo, yo_low + yo)

    @staticmethod
    def _getTransformedROI(ROI, bounds, xCoords, yCoords):
        # ROI transformed overlap area
        ROI_xCoords = (xCoords -bounds[0]) / (bounds[2] - bounds[0])
        ROI_xCoords = (ROI_xCoords * 56).astype(np.int32)
        ROI_xCoords[1] += ROI_xCoords[0] == ROI_xCoords[1]
        ROI_yCoords = (yCoords - bounds[1]) / (bounds[3] - bounds[1])
        ROI_yCoords = (ROI_yCoords * 56).astype(np.int32)
        ROI_yCoords[1] += ROI_yCoords[0] == ROI_yCoords[1]

        ROI_mask = ROI[ROI_yCoords[0]:ROI_yCoords[1], ROI_xCoords[0]:ROI_xCoords[1]]

        return ROI_mask

    def renderDebug(self, image, people, alpha=0.55):
        # Normalize ROIs from (0, 1) to (0, 255)
        ROIsMaskNorm = self._ROI_masks * 255

        # Render masks on image
        for i in range(len(self._ROI_masks)):
            mask = ROIsMaskNorm[i, 0].cpu().detach().to(torch.uint8).numpy()
            bnds = self._ROI_bounds[i]

            # Change colors of mask
            if 0 < alpha:
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)
            else:
                alpha = -alpha
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_PINK)

            # TODO: render contours instead?

            # Resize mask to bounds
            dims = (bnds[2] - bnds[0], bnds[3] - bnds[1])
            mask = cv2.resize(mask, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            mask = mask * alpha + overlap * (1.0 - alpha)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = mask

        for person in people:
            bnds = person.bounds
            image = cv2.rectangle(image, (bnds[0], bnds[1]),
                                  (bnds[2], bnds[3]),
                                  (60, 20, 20), 2)

        return image
