import time

import DenseSense.algorithms.Algorithm
from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
from DenseSense.utils.LMDBHelper import LMDBHelper

import cv2
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

    def __init__(self):
        super().__init__()

        # Generate and maybe load mask generator model()
        self.maskGenerator = AutoEncoder()
        self.modelPath = None

        self._training = False
        self._trainingInitiated = False
        self._ROI_masks = torch.Tensor()
        self._ROI_bounds = np.array([])
        self._overlappingROIs = np.array([])
        self._overlappingROIsValues = np.array([])

    def load_model(self, modelPath):
        self.modelPath = modelPath
        print("Loading Sanitizer MaskGenerator file from: " + self.modelPath)
        self.maskGenerator.load_state_dict(torch.load(self.modelPath, map_location=device))
        self.maskGenerator.to(device)

    def save_model(self, modelPath):
        if modelPath is None:
            print("Don't know where to save model")
        self.modelPath = modelPath
        print("Saving Sanitizer MaskGenerator model to: "+self.modelPath)
        torch.save(self.maskGenerator.state_dict(), self.modelPath)

    def _init_training(self, learningRate, dataset, useDatabase):
        # Dataset is COCO
        print("Initiating training of Sanitizer MaskGenerator")
        print("Loading COCO")
        from pycocotools.coco import COCO
        from os import path

        annFile = topDir+'/annotations/instances_{}.json'.format(dataset)
        self.cocoPath = topDir+'/data/{}'.format(dataset)

        self.coco = COCO(annFile)
        self.personCatID = self.coco.getCatIds(catNms=['person'])[0]
        self.cocoImageIds = self.coco.getImgIds(catIds=self.personCatID)

        def is_not_crowd(imgId):
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=self.personCatID, iscrowd=False)
            annotation = self.coco.loadAnns(annIds)[0]
            return not annotation["iscrowd"]

        self.cocoImageIds = list(filter(is_not_crowd, self.cocoImageIds))
        self.cocoOnDisk = path.exists(self.cocoPath)

        print("Coco dataset size: {}".format(len(self.cocoImageIds)))
        print("Coco images found on disk:", self.cocoOnDisk)

        # Init LMDB_helper
        if useDatabase:
            self.lmdb = LMDBHelper("a")
            self.lmdb.verbose = False

        # Init loss function and optimizer
        self.optimizer = torch.optim.Adam(self.maskGenerator.parameters(), lr=learningRate, amsgrad=True)
        self.loss_function = torch.nn.BCELoss()

        # Init DensePose extractor
        self.denseposeExtractor = DensePoseWrapper()

    def extract(self, people):
        # Generate masks for all ROIs (people) using neural network model
        with torch.no_grad():
            self._ROI_masks, Ss = self._generate_masks(people)

        self._ROI_bounds = np.zeros((len(people), 4), dtype=np.int32)
        for i in range(len(people)):
            self._ROI_bounds[i] = np.array(people[i].bounds, dtype=np.int32)

        if len(self._ROI_masks) == 0:
            return people

        # Multiply masks with with segmentation mask from DensePose
        masked = self._ROI_masks*Ss

        # Find overlapping ROIs
        overlaps, overlapLow, overlapHigh = self._overlapping_matrix(
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
                aMask = self._get_transformed_roi(masked[a, 0], self._ROI_bounds[a], xCoords, yCoords)
                bMask = self._get_transformed_roi(masked[b, 0], self._ROI_bounds[b], xCoords, yCoords)
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

        def get_bi_potential(a, diff):
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
                        potential += get_bi_potential(a, diff)
                    if 0 < potential:
                        coupled[a][0].update(coupled[b][0])
                        for diff in coupled[b][0]:
                            coupled[diff] = coupled[a]
                            coupled[a][1] += potential
                else:
                    potential = overlapsCorr[a, b] + get_bi_potential(a, b)
                    if 0 < potential:
                        coupled[a][0].add(b)
                        coupled[a][1] += potential
                        coupled[b] = coupled[a]
            elif bIn:
                potential = overlapsCorr[b, a] + get_bi_potential(b, a)
                if 0 < potential:
                    coupled[b][0].add(a)
                    coupled[b][1] += potential
                    coupled[a] = coupled[b]
            else:
                n = [{a, b}, overlapsCorr[a, b]]
                coupled[a] = n
                coupled[b] = n

        # Update all people data their data.
        ActiveThreshold = 0.1  # FIXME: magic number
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
                    else:
                        print("Forgot ", active)

        # Generate a mask again for the whole person, allowing for a holistic judgement
        self._generate_masks(newPeople)

        # TODO: find the edges of the person and crop

        return newPeople

    def train(self, epochs=100, learning_rate=0.005, dataset="Coco",
              use_database=True, print_update_every=40,
              visualize=0, tensorboard=False):

        self._training = True
        if not self._trainingInitiated:
            self._init_training(learning_rate, dataset, use_database)

        if tensorboard or type(tensorboard) == str:
            from torch.utils.tensorboard import SummaryWriter

            if type(tensorboard) == str:
                writer = SummaryWriter(topDir+"/data/tensorboard/"+tensorboard)
            else:
                writer = SummaryWriter(topDir+"/data/tensorboard/")
            tensorboard = True


        total_iterations = len(self.cocoImageIds)
        visualize_counter = 0

        meanPixels = []

        print("Starting training")

        for epoch in range(epochs):
            epoch_loss = np.float64(0)
            for i in range(total_iterations):
                # Load instance of COCO dataset
                cocoImage, image = self._get_coco_image(i)
                if image is None:  # FIXME
                    print("Image is None??? Skipping.", i)
                    print(cocoImage)
                    continue

                # Get annotation
                annIds = self.coco.getAnnIds(imgIds=cocoImage["id"], catIds=self.personCatID, iscrowd=False)
                annotation = self.coco.loadAnns(annIds)

                # Get DensePose data from DB or Extractor
                generated = False
                ROIs = None
                if use_database:
                    ROIs = self.lmdb.get("DensePoseWrapper_Coco", str(cocoImage["id"]))
                if ROIs is None:
                    ROIs = self.denseposeExtractor.extract(image)
                    generated = True
                if use_database and generated:
                    self.lmdb.save("DensePoseWrapper_Coco", str(cocoImage["id"]), ROIs)

                # Run prediction
                self._ROI_masks, Ss = self._generate_masks(ROIs)

                # Store bounds
                self._ROI_bounds = np.zeros((len(ROIs), 4), dtype=np.int32)
                for j in range(len(ROIs)):
                    self._ROI_bounds[j] = np.array(ROIs[j].bounds, dtype=np.int32)

                if len(self._ROI_masks) == 0:
                    continue

                if tensorboard:
                    means = [torch.mean(ROI).detach().cpu().numpy() for ROI in self._ROI_masks]
                    meanPixels.append(sum(means)/len(means))

                # Draw each person in annotation to separate mask from polygon vertices
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

                # Find overlaps between bboxes of segs and ROIs
                overlaps, overlapLow, overlapHigh = self._overlapping_matrix(
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
                    ROI_mask = self._get_transformed_roi(self._ROI_masks[a, 0], self._ROI_bounds[a], xCoords, yCoords)

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
                    lossSize = self.loss_function(AL[ind][2], AL[ind][1])
                    lossSize.backward(retain_graph=True)

                    losses.append(lossSize.detach().cpu().float())

                self.optimizer.step()
                self.optimizer.zero_grad()

                lossSize = sum(losses)/len(losses)

                epoch_loss += lossSize/total_iterations
                visualize_counter += 1
                if (i-1) % print_update_every == 0:
                    print("Iteration {} / {}, epoch {} / {}".format(i, total_iterations, epoch, epochs))
                    print("Loss size: {}\n".format(lossSize / print_update_every))
                    if tensorboard:
                        absI = i + epoch * total_iterations
                        writer.add_scalar("Loss size", lossSize, absI)
                        writer.add_histogram("Mean ROI pixel value", np.array(meanPixels), absI)
                        meanPixels = []

                # Show visualization
                if visualize != 0 and visualize <= visualize_counter:
                    visualize_counter = 0
                    image = self.renderDebug(image, None, annotated_segs=segs)
                    cv2.imshow("Sanitizer training", image)
                    for j, m in enumerate(self._ROI_masks):
                        if j > 6:
                            break
                        cv2.imshow("Mask "+str(j), (m[0]*255).cpu().detach().to(torch.uint8).numpy())

                    if len(self._ROI_masks) >= 2:
                        cv2.imshow("Mask diff", (torch.abs(self._ROI_masks[0][0]-self._ROI_masks[1][0]) * 255)
                                   .cpu().detach().to(torch.uint8).numpy())
                    cv2.waitKey(1)

            print("Finished epoch {} / {}. Loss size:".format(epoch, epochs, epoch_loss))
            if tensorboard:
                writer.add_scalar("epoch loss size", epoch_loss, total_iterations*epoch)
            self.save_model(self.modelPath)

        self._training = False

    def _generate_masks(self, ROIs):
        Ss = self._tensorify_ROIs(ROIs)
        masks = []
        if len(ROIs) != 0:
            masks = self.maskGenerator.forward(Ss)

        for i in range(len(ROIs)):
            ROIs[i].A = torch.round(masks[i, 0]).detach().cpu().numpy()

        return masks, Ss

    def _get_coco_image(self, index):
        if self.cocoOnDisk:
            # Load image from disk
            cocoImage = self.coco.loadImgs(self.cocoImageIds[index])[0]
            image = cv2.imread(self.cocoPath + "/" + cocoImage["file_name"])
            return cocoImage, image
        else:
            raise FileNotFoundError("COCO image cant be found on disk")

    @staticmethod
    def _tensorify_ROIs(ROIs):
        S = torch.Tensor(len(ROIs), 1, 56, 56)
        for j in range(len(ROIs)):
            person = ROIs[j]
            S[j][0] = torch.from_numpy(person.S)

        S = S.to(device)
        S[0 < S] = S[0 < S] / 15.0 * 0.8 + 0.2
        return S

    @staticmethod
    def _overlapping_matrix(a, b):
        xo_high = np.minimum(a[:, 2], b[:, None, 2])
        xo_low = np.maximum(a[:, 0], b[:, None, 0])
        xo = xo_high - xo_low

        yo_high = np.minimum(a[:, 3], b[:, None, 3])
        yo_low = np.maximum(a[:, 1], b[:, None, 1])
        yo = yo_high - yo_low

        overlappingMask = np.logical_and((0 < xo), (0 < yo))
        return overlappingMask, (xo_low, yo_low), (xo_low + xo, yo_low + yo)

    @staticmethod
    def _get_transformed_roi(ROI, bounds, x_coords, y_coords):
        # ROI transformed overlap area
        ROI_xCoords = (x_coords - bounds[0]) / (bounds[2] - bounds[0])
        ROI_xCoords = (ROI_xCoords * 56).astype(np.int32)
        ROI_xCoords[1] += ROI_xCoords[0] == ROI_xCoords[1]
        ROI_yCoords = (y_coords - bounds[1]) / (bounds[3] - bounds[1])
        ROI_yCoords = (ROI_yCoords * 56).astype(np.int32)
        ROI_yCoords[1] += ROI_yCoords[0] == ROI_yCoords[1]

        ROI_mask = ROI[ROI_yCoords[0]:ROI_yCoords[1], ROI_xCoords[0]:ROI_xCoords[1]]

        return ROI_mask

    def renderDebug(self, image, people, annotated_segs=None, alpha=0.65):
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

            # Resize mask to bounds
            dims = (bnds[2] - bnds[0], bnds[3] - bnds[1])
            mask = cv2.resize(mask, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            mask = mask * alpha + overlap * (1.0 - alpha)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = mask

        if people is not None:
            for person in people:
                bnds = person.bounds
                image = cv2.rectangle(image, (bnds[0], bnds[1]),
                                      (bnds[2], bnds[3]),
                                      (60, 20, 20), 1)

        # Render annotated segmentations
        if annotated_segs is not None:
            for seg in annotated_segs:
                seg = seg * 60
                image = image.astype(np.int32)
                image[:, :, 0] += seg
                image[:, :, 1] += seg // 3
                image[:, :, 2] += seg // 3
                image = image.clip(0, 255).astype(np.uint8)

        return image


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Make2D(torch.nn.Module):
    def __init__(self, w, h):
        super(Make2D, self).__init__()
        self.w = w
        self.h = h

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1, self.w, self.h)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder_first = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2, 2),
            Flatten()
        )

        self.encoder_second = nn.Sequential(
            nn.Linear(784+1, 100),  # +1 is delta time
            nn.Sigmoid(),
            nn.Linear(100, 5),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 100),
            nn.LeakyReLU(True),
            nn.Linear(100, 784),
            nn.LeakyReLU(True),
            Make2D(14, 14),
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, S):
        extra = torch.Tensor([np.random.rand()*0.5])
        extra = torch.cat(S.shape[0] * [extra]).reshape((S.shape[0], -1))
        # Run model
        x = self.encoder_first(S)
        x = torch.cat([x, extra], 1).to(device)
        x = self.encoder_second(x)
        #print(x)
        x = self.decoder(x)
        return x
