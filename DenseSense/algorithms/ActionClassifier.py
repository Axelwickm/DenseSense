import json
import os

import cv2

import DenseSense.algorithms.Algorithm
import DenseSense.utils.YoutubeLoader
from DenseSense.utils.LMDBHelper import LMDBHelper

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


topDir = os.path.realpath(os.path.dirname(__file__)+"/../..")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("ActionClassifier running on: " + str(device))


class ActionClassifier(DenseSense.algorithms.Algorithm.Algorithm):
    actions = {
        4:  "dance",
        11: "sit",
        14: "walk",
        69: "hand wave",
        
        12: "idle",  # stand
        17: "idle",  # carry/hold (an object)
        36: "idle",  # lift/pick up
        37: "idle",  # listen
        47: "idle",  # put down
    }

    COCO_Datasets = ["val2014", "train2014", "val2017", "train2017"]

    def __init__(self):
        print("Initiating ActionClassifier")
        super().__init__()

        self._modelPath = None
        self._AE_model = AutoEncoder()
        self._training = False

    def loadModel(self, modelPath):
        self._modelPath = modelPath
        print("Loading ActionClassifier file from: " + self._modelPath)
        self._AE_model.load_state_dict(torch.load(self._modelPath, map_location=device))
        self._AE_model.to(device)

    def saveModel(self, modelPath):
        if modelPath is None:
            print("Don't know where to save model")
        self._modelPath = modelPath
        print("Saving ActionClassifier model to: "+self._modelPath)
        torch.save(self._AE_model.state_dict(), self._modelPath)

    def _initTraining(self, learningRate, datasetName, useLMDB):
        self.datasetName = datasetName

        if datasetName in ActionClassifier.COCO_Datasets:
            print("Loading COCO dataset: "+datasetName)
            from pycocotools.coco import COCO
            from os import path

            from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
            from DenseSense.algorithms.Sanitizer import Sanitizer

            annFile = topDir + '/annotations/instances_{}.json'.format(datasetName)
            self.cocoPath = topDir + '/data/{}'.format(datasetName)

            self.coco = COCO(annFile)
            personCatID = self.coco.getCatIds(catNms=['person'])[0]
            self.dataset = self.coco.getImgIds(catIds=personCatID)

            self.denseposeExtractor = DensePoseWrapper()
            self.sanitizer = Sanitizer()
            self.sanitizer.loadModel(topDir + "/models/Sanitizer.pth")

        self.useLMDB = useLMDB
        if useLMDB: # FIXME: make work
            self.lmdb = LMDBHelper("a")
            self.lmdb.verbose = False

        self.optimizer = torch.optim.Adam(self._AE_model.parameters(), lr=learningRate)
        self.loss_function = torch.nn.BCELoss()

    def _load(self, index):
        if self.datasetName in ActionClassifier.COCO_Datasets:
            people = None
            # Load image from disk and process
            cocoImage = self.coco.loadImgs(self.dataset[index])[0]

            if self.useLMDB:
                people = self.lmdb.get("DensePoseWrapper_Sanitized_Coco", str(cocoImage["id"]))

            if people is None:

                image = cv2.imread(self.cocoPath + "/" + cocoImage["file_name"])
                people = self.denseposeExtractor.extract(image)
                people = self.sanitizer.extract(people)
                if self.useLMDB:
                    self.lmdb.save("DensePoseWrapper_Sanitized_Coco", str(cocoImage["id"]), people)
            return people

    def trainAutoEncoder(self, epochs=100, learningRate=0.005, dataset="Coco",
                         useLMDB=True, printUpdateEvery=40,
                         visualize=0, tensorboard=False):
        self._training = True
        self._initTraining(learningRate, dataset, useLMDB)

        # Tensorboard setup
        if tensorboard or type(tensorboard) == str:
            from torch.utils.tensorboard import SummaryWriter

            if type(tensorboard) == str:
                writer = SummaryWriter(topDir+"/data/tensorboard/"+tensorboard)
            else:
                writer = SummaryWriter(topDir+"/data/tensorboard/")
            tensorboard = True

        # Start the training process
        total_iterations = len(self.dataset)
        visualize_counter = 0
        open_windows = set()

        print("Starting training")
        for epoch in range(epochs):
            epochLoss = np.float64(0)
            for i in range(total_iterations):
                people = self._load(i)

                if len(people) == 0:
                    continue

                # Extract the S;es
                S = torch.Tensor(len(people), 1, 56, 56)
                for j in range(len(people)):
                    person = people[j]
                    aspect_adjusted = person.S.copy()
                    S[j][0] = torch.from_numpy(aspect_adjusted)

                S = S.to(device).clone()

                # Normalize
                S[0 < S] = S[0 < S] / 15.0 * 0.8 + 0.2

                # Run prediction
                embedding = self._AE_model.encode(S)
                out = self._AE_model.decode(embedding)

                # Optimize
                lossSize = self.loss_function(out, S)
                lossSize.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                lossSize = lossSize.cpu().item()

                # Give feedback of training process
                epochLoss += lossSize / total_iterations
                visualize_counter += 1
                if (i - 1) % printUpdateEvery == 0:
                    print("Iteration {} / {}, epoch {} / {}".format(i, total_iterations, epoch, epochs))
                    print("Loss size: {}\n".format(lossSize / printUpdateEvery))

                if visualize != 0 and visualize <= visualize_counter:
                    visualize_counter = 0
                    new_open_windows = set()
                    for index, person in enumerate(people):
                        inpS = (S[index, 0].detach()*255).cpu().to(torch.uint8).numpy()
                        outS = (out[index, 0].detach()*255).cpu().to(torch.uint8).numpy()
                        emb = ((embedding[index].detach().cpu().numpy()*0.5+1.0)*255).astype(np.uint8)
                        emb = np.expand_dims(emb, axis=0)
                        emb = np.repeat(emb, repeats=14, axis=0).T
                        emb = np.repeat(emb, repeats=5, axis=0)
                        emb = np.vstack((emb, np.zeros((6, 14), dtype=np.uint8)))
                        comparison = np.hstack((inpS, emb, outS))
                        comparison = cv2.applyColorMap(comparison, cv2.COLORMAP_JET)
                        cv2.imshow("person "+str(index), comparison)
                        new_open_windows.add("person "+str(index))
                        break  # Only show one person

                    for window in open_windows.difference(new_open_windows):
                        cv2.destroyWindow(window)
                    open_windows = new_open_windows
                    cv2.waitKey(1)

                if tensorboard:
                    absI = i + epoch * total_iterations
                    writer.add_scalar("Loss size", lossSize, absI)

            print("Finished epoch {} / {}. Loss size:".format(epoch, epochs, epochLoss))
            self.saveModel(self._modelPath)


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
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(True),
            nn.Linear(100, 784),
            nn.ReLU(True),
            Make2D(14, 14),
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def encode(self, S):
        return self.encoder(S)

    def decode(self, x):
        return self.decoder(x)
