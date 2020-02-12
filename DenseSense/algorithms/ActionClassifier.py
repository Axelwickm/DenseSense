import json
import os

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

    avaFiltered = {}
    classCursors = {}

    iteration = 0

    db = None
    avaGenerator = None
    epoch = 0
    processedImagesThisEpoch = 0

    class Network(nn.Module):
        def __init__(self, outputs):
            super(ActionClassifier.Network, self).__init__()

            self.lstm = nn.LSTM(25*2, 10)
            # TODO: activation function?
            self.linear = nn.Linear(10, outputs)

        def forward(self, features):
            out = self.lstm(features)
            out = self.linear(out)
            return out

    net = None
    loss_function = None
    optimizer = None
    currentPeople = {}

    classCursors = None

    def __init__(self):
        print("Initiating ActionClassifier")
        super().__init__()

        actionIDs = self.actions.keys()
        classCount = len(set(self.actions.values()))

        self._modelPath = None
        self.model = None
        self._training = False

    def loadModel(self, modelPath):
        self._modelPath = modelPath
        print("Loading ActionClassifier file from: " + self._modelPath)
        self.model.load_state_dict(torch.load(self._modelPath, map_location=device))
        self.model.to(device)

    def saveModel(self, modelPath):
        if modelPath is None:
            print("Don't know where to save model")
        self._modelPath = modelPath
        print("Saving ActionClassifier model to: "+self._modelPath)
        torch.save(self.model.state_dict(), self._modelPath)

    def _initTraining(self, learningRate, datasetName, useLMDB):
        self.datasetName = datasetName

        if datasetName is "Coco":
            from torchvision import transforms
            from torchvision.datasets import CocoDetection

            from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
            from DenseSense.algorithms.Sanitizer import Sanitizer

            annFile = topDir + '/annotations/instances_{}.json'.format(datasetName)
            cocoPath = topDir + '/data/{}'.format(datasetName)

            self.dataset = CocoDetection(cocoPath, annFile, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),
                transforms.Lambda(lambda x: (x * 255).byte().numpy()),
                transforms.Lambda(lambda x: x[:, :, ::-1])
            ]))

            self.denseposeExtractor = DensePoseWrapper()
            self.sanitizer = Sanitizer()
            self.sanitizer.loadModel(topDir + "/models/Sanitizer.pth")

        if useLMDB:
            self.useLMDB = True
            self.lmdb = LMDBHelper("a")
            self.lmdb.verbose = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate, ansgrad=True)
        self.loss_function = torch.nn.MSELoss()

    def _load(self, index):
        if self.datasetName is "Coco":
            cocoImage = self.dataset[index]
            if self.useLMDB:
                people = self.lmdb.get(ActionClassifier, "coco" + str(index))

            if people is None:
                ROIs = self.denseposeExtractor.extract(cocoImage[0])
                ROIs = self.sanitizer.extract(ROIs)
                if self.useLMDB:
                    self.lmdb.save(ActionClassifier, "coco" + str(index), ROIs)

    def train(self, epochs=100, learningRate=0.005, dataset="Coco",
              useLMDB=True, printUpdateEvery=40,
              visualize=False, tensorboard=False):

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
        Iterations = len(self.dataset)

        print("Starting training")
        for epoch in range(epochs):
            epochLoss = np.float64(0)
            for i in range(Iterations):
                predictions = None
                groundtruth = None

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
