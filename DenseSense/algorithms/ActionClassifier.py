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
    AVA_Datasets = ["ava_val", "ava_train", "ava_val_predictive", "ava_train_predictive"]

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

        from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
        from DenseSense.algorithms.Sanitizer import Sanitizer
        from DenseSense.algorithms.Tracker import Tracker

        self.denseposeExtractor = DensePoseWrapper()
        self.sanitizer = Sanitizer()
        self.sanitizer.loadModel(topDir + "/models/Sanitizer.pth")
        self.tracker = Tracker()

        if datasetName in ActionClassifier.COCO_Datasets:
            print("Loading COCO dataset: "+datasetName)
            from pycocotools.coco import COCO
            from os import path

            annFile = topDir + '/annotations/instances_{}.json'.format(datasetName)
            self.cocoPath = topDir + '/data/{}'.format(datasetName)

            self.coco = COCO(annFile)
            personCatID = self.coco.getCatIds(catNms=['person'])[0]
            self.dataset = self.coco.getImgIds(catIds=personCatID)

        elif datasetName in ActionClassifier.AVA_Datasets:
            print("Loading AVA dataset: "+datasetName)
            import csv
            from collections import defaultdict
            from DenseSense.utils.YoutubeLoader import  YoutubeLoader

            annFile = topDir + '/annotations/{}.csv'.format(datasetName)
            self.dataset = defaultdict(lambda: defaultdict(defaultdict))
            with open(annFile, 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    video, t, x1, y1, x2, y2, action, person = row
                    actions = {action}
                    if person in self.dataset[video][t]:
                        actions = actions.union(self.dataset[video][t][person]["actions"])
                    self.dataset[video][t][person] = {
                        "bbox": (x1, y1, x2, y2),
                        "actions": actions
                    }

            ordered_data = []
            for key, video in self.dataset.items():
                ordered_data.append((key, []))
                for t, annotation in video.items():
                    ordered_data[-1][1].append((int(t), annotation))
                ordered_data[-1][1].sort(key=lambda x: x[0])

            self.dataset = ordered_data

            self.youtubeLoader = YoutubeLoader()
            for key, video in self.dataset:
                self.youtubeLoader.queue_video(key, video[0][0], video[-1][0])

            self.current_video_index = 0
            self.current_video_frame_index = 0
        else:
            raise Exception("Unknown dataset")

        self.useLMDB = useLMDB
        if useLMDB:
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
                if image is None:
                    print(cocoImage)
                    raise Exception("Could not find image: "+str(index))

                people = self.denseposeExtractor.extract(image)
                people = self.sanitizer.extract(people)
                if self.useLMDB:
                    self.lmdb.save("DensePoseWrapper_Sanitized_Coco", str(cocoImage["id"]), people)
            return people, cocoImage

        elif self.datasetName in ActionClassifier.AVA_Datasets:
            data = None
            people, frameTime = None, None
            key = self.dataset[self.current_video_index][0]
            self.youtubeLoader.verbose = True

            if self.useLMDB:
                data = self.lmdb.get("DensePoseWrapper_Sanitized_AVA",
                                     str(key)+"_"+str(self.current_video_frame_index))

            if data is None:
                print("generating")
                image, yt_key, yt_videoIndex, yt_frameIndex, frameTime = next(self.youtubeLoader.frames())

                people = self.denseposeExtractor.extract(image)
                people = self.sanitizer.extract(people)

                if self.useLMDB:
                    self.lmdb.save("DensePoseWrapper_Sanitized_AVA",
                                   str(key) + "_" + str(self.current_video_frame_index), (people, frameTime))

                if key != yt_key:
                    raise Exception("YoutubeLoader and ActionClassifier video keys don't match")

                if self.current_video_frame_index != yt_frameIndex:
                    raise Exception("YoutubeLoader and ActionClassifier frame indexes don't match")

            else:
                people, frameTime = data
                self.youtubeLoader.blocking = False # TODO: implement
                _, yt_key, _, yt_frameIndex, _ = next(self.youtubeLoader.frames())
                print("skip a")
                while key != yt_key and self.current_video_frame_index != yt_frameIndex:
                    print("skip b")
                    _, yt_key, _, yt_frameIndex, _ = next(self.youtubeLoader.frames())
                self.youtubeLoader.blocking = True

            timestamp = np.round(frameTime)
            avaAnn = None

            sameTimestamp = [v[1] for v in self.dataset[self.current_video_index][1] if v[0] == timestamp]
            print("st")
            if len(sameTimestamp) == 1:
                avaAnn = sameTimestamp[0]
                print(avaAnn)
            else:
                print("Found no ava ann")

            # Change increment video and frame
            self.current_video_frame_index += 1
            if len(self.dataset[self.current_video_index]) == self.current_video_frame_index:
                self.current_video_frame_index = 0
                self.current_video_index += 1
                if len(self.dataset[self.current_video_index]) == self.current_video_index:
                    self.current_video_index = 0

            return people, avaAnn

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

        # For predictive coding training
        current_video = None
        last_time = None

        print("Starting training")
        for epoch in range(epochs):
            epochLoss = np.float64(0)
            for i in range(total_iterations):
                people, annotation = self._load(i)
                if "predictive" in self.datasetName:
                    # Load next
                    nextS = 0

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
                if "predictive" in self.datasetName:
                    lossSize = self.loss_function(out, nextS)
                else:
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
                        emb = np.repeat(emb, repeats=10, axis=0)
                        emb = np.vstack((emb, np.zeros((56-5*10, 14), dtype=np.uint8)))
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
            nn.Linear(100, 5)
        )

        self.decoder = nn.Sequential(
            nn.Linear(5, 100),
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

    def decode(self, x, delta_time=0):
        # TODO: concat delta time
        return self.decoder(x)


if __name__ == '__main__':
    # Experimentation mode
    from argparse import ArgumentParser
    parser = ArgumentParser("Experimentation with trained models")
    parser.add_argument("mode", help="Mode", choices=["AutoEncoder"], type=str)
    parser.add_argument("-ae", "--ae_model", help="AutoEncoder model path", default="./models/", type=str)
    args = parser.parse_args()

    if args.mode in ["AutoEncoder"]:
        modelPath = args.ae_model
        if os.path.isdir(modelPath):
            modelPath = os.path.join(modelPath, "ActionClassifier_AutoEncoder.pth")
        alreadyExists = os.path.exists(modelPath)

        ae = AutoEncoder()
        ae.load_state_dict(torch.load(modelPath, map_location=device))
        ae.to(device)

        if args.mode == "AutoEncoder":
            embedding = torch.zeros((1, 5)).to(device)
            selected = 0
            while True:
                out = ae.decode(embedding)
                outS = (out[0, 0].detach() * 255).cpu().to(torch.uint8).numpy()
                emb = ((embedding[0].detach().cpu().numpy() * 0.5 + 1.0) * 255).astype(np.uint8)
                emb = np.expand_dims(emb, axis=0)
                emb = np.repeat(emb, repeats=14, axis=0).T
                emb = np.repeat(emb, repeats=10, axis=0)
                emb = np.vstack((emb, np.zeros((56-5*10, 14), dtype=np.uint8)))
                comparison = np.hstack((emb, outS))
                comparison = cv2.applyColorMap(comparison, cv2.COLORMAP_JET)
                cv2.imshow("Chosen embedding to person", comparison)

                k = cv2.waitKey(0)

                if k == 27:
                    break  # Esc

                if 48 <= k <= (48+embedding.shape[1]-1):
                    selected = k - 48
                    print("Selected {}".format(selected))

                if k == 82 or k == 119:
                    embedding[0, selected] += 0.05
                    print(embedding[0])

                if k == 84 or k == 115:
                    embedding[0, selected] -= 0.05
                    print(embedding[0])
