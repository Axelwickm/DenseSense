import json
import os
from copy import copy

import cv2

import DenseSense.algorithms.Algorithm
import DenseSense.utils.YoutubeLoader
from DenseSense.utils.LMDBHelper import LMDBHelper
from DenseSense.algorithms.Tracker import Tracker

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

    def loadModel(self, modelPath):  # TODO: load multiple models, refactor name
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

        self.denseposeExtractor = DensePoseWrapper()
        self.sanitizer = Sanitizer()
        self.sanitizer.loadModel(topDir + "/models/Sanitizer.pth")

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
            from DenseSense.utils.YoutubeLoader import YoutubeLoader

            annFile = topDir + "/annotations/{}.csv".format(datasetName.replace("_predictive", ""))
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

            self.youtubeLoader = YoutubeLoader(verbose=False)
            for key, video in self.dataset:
                self.youtubeLoader.queue_video(key, video[0][0], video[-1][0])

            self.current_video_index = 0
            self.current_video_frame_index = 0

            self.tracker = Tracker()
        else:
            raise Exception("Unknown dataset")

        self.useLMDB = useLMDB
        if useLMDB:
            self.lmdb = LMDBHelper("a", max_size=1028*1028*1028*32)
            self.lmdb.verbose = False

        self.optimizer = torch.optim.Adam(self._AE_model.parameters(), lr=learningRate)
        self.loss_function = torch.nn.BCELoss()

    def _load(self, index=None):  # Load next if index is None
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
            image = None
            people, frame_time, is_last = None, None, False
            key = self.dataset[self.current_video_index][0]

            if self.useLMDB:
                data = self.lmdb.get("DensePoseWrapper_Sanitized_AVA",
                                     str(key)+"_"+str(self.current_video_frame_index))

            if data is None:
                image, frame_time, is_last = self.youtubeLoader.get(self.current_video_index,
                                                                    self.current_video_frame_index)
                if image is None:
                    people = []
                    frame_time = 0
                else:
                    people = self.denseposeExtractor.extract(image)
                    people = self.sanitizer.extract(people)

                if self.useLMDB:  # Save processed data
                    self.lmdb.save("DensePoseWrapper_Sanitized_AVA",
                                   str(key) + "_" + str(self.current_video_frame_index), (people, frame_time, is_last))

            else:
                people, frame_time, is_last = data

            timestamp = np.round(frame_time)
            ava_annotation = None

            sameTimestamp = [v[1] for v in self.dataset[self.current_video_index][1] if v[0] == timestamp]
            if len(sameTimestamp) == 1:
                ava_annotation = sameTimestamp[0]

            # To show the whole dataset as it's being downloaded
            if image is not None and True:
                if ava_annotation is not None:
                    for k, p in ava_annotation.items():
                        bbox = np.array([float(p["bbox"][0]), float(p["bbox"][1]),
                                         float(p["bbox"][2]), float(p["bbox"][3])])
                        p1 = bbox[:2] * np.array([image.shape[1], image.shape[0]], dtype=np.float)
                        p2 = bbox[2:] * np.array([image.shape[1], image.shape[0]], dtype=np.float)
                        image = cv2.rectangle(image, tuple(p1.astype(np.int32)), tuple(p2.astype(np.int32)),
                                              (20, 20, 200), 1)
                cv2.imshow("frame", image)
                cv2.waitKey(1)

            # Change increment video and frame
            if is_last:
                self.current_video_frame_index = 0
                self.current_video_index += 1
                if len(self.dataset) == self.current_video_index:
                    self.current_video_index = 0
            else:
                self.current_video_frame_index += 1

            return people, frame_time, is_last, ava_annotation

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

        def tensorify(p, last=False):
            S = torch.Tensor(len(p), 1, 56, 56)
            for j in range(len(p)):
                person = p[j]
                if last:
                    S[j][0] = torch.from_numpy(person.S_last)
                else:
                    S[j][0] = torch.from_numpy(person.S)

            S = S.to(device)
            S[0 < S] = S[0 < S] / 15.0 * 0.8 + 0.2
            return S

        def get_debug_image(index, S, embedding, out, S_next=None): # TODO: move to render debug
            inpS = (S[index, 0].detach() * 255).cpu().to(torch.uint8).numpy()
            outS = (out[index, 0].detach() * 255).cpu().to(torch.uint8).numpy()
            emb = ((embedding[index].detach().cpu().numpy() * 0.5 + 1.0) * 255).astype(np.uint8)
            emb = np.expand_dims(emb, axis=0)
            emb = np.repeat(emb, repeats=14, axis=0).T
            emb = np.repeat(emb, repeats=10, axis=0)
            emb = np.vstack((emb, np.zeros((56 - 5 * 10, 14), dtype=np.uint8)))
            comparison = np.hstack((inpS, emb, outS))
            if S_next is not None:
                Sn = (S_next[index, 0].detach() * 255).cpu().to(torch.uint8).numpy()
                Sn = np.hstack((np.zeros((56, 56 + 14)), Sn))
                comparison = np.vstack((comparison, Sn)).astype(np.uint8)

            return cv2.applyColorMap(comparison, cv2.COLORMAP_JET)

        if self.datasetName in ActionClassifier.COCO_Datasets:
            print("Starting COCO dataset training")
            for epoch in range(epochs):
                epochLoss = np.float64(0)
                for i in range(total_iterations):
                    people, annotation = self._load(i)
                    S = tensorify(people)

                    if S.shape[0] == 0:
                        continue

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
                        for index, _ in enumerate(S):
                            debug_image = get_debug_image(index, S, embedding, out)
                            cv2.imshow("person " + str(index), debug_image)
                            new_open_windows.add("person " + str(index))
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

        elif self.datasetName in ActionClassifier.AVA_Datasets:
            # Unfortunately, needs to run through the whole AVA dataset to determine the size in frames
            print("Going through ava dataset once to determine the size")
            total_iterations = 0
            for video_i in range(len(self.dataset)):
                is_last = False
                while not is_last:
                    people, frame_time, is_last, annotation = self._load()  # Load next
                    total_iterations += 1
                    if (total_iterations - 1) % 500 == 0:
                        print("Frame/iteration {} (video {} / {})".format(total_iterations, video_i, len(self.dataset)))
                if 20 <= video_i:
                    break
            print("Total number of iterations are {}".format(total_iterations))

            print("Starting AVA dataset training")
            last_frame_time = None
            last_people = []
            S_next = None
            current_video = 0
            was_last = False
            for epoch in range(epochs):
                epochLoss = np.float64(0)
                for i in range(total_iterations):
                    people, frame_time, is_last, annotation = self._load()  # Load next
                    current_video += is_last

                    if "predictive" in self.datasetName:
                        # Track the next frame
                        self.tracker.extract(people, time_now=frame_time)
                        if is_last:  # If new video next
                            self.tracker = Tracker()
                            last_frame_time = None

                        # Only save the people who exist in all frames
                        old_ids = list(map(lambda p: p.id, last_people))
                        new_ids = list(map(lambda p: p.id, people))

                        old_people = list(filter(lambda p: p.id in new_ids, last_people.copy()))
                        new_people = list(filter(lambda p: p.id in old_ids, people.copy()))

                        # Filter old Ss
                        S = tensorify(old_people, True)
                        S_next = tensorify(new_people, False)

                        last_people = people
                    else:
                        frame_time = last_frame_time
                        S = tensorify(people)

                    if S.shape[0] == 0:
                        continue

                    delta_time = 0
                    if last_frame_time is not None and was_last is False:
                        delta_time = frame_time - last_frame_time
                    last_frame_time = frame_time

                    # Run prediction
                    embedding = self._AE_model.encode(S, delta_time)
                    out = self._AE_model.decode(embedding)

                    # Optimize
                    if "predictive" in self.datasetName:
                        lossSize = self.loss_function(out, S_next)
                    else:
                        lossSize = self.loss_function(out, S)

                    lossSize.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lossSize = lossSize.cpu().item()

                    # Give feedback of training process
                    epochLoss += lossSize / total_iterations
                    visualize_counter += 1
                    was_last = is_last
                    if (i - 1) % printUpdateEvery == 0:
                        print("Iteration {} / {} (video {}/{}), epoch {} / {}".format(i, total_iterations,
                                                                                      current_video,
                                                                                      len(self.dataset),
                                                                                      epoch, epochs))
                        print("Loss size: {}\n".format(lossSize / printUpdateEvery))

                    if visualize != 0 and visualize <= visualize_counter:
                        visualize_counter = 0
                        new_open_windows = set()
                        for index, _ in enumerate(S):
                            debug_image = get_debug_image(index, S, embedding, out, S_next)
                            cv2.imshow("person " + str(index), debug_image)
                            new_open_windows.add("person " + str(index))
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
        self.encoder_first = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            Flatten()
        )

        self.encoder_second = nn.Sequential(
            nn.Linear(784+1, 100),  # +1 is delta time
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

    def encode(self, S, delta_time=None):
        if delta_time is None:
            delta_time = np.random.rand()*0.5
        # Normalize and concatenate the delta time to every vector in tensor
        delta_time = min(max(0, delta_time), 10)
        delta_time = delta_time / (np.power(delta_time, 0.8) + 0.3)
        delta_time = torch.Tensor([delta_time])
        delta_times = torch.cat(S.shape[0] * [delta_time]).reshape((S.shape[0], -1))
        # Run model
        x = self.encoder_first(S)
        x = torch.cat([x, delta_times], 1).to(device)
        x = self.encoder_second(x)
        return x

    def decode(self, x):
        # Do decoding
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
