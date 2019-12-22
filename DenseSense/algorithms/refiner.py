import DenseSense.algorithms.Algorithm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("PyTorch running on: "+str(device))


class Refiner(DenseSense.algorithms.Algorithm.Algorithm):
    # UNet, inspired by https://github.com/usuyama/pytorch-unet/
    class MaskGenerator(nn.Module):
        def __init__(self):
            super(Refiner.MaskGenerator, self).__init__()

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )

            self.dconv1down = double_conv(1, 8)
            self.dconv2down = double_conv(8, 16)
            self.dconv3down = double_conv(16, 32)

            self.maxpool = nn.MaxPool2d(2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.fc = nn.Linear(2, 14*14)
            self.relu = nn.ReLU()

            self.dconvup2 = double_conv(16 + 32, 16)
            self.dconvup1 = double_conv(8 + 16, 8)

            self.conv_last = nn.Conv2d(8, 1, 1)

        def forward(self, people):
            # Normalize input
            """
            S = np.zeros_like(person.S, dtype=np.float32)
            S[person.S > 0] = person.S.astype(np.float32)[person.S > 0]/15.0*0.5+0.5
            S = np.array([np.array([S])])
            x = torch.from_numpy(S).float().to(device)
            """

            # Send data to device
            x = torch.Tensor(len(people), 1, 56, 56)
            b = torch.Tensor(len(people), 2)
            for i in range(len(people)):
                person = people[i]
                x[i][0] = torch.from_numpy(person.S)
                bnds = person.bounds
                area = np.power(np.sqrt((bnds[2]-bnds[0])*(bnds[3]-bnds[1])), 0.2)
                if bnds[3] == bnds[1]:
                    aspect = 0
                else:
                    aspect = (bnds[2]-bnds[0])/(bnds[3]-bnds[1])
                b[i] = torch.Tensor([area, aspect])
            x = x.to(device)
            b = b.to(device)

            # Normalize input
            x[0 < x] = x[0 < x]/15.0*0.5+0.5

            # Run model
            conv1 = self.dconv1down(x)
            x = self.maxpool(conv1)
            conv2 = self.dconv2down(x)
            x = self.maxpool(conv2)

            x = self.dconv3down(x)

            y = self.fc(b)
            y = self.relu(y).view(-1, 1, 14, 14)
            x = x+y

            x = self.upsample(x)
            x = torch.cat([x, conv2], dim=1)

            x = self.dconvup2(x)
            x = self.upsample(x)
            x = torch.cat([x, conv1], dim=1)

            x = self.dconvup1(x)
            out = self.conv_last(x)

            return out

    def __init__(self, modelPath=None):
        super().__init__()

        # Generate and maybe load mask generator model
        self.maskGenerator = Refiner.MaskGenerator()
        self.modelPath = modelPath
        if self.modelPath is not None:
            print("Loading Refiner MaskGenerator file from: " + self.modelPath)
            self.maskGenerator.load_state_dict(torch.load(self.modelPath))
            self.maskGenerator.to(device)

        self._training = False
        self._trainingInitiated = False
        self._ROI_masks = torch.Tensor()
        self._ROI_bounds = np.array([])

    def _initTraining(self):
        # Dataset is COCO
        print("Initiating training of Refiner MaskGenerator")
        print("Loading COCO")
        from pycocotools.coco import COCO

        # TODO: specify path from outside class
        dataDir = '.'
        dataType = 'val2017'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

        self.coco = COCO(annFile)
        self.personCatID = self.coco.getCatIds(catNms=['person'])[0]
        self.cocoImageIds = self.coco.getImgIds(catIds=self.personCatID)
        print("Dataset size: {}".format(len(self.cocoImageIds)))

        # Init loss function and optimizer
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.maskGenerator.parameters(), lr=0.0003)

        # Init DensePose extractor
        from DenseSense.algorithms.densepose import DenseposeExtractor
        self.denseposeExtractor = DenseposeExtractor()

    def extract(self, people):
        # Generate masks for all ROIs (people) using neural network model
        self._generateMasks(people)

        # TODO: merge masks and negated masks with segmentation mask from DensePose

        # TODO: find overlapping ROIs and merge the ones where the masks correlate

        # TODO: filter people and update their data

        return people

    def _generateMasks(self, ROIs):
        t1 = time.time()
        self._ROI_masks = self.maskGenerator.forward(ROIs)
        t2 = time.time()
        #print("deltaTime", (t2 - t1)*1000)
        self._ROI_bounds = np.zeros((len(ROIs), 4), dtype=np.int32)
        for i in range(len(ROIs)):
            self._ROI_bounds[i] = np.array(ROIs[i].bounds, dtype=np.int32)

    def train(self):
        self._training = True
        if not self._trainingInitiated:
            self._initTraining()

        Epochs = 100
        Iterations = len(self.cocoImageIds)

        #  Asynchronous downloading of images
        from multiprocessing import Process, Manager
        from urllib.request import urlopen

        DownloadedBufferSize = 6

        manager = Manager()
        downloadedImages = manager.dict()
        downloadJobs = {}

        def downloadImage(d, url):
            resp = urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            d[url] = image

        def bufferImages(index):
            for i in range(index, min(index+DownloadedBufferSize, Iterations)):
                coco_img = self.coco.loadImgs(self.cocoImageIds[i])[0]
                url = coco_img["coco_url"]
                if url not in downloadJobs:
                    downloadJobs[url] = Process(target=downloadImage, args=(downloadedImages, url))
                    downloadJobs[url].start()

        print("Starting training")

        for epoch in range(Epochs):
            print("Starting epoch {} out of {}".format(epoch, Epochs))
            for i in range(Iterations):
                bufferImages(i)

                # Load instance of COCO dataset
                cocoImg = self.coco.loadImgs(self.cocoImageIds[i])[0]
                url = cocoImg["coco_url"]
                downloadJobs[url].join()  # Wait for image to be downloaded
                image = downloadedImages[url]

                # Get annotation
                annIds = self.coco.getAnnIds(imgIds=cocoImg["id"], catIds=self.personCatID, iscrowd=None)
                annotation = self.coco.loadAnns(annIds)

                # Draw each person in annotation to separate mask
                segs = []
                for person in annotation:
                    mask = np.zeros(image.shape[0:2])
                    for s in person["segmentation"]:
                        if s != "counts":  # FIXME: why is s sometimes "counts"?
                            s = np.reshape(np.array(s, dtype=np.int32), (-2, 2))
                            cv2.fillPoly(mask, [s], 255)
                        else:
                            print(person["id"])
                    segs.append(mask)

                # Run DensePose extractor
                people = self.denseposeExtractor.extract(image)
                #image = self.denseposeExtractor.renderDebug(image, people)

                # Run prediction
                self._generateMasks(people)
                masks = self._ROI_masks
                image = self.renderDebug(image)

                # TODO: get correlation where there is overlap between COCO-mask for each person and predictions for

                # TODO: optimize model to maximize/minimize correlations
                plt.ion()
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.draw()
                plt.pause(0.01)

                # Delete this image from what's downloaded
                del downloadedImages[url]
                del downloadJobs[url]

        self._training = False

    def renderDebug(self, image):
        # Normalize ROIs from (0, 1) to (0, 255)
        ROIsMaskNorm = self._ROI_masks*255

        # Render masks on image
        for i in range(len(self._ROI_masks)):
            mask = ROIsMaskNorm[i, 0].cpu().detach().to(torch.uint8).numpy()
            bnds = self._ROI_bounds[i]

            raw = self._ROI_masks.cpu().detach().numpy()

            # Change colors of mask
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)

            # TODO: render contours instead
            # Resize mask to bounds
            dims = (bnds[2] - bnds[0], bnds[3] - bnds[1])
            mask = cv2.resize(mask, dims, interpolation=cv2.INTER_AREA)

            # Overlay image
            alpha = 0.65
            overlap = image[bnds[1]:bnds[3], bnds[0]:bnds[2]]
            mask = mask * alpha + overlap * (1.0 - alpha)
            image[bnds[1]:bnds[3], bnds[0]:bnds[2]] = mask

        return image
