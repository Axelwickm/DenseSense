#!/usr/bin/env python2

import json
import time
import traceback
from math import sqrt
from stat import *
import csv

from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color
import cv2
import numpy as np
import pafy
import ffmpeg

import Algorithms
from LMDB_helper import NumpyEncoder


trainSettings = {
    "algorithms" : [],
    "auxiliaryDB" : "",
    "override" : [],
    "saveToFile" : False 
}    


algorithms = []
db = None

# TODO: communication into this so it can stop
def trainingLoop(PDS):
    startTrainingTime = time.time()

    iterations = 55000*20
    printEvery = 100
    saveModelEvery = 4999
    averageLoss = {}
    averageTime = {}
    for alg in algorithms:
        averageLoss[alg.name] = 0.0
        averageTime[alg.name] = 0.0

    statistics = ""
    for i in xrange(0, iterations): # TODO: put this in .train of each algorithm
        epoch = 0
        imageInEpoch = 0
        for alg in algorithms:
            if PDS.DP is None: 
                PDS.DP = Algorithms.Densepose()
            if PDS.PE is None:
                PDS.PE = Algorithms.People_Extractor()
            iteration, loss, extractTime = alg.train(i % saveModelEvery == 0, algorithms={"DP":PDS.DP, "PE":PDS.PE})

            if alg.debugOutput is not None:
                debug_mapfile.seek(0)
                debug_mapfile.write(alg.debugOutput)
                server.broadcast(json.dumps({
                    "type":"showDebug",
                    "windows" : [{
                        "w": alg.debugOutput.shape[0], "h": alg.debugOutput.shape[1],
                        "name":"UV texture"
                    }]}))

            averageLoss[alg.name] += loss
            averageTime[alg.name] += extractTime

            if (i+1) % (printEvery+1) == 0:
                statistics += "Algorithm \"{}\":    training loss: {:.4f}, average time: {:.0f} ms\n".format(
                    alg.name,
                    averageLoss[alg.name] / printEvery,
                    averageTime[alg.name] / printEvery * 1000 )
                averageLoss[alg.name] = 0
                averageTime[alg.name] = 0

                epoch = alg.epoch
                imageInEpoch = alg.processedImagesThisEpoch
            
            if (i+1) % (printEvery+1) == 0:
                    statistics += "Has been runnning for {} iterations and {:.2f} seconds\n".format(i, time.time() - startTrainingTime)
                    statistics += "Current epoch {} and image number {}\n\n".format(epoch, imageInEpoch)
                    server.broadcast(json.dumps({"type":"trained", "statistics":statistics}))
                    statistics = ""


clients = []

class PoseDetectorServer():
        is_processing = False

        DP = None
        PE = None
        PT = None
        UV = None
        DE = None

        DEFilter = {}

        debug = True
        debugUVCount = 5
        debugUVPositions = [[] for _  in range(debugUVCount)]
        uvDim = int(sqrt(25 * 64 * 64))
        debugUV = np.zeros((uvDim, uvDim*debugUVCount, 3), np.uint8)

        debugView = {
            "image" : True,
            "densepose" : False,
            "bboxes" : True, 
            "centers" : True,
            "ecenters" : False,
            "tracker" : False,
            "uvs" : True,
            "labels" : False
        }

        def got_image(self, image):            
            if self.PE is None:
                self.DP = Algorithms.Densepose()
                self.PE = Algorithms.People_Extractor()
                self.PT = Algorithms.People_Tracker()
                self.UV = Algorithms.UV_Extractor()
                self.DE = Algorithms.DescriptionExtractor("/shared/DescriptionFinal.model")
            
            # Do processing
            self.is_processing = True
            UVs = None
            labelVector = None
            try:
                image = white_balance(image)

                boxes, bodies = self.DP.extract(image)
                people, mergedIUVs = self.PE.extract(boxes, bodies, image)
                self.PT.extract(people)
                
                UVs = self.UV.extract(people, mergedIUVs, image)
                labelVector = self.DE.extract(UVs)

                # TODO: pose/motion features
                # TODO: feature classification

                # Respond with processed data
                for i in range(len(people)):
                    if labelVector[i] is not None:
                        people[i]["wearing"] = labelVector[i]

                message = json.dumps({"type":"processed", "data":{
                    "seen":people
                }}, cls=NumpyEncoder)

                self.sendMessage(message)
            
                # Debug visualizations
                if self.debug:
                    global debug_mapfile, server

                    self.debugView["densepose"] = False if mergedIUVs is None else self.debugView["densepose"]
                    self.debugView["centers"] = False if self.PE.debugOutput is None else self.debugView["centers"]
                    self.debugView["ecenters"] = False if self.PE.debugOutput is None else self.debugView["ecenters"]
                    self.debugView["tracker"] = False if self.PT.debugOutput is None else self.debugView["tracker"]
                    self.debugView["uvs"] = False if UVs is None else self.debugView["uvs"]
                    self.debugView["labels"] = False if labelVector is None else self.debugView["labels"]

                    def getColor(id):
                        color = HSVColor(id*60 % 360, 150, 200) 
                        return np.array(convert_color(color, sRGBColor).get_value_tuple())/255.0 # FIXME: This is bgr color

                    if not self.debugView["image"]:
                        image = np.zeros_like(image, np.uint8)

                    if self.debugView["densepose"]:
                        personOverlay = np.dstack((50+mergedIUVs[0]*8, mergedIUVs[1]*255, mergedIUVs[2]*255))
                        personOverlayGuide = np.dstack((mergedIUVs[0], mergedIUVs[0], mergedIUVs[0]))
                        image = np.where(personOverlayGuide, personOverlay, image)

                    image = image.astype(np.int64)

                    # Draw detected people supressed by tracker
                    if self.debugView["tracker"]:
                        for ghostBound in self.PT.debugOutput[0]:
                            x1 = int(ghostBound[0])
                            x2 = int(ghostBound[2])
                            y1 = int(ghostBound[1])
                            y2 = int(ghostBound[3])
                            g = np.clip(image[y1:y2:2, x1:x2:2] - np.array([50, 50, 50]), 0, 255)
                            image[y1:y2:2, x1:x2:2] = g

                    # Densepose view
                    for i in range(len(people)):
                        person = people[i]
                        # Draw person bounding box
                        x1 = int(person["bounds"][0])
                        x2 = int(person["bounds"][2])
                        y1 = int(person["bounds"][1])
                        y2 = int(person["bounds"][3])
                        color = getColor(person["id"])

                        if self.debugView["tracker"]:
                            s = 2 if person["index"] in self.PT.debugOutput[0] else 1 # If this is true, the person is being kept alive by the tracker
                        else:
                            s = 1
                        
                        if self.debugView["bboxes"]:
                            g = np.clip(image[y1:y2:s, x1:x2:s] + color, 0, 255)
                            image[y1:y2:s, x1:x2:s] = g
                        
                        if self.debugView["centers"]: # of body parts
                            for bodypart in person["bodyparts"]:
                                lower = bodypart[1] - np.array([2.0, 2.0])
                                upper = bodypart[1] + np.array([2.0, 2.0])
                                lower = lower.clip(min=0)
                                upper[0] = upper[0].clip(max=image.shape[1])
                                upper[1] = upper[1].clip(max=image.shape[0])
                                image[int(lower[1]):int(upper[1]), int(lower[0]):int(upper[0])] = 255
                        
                        if self.debugView["ecenters"]:
                            # Draw person bodypart centers adjusted by people extractor
                            if person["index"] in self.PE.debugOutput: # Might have been added in by tracker
                                for bodypart in self.PE.debugOutput[person["index"]]:
                                    lower = bodypart - np.array([1.0, 1.0])
                                    upper = bodypart + np.array([1.0, 1.0])
                                    lower = lower.clip(min=0)
                                    upper[0] = upper[0].clip(max=image.shape[1])
                                    upper[1] = upper[1].clip(max=image.shape[0])
                                    image[int(lower[1]):int(upper[1]), int(lower[0]):int(upper[0])] = color

                    image = image.astype(np.uint8)

                    if self.debugView["labels"]:
                        # Draw clothing labels
                        for i in range(len(people)):
                            if labelVector[i] is None:
                                continue
                            
                            person = people[i]
                            so = sorted(labelVector[i].iteritems(), key=lambda x: -x[1]["activation"])
                            so = so[:2]
                            
                            for j in range(len(so)):
                                s = so[j]

                                text = s[0]+": "+str(s[1]["activation"])
                                if "color" in s[1]:
                                    text += " "+s[1]["color"]

                                pos = np.array(person["bounds"][:2].tolist(), np.int64)
                                pos[1] += j*10
                                pos = tuple(pos)
                                cv2.putText(image, text, pos, cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)    

                    # Draw predicted position
                    if self.debugView["tracker"]:
                        for pred in self.PT.debugOutput[2]:
                            pos = tuple(pred[1].astype(np.int32))
                            col = (100, 100, 100)
                            if pred[2]:
                                col = (220, 220, 220)
                            cv2.drawMarker(image, pos, col, markerSize=10)
                    
                    debug_mapfile.seek(0)
                    debug_mapfile.write(image)

                    message = {
                        "type":"showDebug", "windows" : [{
                            "w": image.shape[0], "h": image.shape[1],
                            "name":"Pose Detection debug"
                        }]
                    }

                    if self.debugView["uvs"]:
                        # UV Textures
                        inds = np.full(len(UVs), -1)
                        vacant = np.full(self.debugUVCount, True)
                        homeless = []
                        
                        for i in range(len(UVs)):
                            if UVs[i] is None:
                                continue
                            for j in range(len(self.debugUVPositions)):
                                if self.debugUVPositions[j] == people[i]["id"]:
                                    inds[i] = j
                                    vacant[j] = False
                                    break
                            else:
                                # Wasn't in list
                                homeless.append(i)
                        
                        # Biggest areas are visualized
                        def findArea(i):
                            bbox = people[i]["bounds"]
                            area = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0])
                            return area
                        homeless = sorted(homeless, key=findArea)
                        for h in homeless:
                            v = np.where(vacant)[0]
                            if v.size == 0:
                                break # All vacant filled
                            v = v[0]
                            vacant[v] = False
                            inds[h] = v
                            self.debugUVPositions[v] = people[h]["id"]
                        
                        for i in range(min(len(UVs), self.debugUVCount)):
                            ind = inds[i]
                            if UVs[i] is None or UVs[i] == []:
                                continue
                            
                            try:
                                self.debugUV[:, int(self.uvDim*ind):int(self.uvDim*ind+self.uvDim), :] = UVs[i][:, :]
                                color = getColor(people[i]["id"])
                                dim = 20
                                self.debugUV[0:dim, int(self.uvDim*ind):int(self.uvDim*ind+dim)] = color
                            except Exception as e:
                                print("Error while broadcasting debug UVs")

                        for i in range(len(vacant)):
                            if vacant[i]:
                                self.debugUV[:, int(self.uvDim*i):int(self.uvDim*i+self.uvDim), :] /= 2

                        debug_mapfile.write(self.debugUV)

                        message["windows"].append({
                            "w": self.uvDim, "h": self.uvDim * self.debugUVCount,
                            "name":"UV Textures"
                        })
                    
                    server.broadcast(json.dumps(message))
                
                self.is_processing = False
            
            except Exception as e:
                self.is_processing = False
                traceback.print_exc()
                #raise e
            

        def handleMessage(self):
            try:
                # Process the image
                message = json.loads(self.data)
                if message["type"] == "upd":
                    if not self.is_processing:
                        mapfile.seek(0)
                        img = np.frombuffer(mapfile, np.uint8).reshape(240, 320, 3)
                        self.got_image(img)
                    else:
                        print("Did not have time to process frame.")
                    
                    for what in message["tog"]:
                        print("Debug rendering \""+what+"\": "+str(not self.debugView[what]))
                        self.debugView[what] = not self.debugView[what]
                        
                elif message["type"] == "sendTrainSettings":
                    from LMDB_helper import LMDB_helper, PhotoData
                    global db, algorithms, trainSettings, trainingTread
                    trainSettings = message["settings"]

                    print("Setting training settings: "+json.dumps(trainSettings, indent=4))
                    path = "/shared/"+trainSettings["auxiliaryDB"]

                    if trainSettings["auxiliaryDB"] != "":
                        # Initiate database
                        db = LMDB_helper(path, trainSettings["saveToFile"],
                            trainSettings["override"], trainSettings["loadedLimitMB"])
                        
                        # TODO: move these to Datasets.py
                        def getModanet(key):
                            loadedData = db.getAllLoaded()
                            if loadedData["modanet"] == {}:
                                print("Loading dataset modanet annotations from modanet2018_instances_train.json")
                                with open("/shared/modanet/annotations/modanet2018_instances_train.json", 'r') as f:
                                     modanet_raw = json.load(f) 
                                
                                modanet = {}
                                for a in modanet_raw["annotations"]:
                                    if a["image_id"] not in modanet:
                                        modanet[a["image_id"]] = [a]
                                    else:
                                        modanet[a["image_id"]].append(a)

                                loadedData["modanet"] = modanet
                                print("ModaNet annotates "+str(len(modanet))+" different images.")
                            
                            return "NOTHING"

                        def getModanetImage(key):
                            loadedData = db.getAllLoaded() 
                            if loadedData["modanetImages"] == {}:
                                print("Creating modanetImages accessor")
                                loadedData["modanetImages"] = PhotoData('/shared/photos.lmdb')
                            
                            if key == "":
                                return "NOTHING"
                            else:
                                return None
                        
                        def genDensepose(key):
                            if self.DP is None:
                                self.DP = Algorithms.Densepose()
                            image = db.getData("modanetImages", key)
                            data = self.DP.extract(image)
                            db.saveData("densepose", key, list(data), ["nparray", ["ndarray"]*len(data[1])])
                            return data

                        def genPeople(key):
                            if self.PE is None:
                                self.PE = Algorithms.People_Extractor("/shared/bodyMetrics.json")
                            image = db.getData("modanetImages", key)
                            boxes, bodies = db.getData("densepose", key)
                            data = self.PE(boxes, bodies, image)
                            return data
                        
                        def genUV_Textures(key):
                            if self.PE is None:
                                self.PE = Algorithms.People_Extractor("/shared/bodyMetrics.json")
                            if self.UV is None:
                                self.UV = Algorithms.UV_Extractor()
                            image = db.getData("modanetImages", key)
                            image = white_balance(image)
                            boxes, bodies = db.getData("densepose", key)
                            people, mergedIUVs = self.PE.extract(boxes, bodies, image)
                            data = self.UV.extract(people, mergedIUVs, image, 64) # This 64 is prob suppose to be set by algortihm, but oh well
                            db.saveData("UV_Textures", key, data, "ndarray")
                            return data

                        def getAva(actionIDs):
                            ava = []
                            print("Loading dataset modanet annotations from ava_train_v2.2.csv")
                            with open("/shared/AVA/ava_train_v2.2.csv", 'r') as f:
                                AvaFile = csv.reader(f)
                                for row in AvaFile:
                                    ava.append({
                                        "id" : row[0],
                                        "timestamp" : row[1],
                                        "bbox" : np.array([row[2], row[3], row[4], row[5]]),
                                        "label" : row[6],
                                        "identifier" : row[7]
                                    })
                            
                            db.loadedData["ava"] = []
                            shortestTime = 1 # seconds
                            vids = set()

                            relevant = []
                            currentVideoID = None
                            for i in xrange(len(ava)):
                                if currentVideoID is None:
                                    currentVideoID = ava[i]["id"]
                                elif currentVideoID != ava[i]["id"] or i == len(ava)-1:
                                    currentVideoID = None
                                    # Process the last video
                                    if shortestTime <= len(relevant):
                                        clip = {
                                            "key" : relevant[0]["id"],
                                            "startTime" : relevant[0]["timestamp"],
                                            "endTime" : relevant[-1]["timestamp"],
                                            "people" : {}
                                        }

                                        currentTime = relevant[0]["timestamp"]
                                        for j in range(len(relevant)):
                                            currentPerson = relevant[j]["identifier"]
                                            if currentPerson not in clip["people"]:
                                                clip["people"][currentPerson] = {}
                                            
                                            if relevant[j]["label"] not in clip["people"][currentPerson]:
                                                clip["people"][currentPerson][relevant[j]["label"]] = []
                                            
                                            clip["people"][currentPerson][relevant[j]["timestamp"]] = {
                                                "label" : relevant[j]["label"],
                                                "bbox": relevant[j]["bbox"]
                                            }
                                        clip["people"] = clip["people"].values()
                                        db.loadedData["ava"].append(clip)

                                    relevant = []
                                else:
                                    if ava[i]["label"] in actionIDs:
                                        vids.add(ava[i]["id"])
                                        relevant.append(ava[i])
                            
                            print("Loaded "+str(len(db.loadedData["ava"]))+" clips from AVA-dataset")
                            return "NOTHING"
                        
                        def getYoutubeVideo(key):
                            ytURL = key.split("|")[0]
                            startTime = int(key.split("|")[1])
                            endTime = int(key.split("|")[2])

                            goalDimensions = np.array([400, 300])
                            video = None
                            try:
                                video = pafy.new(ytURL)
                            except IOError:
                                print("Video doesn't exists")
                                return False
                            bestStream = None
                            best = 100000
                            for stream in video.streams:
                                if stream.extension != "mp4":
                                    continue
                                dimensions = np.array(stream.dimensions)
                                distance = np.linalg.norm(np.abs(dimensions-goalDimensions))
                                if distance < best:
                                    best = distance
                                    bestStream = stream
                            
                            video = ffmpeg.input(bestStream.url, ss=startTime, t=endTime-startTime, format="mp4", loglevel="error")
                            video = video.output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel="error")
                            out, err = video.run(capture_stdout=True)

                            video = (
                                np
                                .frombuffer(out, np.uint8)
                                .reshape([-1, bestStream.dimensions[1], bestStream.dimensions[0], 3])
                            )

                            # FPS reduction
                            targetFPS = 10 # hz
                            variation = 0 # hz
                            fps = video.shape[0] / (endTime-startTime)
                            minStride = np.ceil(fps/(targetFPS + variation))
                            maxStride = np.ceil(fps/(targetFPS - variation))
                            if minStride == maxStride:
                                strides = np.full(video.shape[0], minStride, np.int32)
                            else:
                                strides = np.random.randint(low=minStride, high=maxStride, size=video.shape[0])
                            frames = np.cumsum(strides) - strides[0]
                            frames = frames[frames < video.shape[0]]
                            video = video[frames]
                            return video
                        

                        db.registerGenerator("modanet", getModanet)
                        db.registerGenerator("modanetImages", getModanetImage)
                        db.registerGenerator("densepose", genDensepose) # TODO: change this to denseposeModa
                        db.registerGenerator("people", genDensepose)
                        db.registerGenerator("UV_Textures", genUV_Textures)
                        db.registerGenerator("ava", getAva)
                        db.registerGenerator("youtube", getYoutubeVideo)

                    for sub in Algorithms.Algorithm.__subclasses__():
                        if sub.name in trainSettings["algorithms"]:
                            algorithms.append(sub(db=db))
                            break
                    
                    print("Starting training thread")
                    trainingTread = threading.Thread(target = trainingLoop, args=[self])
                    trainingTread.start()

            except Exception as e:
                print("Websocket hangleMessage exception:")
                traceback.print_exc()
            
        def handleConnected(self):
            print('websocket connected', self.address)
            clients.append(self)

        def handleClose(self):
            print('websocket closed', self.address)
            clients.remove(self)
            # TODO: send signal to end training
