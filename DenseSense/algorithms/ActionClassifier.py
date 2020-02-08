import torch
import torch.nn as nn
import torch.nn.functional as F

import DenseSense.utils.YoutubeLoader

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
            super(Action_Classifier.Network, self).__init__()

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
    upcoming = []
    videoBuffer = {}
    lastVideo = None
    recent = deque()

    db = None
    epoch = 0
    iteration = 0

    def __init__(self, db=None):
        print("Initiating ActionClassifier")
        super().__init__()

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

        return self.iteration, 0.1, (t2-t1) # Loss, time

