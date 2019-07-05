
import torch
import torch.nn as nn
import torch.nn.functional as F

from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
print("PyTorch running on "+str(device))

class DescriptionExtractor(Algorithm):
    iteration = 0

    availableLabels = {
        3 : "boots",
        4 : "footwear",
        5 : "outer",
        6 : "dress",
        7 : "sunglasses",
        8 : "pants",
        9 : "top",
        10 : "shorts",
        11 : "skirt",
        12 : "headwear",
        13 : "scarfAndTie"
    }

    labelBodyparts = { # https://github.com/facebookresearch/DensePose/issues/64#issuecomment-405608749 PRAISE 
        "boots" : [5, 6],
        "footwear": [5, 6],
        "outer" : [1, 2, 15, 17, 16, 18, 19, 21, 20, 22],
        "dress" : [1, 2],
        "sunglasses" : [],
        "pants" : [7, 9, 8, 10, 11, 13, 12, 14],
        "top" : [1, 2],
        "shorts" : [7, 9, 8, 10],
        "skirt" : [1, 2],
        "headwear" : [23, 24],
        "scarfAndTie" : []
    }

    colors = [ # TODO: use color model file
        ((255, 255, 255),  "white"),
        ((210, 209, 218),  "white"),
        ((145, 164, 164),  "white"),
        ((169, 144, 135),  "white"),
        ((197, 175, 177),  "white"),
        ((117, 126, 115),  "white"),
        ((124, 126, 129),  "white"),
        ((0, 0, 0),        "black"),
        ((10, 10, 10),     "black"),
        ((1, 6, 9),        "black"),
        ((5, 10, 6),       "black"),
        ((18, 15, 11),     "black"),
        ((18, 22, 9),      "black"),
        ((16, 16, 14),     "black"),
        ((153, 153, 0),    "yellow"),
        ((144, 115, 99),   "pink"),
        ((207, 185, 174),  "pink"),
        ((206, 191, 131),  "pink"),
        ((208, 179, 54),   "pink"),
        ((202, 19, 43),    "red"),
        ((206, 28, 50),    "red"),
        ((82, 30, 26),     "red"),
        ((156, 47, 35),    "orange"),
        ((126, 78, 47),    "wine red"),
        ((74, 72, 77),     "green"),
        ((31, 38, 38),     "green"),
        ((40, 52, 79),     "green"),
        ((100, 82, 116),   "green"),
        ((8, 17, 55),      "green"),
        ((29, 31, 37),     "dark green"),
        ((46, 46, 36),     "blue"),
        ((29, 78, 60),     "blue"),
        ((74, 97, 85),     "blue"),
        ((60, 68, 67),     "blue"),
        ((181, 195, 232),  "neon blue"),
        ((40, 148, 184),   "bright blue"),
        ((210, 40, 69),    "orange"),
        ((66, 61, 52),     "gray"),
        ((154, 120, 147),  "gray"),
        ((124, 100, 86),   "gray"),
        ((46, 55, 46),     "gray"),
        ((119, 117, 122),  "gray"),
        ((88, 62, 62),     "brown"),
        ((60, 29, 17),     "brown"),
        ((153, 50, 204),   "purple"),
        ((77, 69, 30),     "purple"),
        ((153, 91, 14),    "violet"),
        ((207, 185, 151),  "beige")        
    ]

    colorsHSV = None

    net = None
    criterion = None
    optimizer = None
    modelFile = None

    noActivation = 0.2
    onActivation = 0.8

    debugOutput = 0
    lossAvg = 0

    db = None
    modanetGenerator = None
    epoch = 0
    processedImagesThisEpoch = 0

    class Network(nn.Module):
        def __init__(self, labels): # FIXME: make this work!
            super(DescriptionExtractor.Network, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
            )
            """
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
            """
            self.fc1 = nn.Linear(75843, 750) 
            self.relu1 = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(750, labels)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            out = self.layer1(x)
            #out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return out

    def __init__(self, model=None, db = None):
        print("Initiating DescriptionExtractor")

        self.modelFile = model
        # Init classifier
        self.net = DescriptionExtractor.Network(len(self.availableLabels))
        if self.modelFile is not None:
            print("Loading people description file from: "+self.modelFile)
            self.net.load_state_dict(torch.load(model))
        self.net.to(device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0003)

        # Init color lookup KD-tree
        self.colorsHSV = []
        for c in self.colors:
            RGBobj = sRGBColor(c[0][0], c[0][1], c[0][2])
            self.colorsHSV.append(convert_color(RGBobj, HSVColor))

        # Set database and generator for pulling 
        if db is not None:
            self.db = db
            self.modanetGenerator = self.createModanetGenerator()

    def extract(self, peopleTexture, training = False):
        labelsPeople = []
        labelsPeopleVector = [] # For training
        first = True

        # Do label classification
        for personTexture in peopleTexture: # TODO: batch run all people?
            if personTexture is None:
                labelsPeople.append(None)
                labelsPeopleVector.append(None)
                continue
            
            # Run the classification on it
            pyTorchTexture = torch.from_numpy(np.array([np.moveaxis(personTexture.astype(float)/255.0, -1, 0)])).float()
            if training: # Make this a variable if it isn't training
                pyTorchTexture = torch.autograd.Variable(pyTorchTexture)
        
            pyTorchTexture = pyTorchTexture.to(device)
            labelVector = self.net(pyTorchTexture)[0]

            def findColorName(areas):
                areaS = int(personTexture.shape[0]/5)
                Rs, Gs, Bs= [], [], []

                # Pick out colors
                for i in areas:
                    xMin = int((i%5)*areaS)
                    yMin = int(np.floor(i/5)*areaS)
                    for j in range(20):
                        x = np.random.randint(xMin, xMin + areaS)
                        y = np.random.randint(yMin, yMin + areaS)
                        b = personTexture[x, y, 0]
                        g = personTexture[x, y, 1]
                        r = personTexture[x, y, 2]
                        
                        if r != 0 or b != 0 or g != 0:
                            Rs.append(r)
                            Gs.append(g)
                            Bs.append(b)
                
                if len(Rs) + len(Gs) + len(Bs) < 3:
                    return 0
                
                # Find mean color
                r = np.mean(np.array(Rs)).astype(np.uint8)
                g = np.mean(np.array(Gs)).astype(np.uint8)
                b = np.mean(np.array(Bs)).astype(np.uint8)

                # This prints the color colored in the terminal
                RESET = '\033[0m'
                def get_color_escape(r, g, b, background=False):
                    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)
                colorRepr = get_color_escape(r, b, g)+"rgb("+str(r)+", "+str(g)+", "+str(b)+")"+RESET

                # Get nearest color name
                HSVobj = convert_color(sRGBColor(r, g, b), HSVColor)

                nearestIndex = -1
                diffMin = 100000
                for i in xrange(len(self.colorsHSV)):
                    colEntry = self.colorsHSV[i]

                    d = HSVobj.hsv_h - colEntry.hsv_h
                    dh = min(abs(d), 360-abs(d)) / 180.0
                    ds = abs(HSVobj.hsv_s - colEntry.hsv_s)
                    dv = abs(HSVobj.hsv_v - colEntry.hsv_v) / 255.0
                    diff = np.sqrt(dh*dh + ds*ds + dv*dv)
                    if diff < diffMin:
                        diffMin = diff
                        nearestIndex = i

                return { "color":self.colors[nearestIndex][1], "colorDistance":diffMin, "coloredStr":colorRepr }
            
            # Store the data
            if not training:
                labelVectorHost = labelVector.detach().cpu().numpy()
                labels = {}
                for j in range(len(labelVector)):
                    label = self.availableLabels.values()[j]
                    d = (self.onActivation - self.noActivation)/2
                    val = (labelVectorHost[j] - d) / d + 0.5

                    info = { "activation" : min(max(val, 0.0), 1.0) }
                    if 0.7 < val:
                        color = findColorName(self.labelBodyparts[label])
                        if color != 0:
                            info.update(color)
                            #print(color["color"]+"  "+color["coloredStr"])
                    labels[label] = info
                
                labelsPeople.append(labels)
            labelsPeopleVector.append(labelVector)

        if training:
            return labelsPeopleVector
        torch.cuda.empty_cache()        
        return labelsPeople

    def createModanetGenerator(self):
        self.db.getData("modanet", "")
        modanet = self.db.getAllLoaded()["modanet"]

        for key, annotations in modanet.iteritems():
            yield key, annotations
        
        # Reached end of dataset. Restart
        self.epoch += 1
        self.processedImagesThisEpoch = 0
        self.modanetGenerator = self.createModanetGenerator()
        yield self.modanetGenerator.next() # Passes on the torch as a last act, then dies. How tragic and beautiful :,)

    def train(self, saveModel):
        self.iteration += 1

        # Load annotations
        imageID, annotations = self.modanetGenerator.next()

        # Convert annotation labels to vector
        labels = np.full(len(self.availableLabels), self.noActivation)
        
        
        for a in annotations:
            if a["category_id"] in self.availableLabels:
                labelIndex = self.availableLabels.keys().index(a["category_id"])
                labels[labelIndex] = self.onActivation
                
        labels = torch.autograd.Variable(torch.from_numpy(labels)).float().to(device)

        # Get UV texture
        UV_Textures = self.db.getData("UV_Textures", imageID)  

        if len(UV_Textures) == 0:
            self.processedImagesThisEpoch += 1
            return self.train(saveModel) # No person in this image (rare)

        self.debugOutput = UV_Textures[0]

        self.optimizer.zero_grad()

        startTime = time.time()
        output = self.extract(UV_Textures, True) # Got texture from drive
        endTime = time.time()
        output = output[0] # Only 1 person per picture in modanet-dataset
        
        loss_size = self.criterion(output, labels)
        loss_size.backward()
        self.optimizer.step()
        loss_size_detached = loss_size.item()

        self.lossAvg += loss_size_detached
        if self.iteration%6000 == 0:
            writeTrainingData(self.lossAvg/6000.0, loss_size_detached)

        self.processedImagesThisEpoch += 1
        torch.cuda.empty_cache()
        print("TARGET", labels)
        print("GOT   ", output)
        #print("ALLOCATED CUDA ", torch.cuda.memory_allocated(device))
        if saveModel:
            torch.save(self.net.state_dict(), "/shared/trainedDescription.model") # TODO: have this be the self.modelfile instead
        return (self.iteration-1, loss_size_detached, endTime - startTime)