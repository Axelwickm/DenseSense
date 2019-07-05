import cv2
import numpy as np


class People_Tracker(Algorithm):
    trackedObjects = []
    seenPeople = 0
    frame = 0
    lastFrameTime = 0
    def __init__(self):
        Algorithm.__init__(self, "tracker")
        self.lastFrameTime = time.time()
    
    def extract(self, people):
        self.frame += 1
        nowTime = time.time()
        dt = self.lastFrameTime - nowTime
        self.lastFrameTime = nowTime

        distThreshold = 500

        delete = 0.3
        hide = 0.5
        minFrames = 3
        maxPersistanceBufferSize = 15


        # Find optimal tracker <-> person configuration
        status = [-1 for _ in xrange(len(people))]
        associations = [(-1, 10000) for _ in xrange(len(self.trackedObjects))]

        def match(person):
            # Convert to center and dim coords
            bounds = person["bounds"]
            dims = np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]])
            center = np.array([bounds[0]+dims[0]/2, bounds[1]+dims[1]/2])
            
            # Find the closest viable tracked object
            displacedPerson = None
            closest = 1000000
            cInd = -1
            for i in xrange(len(self.trackedObjects)):
                predictedCenter = np.squeeze(self.trackedObjects[i][11][:2])
                dist = np.linalg.norm(predictedCenter-center)
                sizeDiff = np.linalg.norm(dims-np.squeeze(self.trackedObjects[i][11][4:]))
                #dist += sizeDiff/5
                if dist < closest:
                    # Is this the closest not reserved?
                    if associations[i][0] == -1 or dist < associations[i][1]:
                        if associations[i][0] != -1:
                            displacedPerson = associations[i][0]
                        else:
                            displacedPerson = None
                        closest = dist
                        cInd = i

            # If this was close enough to be this object
            if closest < distThreshold:
                return cInd, closest, displacedPerson
            return None, None, None

        
        i = 0
        while i < len(people):
            if status[i] != -1:
                i += 1
                continue
            person = people[i]
            ind, dist, displaced = match( person)
            if ind is not None:
                associations[ind] = (i, dist)
                status[i] = ind
                if displaced is not None:
                    # A person (guaranteed lower index) has been displaced. Reconsider this person
                    status[displaced] = -1
                    i = displaced
                    continue
            i += 1

        
        for i in range(len(people)):
            person = people[i]
            bounds = person["bounds"]
            dims = np.array([bounds[2]-bounds[0], bounds[3]-bounds[1]])
            center = np.array([bounds[0]+dims[0]/2, bounds[1]+dims[1]/2])

            if status[i] != -1:
                cInd = status[i]
                person["id"] = self.trackedObjects[cInd][0]
                oldPosition = self.trackedObjects[cInd][1]
                self.trackedObjects[cInd][1] = center 
                self.trackedObjects[cInd][4] = person["index"]
                self.trackedObjects[cInd][5] = self.frame
                self.trackedObjects[cInd][6] = person
                self.trackedObjects[cInd][7].append(1)
                self.trackedObjects[cInd][9] = nowTime

                self.trackedObjects[cInd][10].transitionMatrix[0, 2] = dt*0.4
                self.trackedObjects[cInd][10].transitionMatrix[1, 3] = dt*0.4
                self.trackedObjects[cInd][10].correct(np.concatenate([center, dims]))
                self.trackedObjects[cInd][11] = self.trackedObjects[cInd][10].predict()
                
            else:
                # This must be a new object
                person["id"] = self.seenPeople
                t = cv2.CV_64F 
                s = 6 # x, y, vx, vy, wx, w, h
                m = 4 # x, y, w, h 
                kalman = cv2.KalmanFilter(s, m, 0, type=t)

                kalman.transitionMatrix= np.eye(s)
                kalman.measurementMatrix = np.zeros((m, s))
                kalman.measurementMatrix[0, 0] = 1
                kalman.measurementMatrix[1, 1] = 1
                kalman.measurementMatrix[2, 4] = 1
                kalman.measurementMatrix[3, 5] = 1

                kalman.processNoiseCov = 1e-2 * np.eye(s)
                kalman.processNoiseCov[2, 2] = 5
                kalman.processNoiseCov[3, 3] = 5

                kalman.measurementNoiseCov = 1e-1 * np.eye(m)

                kalman.statePre = np.array([center[0], center[1], 0, 0, dims[0], dims[1]], np.float)
                

                self.trackedObjects.append([ # TODO: use something different than a list for this
                    self.seenPeople, # 0 ID
                    center,          # 1 Center
                    0,               # 2 Times seen
                    self.frame,      # 3 First seen
                    person["index"], # 4 Raw index
                    self.frame,      # 5 Last seen
                    person,          # 6 Person object
                    deque([1]),      # 7 Persistance buffer
                    False,           # 8 Is visible
                    nowTime,         # 9 Last seen time
                    kalman,          # 10 Velocity kalman-filter
                    kalman.statePre  # 11 Prediction new frame
                ])

                self.seenPeople += 1

        toRemove = []
        i = 0
        while i < len(self.trackedObjects):
            deltaLastSeen = nowTime - self.trackedObjects[i][9]
            if i < len(associations):
                if associations[i][0] == -1:
                    self.trackedObjects[i][7].append(0)

            if maxPersistanceBufferSize < len(self.trackedObjects[i][7]):
                self.trackedObjects[i][7].popleft()
            persistance = float(sum(self.trackedObjects[i][7]))/len(self.trackedObjects[i][7])

            # If should be removed and is seen
            if hide < persistance and minFrames < len(self.trackedObjects[i][7]):
                self.trackedObjects[i][8] = True
            else:
                self.trackedObjects[i][8] = False
                if self.trackedObjects[i][5] == self.frame:
                    toRemove.append(self.trackedObjects[i][4]) 
            
            if persistance < delete:
                del self.trackedObjects[i]
                continue
            
            # If shouldn't be removed and not seen
            if self.trackedObjects[i][5] != self.frame and self.trackedObjects[i][8]:
                self.trackedObjects[i][6]["index"] = len(people)
                people.append(self.trackedObjects[i][6])
            
            i += 1
        
        for index in sorted(toRemove, reverse=True):
            del people[index]
        
        ghostBounds = []
        keptAliveIndex = []
        predictions = []
        for to in self.trackedObjects:
            predictions.append((to[0], np.squeeze(to[11][:2]), to[8]))
            if not to[8]:
                ghostBounds.append(to[6]["bounds"])
            elif to[8] and to[5] != self.frame:
                keptAliveIndex.append(to[6]["index"])
        
        print("Tracking: "+str(len(self.trackedObjects))+" ("+str(len(ghostBounds))+" ghosts)")
        
        self.debugOutput = (
            np.array(ghostBounds),
            keptAliveIndex,
            predictions
        )