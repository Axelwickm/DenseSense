
import numpy as np

import DenseSense.algorithms.Algorithm


class PeopleExtractor(DenseSense.algorithms.Algorithm.Algorithm):
    assigments = np.empty(0)

    def __init__(self, db=None):
        return
    
    def extract(self, boxes, bodies, image, training=False): # TODO: make training a member variable
        # Merge into one inds
        mergedIUVs = np.zeros((3, image.shape[0], image.shape[1]), dtype=np.float)
        if boxes.shape[0] == 0:
            return [], mergedIUVs

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(areas) 
        for i in sorted_inds:
            bbox = boxes[i, :4]
            IUVs = bodies[i]
            x1 = int(bbox[0])
            x2 = int(bbox[0] + IUVs.shape[2])
            y1 = int(bbox[1])
            y2 = int(bbox[1] + IUVs.shape[1])
            mergedIUVs[:, y1:y2, x1:x2] = IUVs
        
        mergedInds = mergedIUVs[0].astype(np.uint8)
        
        
        # Find the contours of all body parts
        COMs = []
        shift = 2
        minArea = 8*8
        bodyparts = []
        indexRegion = [None]
        
        t1 = time.time()

        def range_overlap(a_min, a_max, b_min, b_max):
                return (a_min <= b_max) and (b_min <= a_max)

        for ind in range(1, 25):
            # Extract contours from this body part
            regionWhere = np.where(mergedInds == ind)
            if len(regionWhere[0]) == 0:
                indexRegion.append(None)
                continue
            
            region = np.zeros_like(mergedInds, np.uint8)
            region[regionWhere] = 255
            region = cv2.Canny(region, 100, 101)
            kernel = np.ones((5,5),np.uint8)
            
            iterations = 5
            dialted = cv2.dilate(region, kernel, iterations=iterations)

            indexRegion.append(dialted)

            contour, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
            
            # Turn these to bounding boxes
            contourNP = np.asarray(contour)
            cb = np.zeros((len(contour), 4))
            for j in range(len(contour)):
                cb[j] = (
                    np.amin(contourNP[j][:, 0, 0]),
                    np.amin(contourNP[j][:, 0, 1]),
                    np.amax(contourNP[j][:, 0, 0]),
                    np.amax(contourNP[j][:, 0, 1])
                )

            cb = cb.tolist()
            
            # Merge overlaping bounds
            j = 0
            while j < len(contour):
                # Delete too small
                if (cb[j][2] - cb[j][0])*(cb[j][3] - cb[j][1]) < minArea:
                    del contour[j]
                    del cb[j]
                    if len(contour) == 0:
                        break
                    continue

                k = j+1
                while k < len(contour):
                    if k != j and range_overlap(cb[j][0], cb[j][2], cb[k][0], cb[k][2]) \
                                and range_overlap(cb[j][1], cb[j][3], cb[k][1], cb[k][3]):
                        # Boxes overlap, merge them!
                        contour[j] = np.append(contour[j], contour[k], axis=0) # FIXME: do this properly
                        del contour[k]
                        del cb[k]

                        c_np = np.asarray(contour[j])
                        cb[j] = (
                            np.amin(c_np[:, 0, 0]),
                            np.amin(c_np[:, 0, 1]),
                            np.amax(c_np[:, 0, 0]),
                            np.amax(c_np[:, 0, 1])
                        )
                        k = j+1
                    k += 1              
                j += 1
            contour = np.asarray(contour)
            for j in range(len(contour)):
                if len(contour[j]) != 0:
                    bodyparts.append((ind, cb[j]))


        # Function for merging parts (disjoint set)
        self.assigments = np.arange(-len(bodyparts), 0, dtype=np.int32)

        def find(i):
            if self.assigments[i] != i:
                self.assigments[i] = find(self.assigments[i])
            return self.assigments[i]


        # Find overlap between body parts
        overlaps = np.zeros((len(bodyparts), len(bodyparts)))
        for i in range(len(bodyparts)):
            for j in range(i+1, len(bodyparts)):
                ind_i, b_i = bodyparts[i]
                ind_j, b_j = bodyparts[j]
                if range_overlap(b_j[0], b_j[2], b_i[0], b_i[2]) \
                    and range_overlap(b_j[1], b_j[3], b_i[1], b_i[3]):
                        xMin = int(min(b_j[0], b_i[0]))
                        yMin = int(min(b_j[1], b_i[1]))
                        xMax = int(max(b_j[2], b_i[2]))
                        yMax = int(min(b_j[3], b_i[3]))

                        r_i = np.asarray(indexRegion[ind_i][yMin:yMax, xMin:xMax], np.bool)
                        r_j = np.asarray(indexRegion[ind_j][yMin:yMax, xMin:xMax], np.bool)
                        
                        overlap = np.sum(np.bitwise_and(r_i, r_j))
                        if overlap > 2:
                            if self.assigments[i] < 0:
                                self.assigments[i] = i
                            iRoot = find(self.assigments[i])
                            jRoot = find(self.assigments[j])
                            if iRoot != jRoot:
                                self.assigments[jRoot] = iRoot

        
        for i in range(len(self.assigments)):
            find(i) # Loop through one last time to make sure the tree is flat (so not a tree?)

        if training:
            self.bodyPartCounts.append(partCounts)

        t2 = time.time()

        # Save data
        people = []
        #self.debugOutput = {}
        j = 0
        bodyparts = np.asarray(bodyparts, object)

        for i in np.unique(self.assigments):
            parts = bodyparts[self.assigments == i]
            partBounds = parts[:, 1]
            savedParts = {}
            for p in parts:
                savedParts[int(p[0])] = np.asarray(p[1][0], np.float)

            xMin = 1000000
            yMin = 1000000
            xMax = 0
            yMax = 0
            for pb in partBounds:
                pb = np.array(pb)
                xMin = min(xMin, pb[0])
                yMin = min(yMin, pb[1])
                xMax = max(xMax, pb[2])
                yMax = max(yMax, pb[3])

            
            person = {
                "id" : j,
                "index" : j,
                "bounds": np.array([
                    xMin, yMin,
                    xMax, yMax
                ], dtype=np.float),
                "bodyparts": savedParts
            }

            people.append(person)
            #self.debugOutput[j] = centers[self.assigments == i]
            j += 1

        #print(people)
        return people, mergedIUVs

    def train(self, saveModel):
        raise Exception("PeopleExtractor algorithm cannot be trained")
