import numpy as np

from scipy.interpolate import griddata
from scipy.spatial import qhull


class UV_Extractor(Algorithm):
    
    def __init__(self, db = None):
        return
    
    def extract(self, people, mergedIUVs, image, threshold=100):
        resolution = 64
        I, U, V = mergedIUVs[0].astype(np.uint8), mergedIUVs[1], mergedIUVs[2]

        peopleTexture = []
        for person in people:
            bbox = person["bounds"]
            area = (bbox[3]-bbox[1]) * (bbox[2]-bbox[0])
            if area < threshold:
                peopleTexture.append(None)
                continue

            personTexture = np.zeros((25, resolution, resolution, 3), dtype=np.uint8)
            
            for partIndex in xrange(1, 25):
                box = np.asarray(person["bounds"], np.int32)

                x,y = np.where(I[box[1]:box[3], box[0]:box[2]] == partIndex)
                if x.size < 4: # Need at least 4 pixels geo interpolate
                    continue # Did not find this bodypa ge
                
                u = U[box[1]:box[3], box[0]:box[2]][x,y]
                v = V[box[1]:box[3], box[0]:box[2]][x,y]

                # Add box global location
                x += np.floor(box[1]).astype(np.int64) # CHANGE
                y += np.floor(box[0]).astype(np.int64) # CHANGE
                
                pixels = image[x,y]
                
                gx, gy = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]

                # Interpolate values. This can be a bit slow...
                try:
                    # TODO: use cuda accelerated interpolation instead
                    texture = griddata((u, v), pixels, (gx, gy), # Nearest looks weird, but is the consistently much fastest
                        method="nearest", fill_value=0).astype(np.uint8)
                except qhull.QhullError as e:
                    continue
 
                personTexture[partIndex] = texture

            # Put all textures in one square image
            personTexture = np.split(personTexture, 5)
            personTexture = np.concatenate(personTexture, axis=2)
            personTexture = np.concatenate(personTexture, axis=0)

            peopleTexture.append(personTexture)
        
        return np.array(peopleTexture)
    
    def train(self, saveModel):
        raise Exception("UV extraction algorithm cannot be trained")
