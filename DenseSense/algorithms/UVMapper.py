import DenseSense.algorithms.Algorithm

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import qhull


class UVMapper(DenseSense.algorithms.Algorithm.Algorithm):

    def __init__(self, db=None):
        super().__init__()

    def extract(self, people, image, threshold=100):
        resolution = 32

        peopleMaps = []
        for person in people:
            bbox = person.bounds

            # Check if person is large enough to work with
            area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            if area < threshold:
                peopleMaps.append(None)
                continue

            # Create person texture
            personMap = np.zeros((15, resolution, resolution, 3), dtype=np.uint8)

            # Go through each body part
            for partIndex in range(1, 15):  # 0 is background
                x, y = np.where(person.S == partIndex)
                if x.size < 4:  # Need at least 4 pixels to interpolate
                    continue

                u = person.U[x, y]
                v = person.V[x, y]

                # Add box global location
                x = np.floor(x*(bbox[3]-bbox[1])/56.0).astype(np.int32)
                x += bbox[1].astype(np.int32)
                y = np.floor(y*(bbox[2]-bbox[0])/56.0).astype(np.int32)
                y += bbox[0].astype(np.int32)

                pixels = image[x, y]

                gx, gy = np.mgrid[0:1:complex(0, resolution), 0:1:complex(0, resolution)]

                # Interpolate values. This can be a bit slow...
                try:
                    # TODO: use cuda accelerated interpolation instead
                    texture = griddata((u, v), pixels, (gx, gy),
                                       # Nearest looks weird, but is the consistently much fastest
                                       method="nearest", fill_value=0).astype(np.uint8)
                except qhull.QhullError as e:
                    continue

                personMap[partIndex] = texture

            peopleMaps.append(personMap)

        return np.array(peopleMaps)

    def train(self, saveModel):
        raise Exception("UV extraction algorithm cannot be trained")

    def getPeopleTexture(self, peopleMaps):
        peopleTextures = []
        for personMap in peopleMaps:
            # Put all textures in one square image
            personTexture = np.split(personMap, 5)
            personTexture = np.concatenate(personTexture, axis=2)
            personTexture = np.concatenate(personTexture, axis=0)
            peopleTextures.append(personTexture)

        return peopleTextures
