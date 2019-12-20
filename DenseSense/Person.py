import numpy as np


class Person(object):
    incremental = 0

    def __init__(self):
        self.id = Person.incremental
        Person.incremental += 1

        self.attrs = {}

        self.bounds = np.array([])  # x1, y1, x2, y2 bounds
        self.S = None      # Coarse segmentation
        self.I = None      # Activation for each body part
        self.U = None      # U mapping for each body part
        self.V = None      # V mapping for each body part

        self.S_ind = None  # The most activated segmentation ( or background) for each pixel
        self.I_ind = None  # The most activated body part ( or background) for each pixel

    def become(self, otherInstance):
        self.incremental = otherInstance.incremental
        self.attrs.update(otherInstance.attrs)
        return self
