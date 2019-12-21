import numpy as np
from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color


class Person(object):
    incremental = 0

    def __init__(self):
        self.id = Person.incremental
        Person.incremental += 1

        color = HSVColor(self.id * 66 % 360, 0.6, 0.78)
        color = convert_color(color, sRGBColor).get_value_tuple()
        self.color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # Convert to BGR tuple

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
        self.S = otherInstance.S
        self.I = otherInstance.I
        self.U = otherInstance.U
        self.V = otherInstance.V

        self.S_ind = otherInstance.S_ind
        self.I_ind = otherInstance.I_ind

        self.attrs.update(otherInstance.attrs)
        return self
