import numpy as np
from colormath.color_objects import sRGBColor, HSVColor
from colormath.color_conversions import convert_color
import cv2

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

        self.A = None      # Alpha channel or mask to be applied to data

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

    def applyAlpha(self, which=None):
        if which is None or "S" in which:
            self.S = self.S * self.A

        if which is None or "I" in which:
            self.I = self.I * self.A

        if which is None or "U" in which:
            self.U = self.U * self.A

        if which is None or "V" in which:
            self.V = self.V * self.A

    def applyBounds(self, newBounds, which=None):
        oldDims = self.bounds[2:]-self.bounds[:2]
        newDims = newBounds[2:]-newBounds[:2]
        relativeDims = oldDims/newDims
        d = tuple(np.floor(np.array([56, 56])*relativeDims).astype(np.int32))
        p = np.floor((self.bounds[:2]-newBounds[:2])/newDims*56).astype(np.int32)

        if which is None or "S" in which:
            S = np.zeros((56, 56))
            S[p[1]:p[1]+d[1], p[0]:p[0]+d[0]] = cv2.resize(self.S, d, interpolation=cv2.INTER_AREA)
            self.S = S

        if which is None or "I" in which:
            I = np.zeros((56, 56))
            I[p[1]:p[1]+d[1], p[0]:p[0]+d[0]] = cv2.resize(self.I, d, interpolation=cv2.INTER_AREA)
            self.I = I

        if which is None or "U" in which:
            U = np.zeros((56, 56))
            U[p[1]:p[1] + d[1], p[0]:p[0] + d[0]] = cv2.resize(self.U, d, interpolation=cv2.INTER_AREA)
            self.U = U

        if which is None or "V" in which:
            V = np.zeros((56, 56))
            V[p[1]:p[1] + d[1], p[0]:p[0] + d[0]] = cv2.resize(self.V, d, interpolation=cv2.INTER_AREA)
            self.V = V

        if which is None or "A" in which:
            A = np.zeros((56, 56))
            A[p[1]:p[1] + d[1], p[0]:p[0] + d[0]] = cv2.resize(self.A, d, interpolation=cv2.INTER_AREA)
            self.A = A

        self.bounds = newBounds

    def merge(self, otherInstances, which=None):
        newBounds = self.bounds
        for o in otherInstances:
            newBounds[:2] = np.minimum(newBounds[:2], o.bounds[:2])
            newBounds[2:] = np.maximum(newBounds[2:], o.bounds[2:])

        self.applyAlpha(which=which)
        self.applyBounds(newBounds, which=which)

        for o in otherInstances:
            o.applyAlpha(which=which)
            o.applyBounds(newBounds, which=which)
            if which is None or "S" in which:
                self.S = np.where(o.A, o.S, self.S)
            if which is None or "I" in which:
                self.I = np.where(o.A, o.I, self.I)
            if which is None or "U" in which:
                self.U = np.where(o.A, o.U, self.U)
            if which is None or "V" in which:
                self.V = np.where(o.A, o.V, self.V)
            if which is None or "A" in which:
                self.A += o.A
