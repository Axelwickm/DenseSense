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
        self.S_last = None
        self.I = None      # Activation for each body part
        self.U = None      # U mapping for each body part
        self.V = None      # V mapping for each body part

        self.A = None      # Alpha channel or mask to be applied to data
        self.pose_vector = None

    def become(self, otherInstance):
        self.incremental = otherInstance.incremental
        self.S_last = self.S
        self.S = otherInstance.S
        self.I = otherInstance.I
        self.U = otherInstance.U
        self.V = otherInstance.V
        self.A = otherInstance.A

        self.attrs.update(otherInstance.attrs)
        self.bounds = otherInstance.bounds
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
        oldBounds = self.bounds
        oldDims = oldBounds[2:]-oldBounds[:2]
        newDims = newBounds[2:]-newBounds[:2]
        posDelta = newBounds[:2]-oldBounds[:2]

        clipOld = (56 * np.array([
            posDelta[0] / oldDims[0], posDelta[1] / oldDims[1],
            (posDelta[0] + newDims[0]) / oldDims[0], (posDelta[1] + newDims[1]) / oldDims[1]
        ]).clip(0, 1)).astype(np.int32)

        clipNew = (56*np.array([
            -posDelta[0]/newDims[0], -posDelta[1]/newDims[1],
            (-posDelta[0]+oldDims[0])/newDims[0], (-posDelta[1]+oldDims[1])/newDims[1]
        ]).clip(0, 1)).astype(np.int32)

        newDims = tuple([clipNew[2] - clipNew[0], clipNew[3] - clipNew[1]])

        if which is None or "S" in which:
            oldCrop = self.S[clipOld[1]: clipOld[3], clipOld[0]: clipOld[2]]
            S_ = np.zeros((56, 56), dtype=np.int32)
            if oldCrop.shape[0] != 0 and oldCrop.shape[1] != 0 \
                    and newDims[0] != 0 and newDims[1] != 0:
                S_[clipNew[1]:clipNew[3], clipNew[0]:clipNew[2]] = \
                    cv2.resize(oldCrop.astype(np.float32),
                               newDims, interpolation=cv2.INTER_NEAREST).astype(np.int32)
            self.S = S_

        if which is None or "I" in which:
            oldCrop = self.I[clipOld[1]: clipOld[3], clipOld[0]: clipOld[2]]
            I_ = np.zeros((56, 56), dtype=np.int32)
            if oldCrop.shape[0] != 0 and oldCrop.shape[1] != 0 \
                    and newDims[0] != 0 and newDims[1] != 0:
                I_[clipNew[1]:clipNew[3], clipNew[0]:clipNew[2]] = \
                    cv2.resize(oldCrop.astype(np.float32),
                               newDims, interpolation=cv2.INTER_NEAREST).astype(np.int32)
            self.I = I_

        if which is None or "U" in which:
            oldCrop = self.U[clipOld[1]: clipOld[3], clipOld[0]: clipOld[2]]
            U_ = np.zeros((56, 56), dtype=np.float32)
            if oldCrop.shape[0] != 0 and oldCrop.shape[1] != 0 \
                    and newDims[0] != 0 and newDims[1] != 0:
                U_[clipNew[1]:clipNew[3], clipNew[0]:clipNew[2]] = \
                    cv2.resize(oldCrop,
                               newDims, interpolation=cv2.INTER_NEAREST)
            self.U = U_

        if which is None or "V" in which:
            oldCrop = self.U[clipOld[1]: clipOld[3], clipOld[0]: clipOld[2]]
            V_ = np.zeros((56, 56), dtype=np.float32)
            if oldCrop.shape[0] != 0 and oldCrop.shape[1] != 0 \
                    and newDims[0] != 0 and newDims[1] != 0:
                V_[clipNew[1]:clipNew[3], clipNew[0]:clipNew[2]] = \
                    cv2.resize(oldCrop,
                               newDims, interpolation=cv2.INTER_NEAREST)
            self.V = V_

        if which is None or "A" in which:
            oldCrop = self.A[clipOld[1]: clipOld[3], clipOld[0]: clipOld[2]]
            A_ = np.zeros((56, 56), dtype=np.int32)
            if oldCrop.shape[0] != 0 and oldCrop.shape[1] != 0 \
                    and newDims[0] != 0 and newDims[1] != 0:
                A_[clipNew[1]:clipNew[3], clipNew[0]:clipNew[2]] = \
                    cv2.resize(oldCrop.astype(np.float32),
                               newDims, interpolation=cv2.INTER_NEAREST).astype(np.int32)
            self.A = A_

        self.bounds = newBounds

    def merge(self, otherInstances, which=None):
        newBounds = self.bounds.copy()
        for o in otherInstances:
            newBounds[:2] = np.minimum(newBounds[:2], o.bounds[:2])
            newBounds[2:] = np.maximum(newBounds[2:], o.bounds[2:])

        self.applyAlpha(which=which)
        self.applyBounds(newBounds, which=which)

        for o in otherInstances:
            o.applyAlpha(which=which)
            o.applyBounds(newBounds, which=which)
            if which is None or "S" in which:
                self.S = np.where(self.A, self.S, o.S)
            if which is None or "I" in which:
                self.I = np.where(self.I, self.I, o.I)
            if which is None or "U" in which:
                self.U = np.where(self.A, self.U, o.U)
            if which is None or "V" in which:
                self.V = np.where(self.A, self.V, o.V)
            if which is None or "A" in which:
                self.A = o.A
