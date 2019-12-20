import numpy as np


class DebugRenderer:
    def __init__(self):
        self.toRender = []

    def setQueue(self, queue):
        self.toRender = queue

    def render(self, image, people, showInput=True):
        if not showInput:
            image = np.zeros_like(image, np.uint8)

        for algorithm in self.toRender:
            image = algorithm.renderDebug(image, people)
        return image
