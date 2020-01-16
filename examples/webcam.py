#!/usr/bin/env python3
"""
Capture web camera input and run DensePose, people tracking,
UV-extraction and label classification with debug views
turned on.
"""

import cv2
import numpy as np

from DenseSense.algorithms.DensePoseWrapper import DensePoseWrapper
from DenseSense.algorithms.Sanitizer import Sanitizer
from DenseSense.algorithms.Tracker import Tracker
#from DenseSense.algorithms.uv_extractor import UVMapper


def white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def main():
    cam = cv2.VideoCapture(0)

    densepose = DensePoseWrapper()
    sanitizer = Sanitizer()
    sanitizer.loadModel("./models/Sanitizer.pth")
    tracker = Tracker()
    #uvMapper =  UVMapper()

    while True:
        # Get image from webcam
        return_value, image = cam.read()
        assert return_value, "Failed to read from web camera"

        # White balance the image to get better color features
        image = white_balance(image)
        debugImage = image.copy()

        # Send image to DensePose
        people = densepose.extract(image)
        debugImage = densepose.renderDebug(debugImage, people)

        # Refine DensePose output to get actual people
        people = sanitizer.extract(people)
        debugImage = sanitizer.renderDebug(debugImage, alpha=0.2)

        # Track the people (which modifies the people variables)
        tracker.extract(people, True)
        debugImage = tracker.renderDebug(debugImage, people)

        # Extact UV map for each person
        #uvs = uvMapper.extract(people, mergedIUVs, image)

        # Show image
        print("Show image")
        cv2.imshow("debug image", debugImage)

        # Quit on escape
        if cv2.waitKey(1) == 27:
            break

        print("")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

