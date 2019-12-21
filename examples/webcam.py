#!/usr/bin/env python3
"""
Capture webcamera input and run densepose, people tracking,
UV-extraction and label classification with debug views
turned on.
"""

import cv2
import numpy as np

from DenseSense.algorithms.densepose import DenseposeExtractor
from DenseSense.algorithms.people_extractor import PeopleExtractor
from DenseSense.algorithms.people_tracker import People_Tracker
#from DenseSense.algorithms.uv_extractor import UV_Extractor


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

    dp = DenseposeExtractor()
    #pe = PeopleExtractor()
    pt = People_Tracker()
    #uv =  UV_Extractor()

    while True:
        # Get image from webcam
        return_value, image = cam.read()

        # White balance the image to get better color features
        image = white_balance(image)
        debugImage = image.copy()

        # Send image to DensePose
        people = dp.extract(image)
        debugImage = dp.renderDebug(debugImage, people)

        # Track the people (which modifies the people variables)
        pt.extract(people, True)
        #debugImage = pt.renderDebug(debugImage, people)

        # Extact UV map for each person
        #uvs = uv.extract(people, mergedIUVs, image)

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

