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
from DenseSense.utils.DebugRenderer import DebugRenderer
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
    dr = DebugRenderer()

    while True:
        # Get image from webcam
        return_value, image = cam.read()

        # White balance the image to get better color features
        image = white_balance(image)

        # Send image to Densepose
        people = dp.extract(image)
        print("len", len(people))

        # Track the people (which modifies the people variables)
        #pt.extract(people, True)

        print("len after", len(people))

        # Extact UV map for each person
        #uvs = uv.extract(people, mergedIUVs, image)


        for person in people:
            break
            """
            for i in range(0, 25):
                gray = person.I[i].numpy()
                gray[gray < 0.0] = 0
                gray = cv2.normalize(gray, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.imshow(str(i), gray)"""
            if cv2.waitKey(0) == 27:
                continue
        #return



        # Show image
        dr.setQueue([dp])
        debugImage = dr.render(image, people)

        cv2.imshow("input image", image)
        cv2.imshow("debug image", debugImage)

        # Quit on escape
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

