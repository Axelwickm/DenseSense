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
from DenseSense.algorithms.UVMapper import UVMapper
from DenseSense.algorithms.DescriptionExtractor import DescriptionExtractor


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
    frameIndex = 0
    oldOpenWindows = set()

    densepose = DensePoseWrapper()
    sanitizer = Sanitizer()
    sanitizer.loadModel("./models/Sanitizer.pth")
    tracker = Tracker()
    uvMapper = UVMapper()
    descriptionExtractor = DescriptionExtractor()

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

        print("DensePose people:", len(people))

        # Refine DensePose output to get actual people
        people = sanitizer.extract(people)
        debugImage = sanitizer.renderDebug(debugImage, people, alpha=0.2)
        print("Sanitizer people", len(people))

        # Track the people (which modifies the people variables)
        tracker.extract(people, True)
        debugImage = tracker.renderDebug(debugImage, people)
        print("Tracker people", len(people))

        # Extract UV map for each person
        peopleMaps = uvMapper.extract(people, image)
        peopleTextures = uvMapper.getPeopleTexture(peopleMaps)

        # Classify what the person is wearing
        clothes = descriptionExtractor.extract(peopleMaps)
        clothingImage = descriptionExtractor.getLabelImage()


        # Per person window management
        newOpenWindows = set()
        for i, person in enumerate(people):
            windowName = "UV image "+str(person.id)
            newOpenWindows.add(windowName)
            #cv2.imshow(windowName, peopleTextures[i])
            cv2.imshow(windowName, clothingImage[i])

        for oldWindow in oldOpenWindows:
            if oldWindow not in newOpenWindows:
                cv2.destroyWindow(oldWindow)
        oldOpenWindows = newOpenWindows

        # Show image
        print("Show frame:", frameIndex, "\n")
        cv2.imshow("debug image", debugImage)
        frameIndex += 1

        # Quit on escape
        if cv2.waitKey(1) == 27:
            break


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

