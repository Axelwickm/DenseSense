#!/usr/bin/env python3
"""
Capture webcamera input and run densepose, people tracking,
UV-extraction and label classification with debug views
turned on.
"""

import cv2
import numpy as np

import DenseSense
import DenseSense.algorithms.PeopleExtractor


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
    while True:
        # Get image from webcam
        return_value, image = cam.read()
        image = np.zeros((200, 300, 3), np.uint8)
        cv2.imshow("input image", image)

        # White balance the image to get better color features


        # Quit on escape
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

