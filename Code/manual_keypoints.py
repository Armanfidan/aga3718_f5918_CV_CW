import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


img1 = cv2.imread(input("Please enter the first image path.\n"))
while img1 is None:
    img1 = cv2.imread(input("The image did not exist or could not be read. Please enter the first image path.\n"))
img2 = cv2.imread(input("Please enter the second image path.\n"))
while img2 is None:
    img2 = cv2.imread(input("The image did not exist or could not be read. Please enter the second image path.\n"))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)


def click_keypoints(img, shrink_factor=4) -> list:
    keypoints = []

    height = img.shape[0]
    width = img.shape[1]
    imgs = cv2.resize(img, (width // shrink_factor, height // shrink_factor))

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            x_real = x * shrink_factor
            y_real = y * shrink_factor
            keypoints.append(cv2.KeyPoint(x_real, y_real, 5))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(imgs, (x, y), 1, (0, 0, 255), 2)
            # cv2.putText(imgs, "Number of keypoints: " + str(len(keypoints)), (20, 30), font, .5, (0, 0, 255), 1)
            cv2.imshow(' Image', imgs)

    cv2.imshow(' Image', imgs)
    cv2.setMouseCallback(' Image', click_event)
    cv2.waitKey(0)

    return keypoints


scale = int(input("Please enter a shrink factor to shrink each image by.\n"))
input_kp1 = click_keypoints(img1, scale)
input_kp2 = click_keypoints(img2, scale)

print('\nImage 1: ', [point.pt for point in input_kp1])
print('Image 2:', [point.pt for point in input_kp2], '\n')
input("Press Enter to exit.")
