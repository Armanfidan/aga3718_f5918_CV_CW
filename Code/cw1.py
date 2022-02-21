import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


def show_image(image, image_size=(15, 15), colour=True):
    fig, axis = plt.subplots()
    axis.imshow(image, cmap="gray" if not colour else "viridis")
    fig.set_size_inches(image_size)
    axis.axis("off")
    plot.show()


def show_both_images(image_1, image_2, image_size=(30, 15), colour=True):
    figure, axes = plt.subplots(1, 2, figsize=image_size)
    axes[0].imshow(image_1, cmap="gray" if not colour else "viridis")
    axes[1].imshow(image_2, cmap="gray" if not colour else "viridis")
    axes[0].axis("off")
    axes[1].axis("off")
    plt.show()


img1 = cv2.imread("./FD/1.png")
img2 = cv2.imread("./FD/2.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)


# show_both_images(img1, img2)

orb = cv2.SIFT_create(50)

kp_img1, des_img1 = orb.detectAndCompute(img1, None)
kp_img2, des_img2 = orb.detectAndCompute(img2, None)

# print([str(kp.pt) + ", " + str(kp.size) for kp in kp_img1])
# print([str(kp.pt) + ", " + str(kp.size) for kp in kp_img2])

kp_img1_overlayed = cv2.drawKeypoints(img1, kp_img1, outImage=None, color=(255, 0, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2_overlayed = cv2.drawKeypoints(img2, kp_img2, outImage=None, color=(255, 0, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show_both_images(kp_img1_overlayed, kp_img2_overlayed)


def click_keypoints(img, shrink_factor=5) -> list:
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
            cv2.putText(imgs, str(x_real) + ',' + str(y_real), (x + 3, y - 3), font, .5, (0, 0, 255), 1)
            cv2.imshow('image', imgs)

    cv2.imshow('image', imgs)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)

    return keypoints


input_kp1 = click_keypoints(img1)
input_kp2 = click_keypoints(img2)

print([point.pt for point in input_kp1])
print([point.pt for point in input_kp2])

kp_img1, des_img1 = orb.compute(img1, input_kp1)
kp_img2, des_img2 = orb.compute(img2, input_kp2)

kp_img1_overlayed = cv2.drawKeypoints(img1, input_kp1, outImage=None, color=(255, 0, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2_overlayed = cv2.drawKeypoints(img2, input_kp2, outImage=None, color=(255, 0, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_both_images(kp_img1_overlayed, kp_img2_overlayed)
