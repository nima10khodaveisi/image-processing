import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def pltShowImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.imshow(img)
    plt.show()
    plt.imsave('res02.jpg', img)
    return


image = cv2.imread('resources/Yellow.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(image, (18, 100, 100), (30, 255, 255))

image[mask > 0, 0] -= 18

pltShowImage(image.copy())
