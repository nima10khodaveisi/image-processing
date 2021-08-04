import cv2
from matplotlib import pyplot as plt
import math
import numpy as np


def pltShowImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    plt.imshow(img)
    plt.show()
    plt.imsave('res03.jpg', img)
    return


image = cv2.imread('resources/Pink.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(image, (150, 60, 0), (180, 255, 255))

image[mask > 0, 0] -= 60


pltShowImage(image.copy())
