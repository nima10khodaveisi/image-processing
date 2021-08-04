import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def point_operations(a):
    res = np.uint8(np.log(1 + (a * image)) * 255 / math.log(1 + (255 * a)))
    return res


image = cv2.imread('resources/Dark.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = point_operations(0.05)
result = cv2.GaussianBlur(result, (3, 3), cv2.BORDER_DEFAULT)
plt.imshow(result)
plt.show()
plt.imsave('res01.jpg', result)
