import cv2
from matplotlib import pyplot as plt
import numpy as np


def normalize_hist(image, name):
    for i, col in enumerate(('b', 'g', 'r')):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        cv2.normalize(histogram, histogram)
        plt.plot(histogram, color=col)
        plt.xlim([0, 256])
    if name == 'res05.jpg':
        plt.savefig('res05.jpg')
    plt.show()


darkImage = cv2.imread('resources/Dark.jpg')
darkImage = cv2.cvtColor(darkImage, cv2.COLOR_BGR2RGB)

pinkImage = cv2.imread('resources/Pink.jpg')
pinkImage = cv2.cvtColor(pinkImage, cv2.COLOR_BGR2RGB)

plt.hist(pinkImage.ravel(), 256, [0, 256])
plt.show()

normalize_hist(darkImage, None)


for channel in range(3):
    pinkChannel = np.zeros((pinkImage.shape[0], pinkImage.shape[1]), dtype='uint8')
    pinkChannel[:, :] = pinkImage[:, :, channel]

    darkChannel = np.zeros((darkImage.shape[0], darkImage.shape[1]), dtype='uint8')
    darkChannel[:, :] = darkImage[:, :, channel]

    h1 = []
    h2 = []

    pinkCount = 0
    darkCount = 0

    for intensity in range(256):
        pinkCount += np.count_nonzero(pinkChannel == intensity)
        h1.append(pinkCount)
        darkCount += np.count_nonzero(darkChannel == intensity)
        h2.append(darkCount)

    for intensity in range(256):
        h1[intensity] = (h1[intensity] / h1[255]) * 100
        h2[intensity] = (h2[intensity] / h2[255]) * 100

    to = np.zeros(256, dtype='uint8')

    target = 0
    for intensity in range(256):
        while target < 255 and h1[target] < h2[intensity]:
            target = target + 1
        to[intensity] = target

    darkImage[:, :, channel] = to[darkChannel[:, :]]

plt.imshow(darkImage)
plt.show()
plt.imsave('res06.jpg', darkImage)

normalize_hist(darkImage, 'res05.jpg')
normalize_hist(pinkImage, None)
