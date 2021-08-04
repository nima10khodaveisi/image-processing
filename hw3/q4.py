import cv2
import numpy as np
from sklearn.cluster import MeanShift
from skimage.segmentation import felzenszwalb
from matplotlib import pyplot as plt


image = cv2.imread('resources/birds.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

segmentation = felzenszwalb(image, sigma=1, scale=320, min_size=190)

numberOfSegments = np.max(segmentation) + 1

vectors = np.zeros((numberOfSegments, 6), dtype='float32')

for k in range(numberOfSegments):
    r_av = np.average(image[segmentation==k][:, 0])
    g_av = np.average(image[segmentation==k][:, 1])
    b_av = np.average(image[segmentation==k][:, 2])
    r_std = np.std(image[segmentation==k][:, 0])
    b_std = np.std(image[segmentation==k][:, 1])
    g_std = np.std(image[segmentation==k][:, 2])
    vectors[k, 0] = r_av
    vectors[k, 1] = g_av
    vectors[k, 2] = b_av
    vectors[k, 3] = r_std
    vectors[k, 4] = g_std
    vectors[k, 5] = b_std


clusters = MeanShift(bandwidth=18).fit(vectors)
bird_point = (916, 1513)

result = np.zeros(image.shape, dtype='uint8')

for k in range(numberOfSegments):
    if clusters.labels_[k] == clusters.labels_[segmentation[bird_point[0], bird_point[1]]]:
        print("there you go")
        print(image[segmentation==k])
        result[segmentation==k] = image[segmentation==k]

result = cv2.resize(result, (0, 0), fx=2, fy=2)

plt.imshow(result)
plt.show()
plt.imsave('res08.jpg', result)
