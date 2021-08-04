import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.segmentation import mark_boundaries


def dis(img, x0, y0, x1, y1, alpha):
    ret = 0
    for channel in range(3):
        val = img[x0, y0, channel] - img[x1, y1, channel]
        ret += val * val
    eclud = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)
    alpha = 1
    return ret + eclud * alpha


image = cv2.imread('resources/slic.jpg')
image = cv2.medianBlur(image, 5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
image = image.astype('float32')


image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
k = 256
h = image.shape[0]
w = image.shape[1]



centers = []

sq = math.floor(math.sqrt(k))

for i in range(sq):
    for j in range(sq):
        center = (math.floor((h / sq) * i + (h / (2 * sq))), math.floor((w / sq) * j + (w / (2 * sq))))
        centers.append(center)

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
gradient = sobelx ** 2 + sobely ** 2

dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
dy = [-1, 1, 0, -1, 1, 0, -1, 1, 0]

for c in centers:
    mn = gradient[c[0], c[1], 0] + gradient[c[0], c[1], 1] + gradient[c[0], c[1], 2]
    ind = 5
    for dif in range(9):
        nx = dx[dif] + c[0]
        ny = dy[dif] + c[1]
        if gradient[nx, ny, 0] + gradient[nx, ny, 1] + gradient[nx, ny, 2] < mn:
            ind = dif
            mn = gradient[nx, ny, 0] + gradient[nx, ny, 1] + gradient[nx, ny, 2]

s = math.ceil(math.sqrt((image.shape[0] * image.shape[1]) / k)) + 5
m = 10


best_match = np.zeros((h, w, 2), dtype='int32')

count = 0

while True:
    old_centers = centers.copy()
    best_match.fill(-1)
    for c in centers:
        for i in range(max(c[0] - s, 0), min(c[0] + s, h - 1), 1):
            for j in range(max(c[1] - s, 0), min(c[1] + s, w - 1), 1):
                point = (i, j)
                if best_match[point[0], point[1], 0] == -1:
                    best_match[point[0], point[1], 0] = c[0]
                    best_match[point[0], point[1], 1] = c[1]
                else:
                    old_dis = dis(image, point[0], point[1], best_match[point[0], point[1], 0],
                                  best_match[point[0], point[1], 1], m / s)
                    new_dis = dis(image, point[0], point[1], c[0], c[1], m / s)
                    if new_dis < old_dis:
                        best_match[point[0], point[1], 0] = c[0]
                        best_match[point[0], point[1], 1] = c[1]

    # calculate new centers
    average = np.zeros((h, w, 3), dtype='float32')
    for i in range(h):
        for j in range(w):
            c0 = best_match[i, j, 0]
            c1 = best_match[i, j, 1]
            average[c0, c1, 0] += i
            average[c0, c1, 1] += j
            average[c0, c1, 2] += 1
    centers = []
    dif = 0
    for i in range(h):
        for j in range(w):
            c0 = best_match[i, j, 0]
            c1 = best_match[i, j, 1]
            if c0 == i and c1 == j:
                average[c0, c1, 0] /= average[c0, c1, 2]
                average[c0, c1, 1] /= average[c0, c1, 2]
                # this is a centroid
                val0 = (c0 - average[c0, c1, 0])
                val0 *= val0
                val1 = (c1 - average[c0, c1, 1])
                val1 *= val1
                dif += math.sqrt(val0 + val1)
                centers.append((math.floor(average[c0, c1, 0]), math.floor(average[c0, c1, 1])))
    count = count + 1
    if dif < 100 or count > 5:
        break

dx = [-1, 1, 0, 0]
dy = [0, 0, 1, -1]

image = image.astype('uint8')
image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


label = 1
labels = np.zeros((h, w), dtype='int8')
for i in range(h):
    for j in range(w):
        if best_match[i, j, 0] == i and best_match[i, j, 1] == j:
            labels[i, j] = label
            label = label + 1


for i in range(h):
    for j in range(w):
        if best_match[i, j, 0] != -1:
            labels[i, j] = labels[best_match[i, j, 0], best_match[i, j, 1]]

print(labels==0)

image = mark_boundaries(image, labels, color=(0, 0, 0))
image *= 255
image = image.astype('uint8')


image = cv2.resize(image, (0, 0), fx=10, fy=10)

plt.imshow(image)
plt.show()
plt.imsave('res06.jpg', image)
