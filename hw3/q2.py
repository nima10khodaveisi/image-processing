import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
import numpy as np


def do_meanshift(img):
    H, W = img.shape[:2]
    y, x = np.mgrid[:H, :W]
    y = (255 / H) * y
    x = (255 / W) * x
    arr = np.concatenate((y[..., np.newaxis], x[..., np.newaxis], img), 2).astype('float')
    clusters = MeanShift(bandwidth=30, bin_seeding=True).fit(arr.reshape(-1, arr.shape[2]))
    res = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    mx = np.max(clusters.labels_)
    plt.imshow(img)
    plt.show()
    for k in range(mx):
        sum_r = 0
        sum_g = 0
        sum_b = 0
        cnt = 0
        for i in range(H):
            for j in range(W):
                if clusters.labels_[i * W + j] == k:
                    sum_r += img[i, j, 0]
                    sum_g += img[i, j, 1]
                    sum_b += img[i, j, 2]
                    cnt = cnt + 1
        if cnt == 0:
            break
        sum_r /= cnt
        sum_b /= cnt
        sum_g /= cnt
        for i in range(H):
            for j in range(W):
                if clusters.labels_[i * W + j] == k:
                    res[i, j, 0] = sum_r.astype('uint8')
                    res[i, j, 1] = sum_g.astype('uint8')
                    res[i, j, 2] = sum_b.astype('uint8')
    return res


image = cv2.imread('resources/park.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
image = cv2.medianBlur(image, 5)
result = do_meanshift(image)
result = cv2.resize(result, (0, 0), fx=4, fy=4)
result = cv2.medianBlur(result, 5)
plt.imshow(result)
plt.show()
plt.imsave('res04.jpg', result)
