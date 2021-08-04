import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy
from scipy.sparse import lil_matrix


def fix(img):
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def show_image(img):
    img = fix(img)
    img = img.astype('uint8')
    plt.imshow(img)
    plt.show()
    plt.imsave('res01.jpg', img)


def get_laplacian(img, x, y):
    return 4 * img[x, y] - img[x - 1, y] - img[x + 1, y] - img[x, y - 1] - img[x, y + 1]


target = cv2.imread('resources/sky1.jpg')
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
# lake = lake.astype('float32')

source = cv2.imread('resources/bernie4.jpg')
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
# bear = bear.astype('float32')




y0 = 300
y1 = 550
x0 = 0
x1 = 430
ty0 = 10
tx0 = 100
ty1 = y1 - y0 + ty0
tx1 = x1 - x0 + tx0
n = y1 - y0 + 1
m = x1 - x0 + 1


source_mask = np.zeros((source.shape[0], source.shape[1]), dtype='uint8')
source_mask[x0: x1 + 1, y0: y1 + 1] = 255

target_mask = np.zeros((target.shape[0], target.shape[1]), dtype='uint8')
target_mask[tx0: tx1 + 1, ty0: ty1 + 1] = 255


kernel = np.ones((3, 3), dtype='uint8')
erosion = cv2.erode(target_mask, kernel, iterations=1)
edge = target_mask - erosion


cnt = 0
A = lil_matrix((n * m, n * m))

ind = np.zeros((m, n), dtype='uint32')
for i in range(0, m):
    for j in range(0, n):
        ind[i, j] = cnt
        cnt = cnt + 1
cnt = 0

edge_ind = np.argwhere(edge > 0)

for i in range(m):
    for j in range(n):
        if i == 0 or i == m - 1 or j == 0 or j == n - 1:
            A[cnt, ind[i, j]] = 1
            cnt = cnt + 1

for i in range(m):
    for j in range(n):
        if i != 0 and i != m - 1 and j != 0 and j != n - 1:
            A[cnt, ind[i, j]] = 4
            A[cnt, ind[i + 1, j]] = -1
            A[cnt, ind[i, j + 1]] = -1
            A[cnt, ind[i - 1, j]] = -1
            A[cnt, ind[i, j - 1]] = -1
            cnt = cnt + 1

cnt = 0
res = np.zeros((m, n, 3), dtype='uint8')
A = scipy.sparse.csr_matrix(A)

for channel in range(3):
    source_channel = np.zeros((source.shape[0], source.shape[1]), dtype='uint8')
    source_channel = source[:, :, channel]
    target_channel = np.zeros((target.shape[0], target.shape[1]), dtype='uint8')
    target_channel = target[:, :, channel]
    B = np.zeros((n * m, 1), dtype='float32')
    cnt = 0
    for i in range(m):
        for j in range(n):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                B[cnt, 0] = target_channel[i + tx0, j + ty0]
                cnt = cnt + 1

    for i in range(0, m):
        for j in range(0, n):
            if i != 0 and i != m - 1 and j != 0 and j != n - 1:
                dx = i + x0
                dy = j + y0
                B[cnt, 0] = get_laplacian(source_channel, dx, dy)
                cnt = cnt + 1

    x, istop, itn, normr = scipy.sparse.linalg.lsqr(A, B)[:4]
    x = x.reshape((n * m, 1))
    x[x < 0] = 0
    x[x > 255] = 255
    cnt = 0
    for i in range(m):
        for j in range(n):
            res[i, j, channel] = x[cnt]
            cnt = cnt + 1

    cnt = 0


for i in range(m):
    for j in range(n):
        for channel in range(3):
            target[i + tx0, j + ty0, channel] = res[i, j, channel]

show_image(target)
