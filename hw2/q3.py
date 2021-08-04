import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


def warp(img, mat):
    mat = np.linalg.inv(mat)
    A = np.zeros((img.shape[0] * img.shape[1] * 3, 9), dtype='float32')
    X = np.float32([[mat[0, 0]], [mat[0, 1]], [mat[0, 2]], [mat[1, 0]], [mat[1, 1]], [mat[1, 2]], [mat[2, 0]],
                    [mat[2, 1]], [mat[2, 2]]])

    cnt = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            A[cnt, 0] = i
            A[cnt, 1] = j
            A[cnt, 2] = 1
            cnt = cnt + 1
            A[cnt, 3] = i
            A[cnt, 4] = j
            A[cnt, 5] = 1
            cnt = cnt + 1
            A[cnt, 6] = i
            A[cnt, 7] = j
            A[cnt, 8] = 1
            cnt = cnt + 1

    B = np.matmul(A, X)
    cnt = 0
    ret = np.zeros((img.shape[1], img.shape[0], 3), dtype='float32')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B[cnt] /= B[cnt + 2]
            B[cnt + 1] /= B[cnt + 2]
            for channel in range(3):
                a = B[cnt + 1] - math.floor(B[cnt + 1])
                b = B[cnt] - math.floor(B[cnt])
                x = math.floor(B[cnt + 1])
                y = math.floor(B[cnt])
                try:
                    K = np.float32([
                        [img[x, y, channel], img[x, y + 1, channel]],
                        [img[x + 1, y, channel], img[x + 1, y + 1, channel]]
                    ])
                    L = np.zeros((1, 2), dtype='float32')
                    L[0, 0] = 1 - a
                    L[0, 1] = a
                    T = np.matmul(L, K)
                    G = np.zeros((2, 1))
                    G[0, 0] = 1 - b
                    G[1, 0] = b
                    T = np.matmul(T, G)
                    ret[j, i, channel] = T[0]
                except:
                    if img.shape[0] > x > 0 and y < img.shape[1] and y > 0:
                        ret[i, j, channel] = img[x, y, channel]
            cnt = cnt + 3
    return ret


def transform(p, im, fileName):
    book = np.float32(p)
    x = im.shape[0]
    y = im.shape[1]
    original = np.float32([[0, 0], [x - 1, 0], [0, y - 1], [x - 1, y - 1]])
    matrix = cv2.findHomography(book, original)[0]
    ans = warp(im, matrix)
    ans = ans.astype('uint8')
    plt.imshow(ans)
    plt.show()
    plt.imsave(fileName, ans)


image = cv2.imread('resources/books.jpg').astype('float32')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


transform([[362, 743], [155, 707], [408, 464], [206, 426]], image, 'res05.jpg')

transform([[669, 209], [602, 394], [384, 106], [318, 288]], image, 'res04.jpg')

transform([[814, 967], [613, 1101], [617, 669], [424, 792]], image, 'res06.jpg')

