import cv2
from matplotlib import pyplot as plt
import numpy as np

const_d = 15

def fix(img):
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def show_image(img):
    img = fix(img)
    img = img.astype('uint8')
    plt.imshow(img)
    plt.show()
    plt.imsave('res02.jpg', img)

def get_d_distance_mat(mask, d):
    res = mask.copy()
    kernel = np.ones((3, 3), dtype='uint8')
    for i in range(d):
        dilation = cv2.dilate(mask, kernel, iterations=1)
        edge = dilation - mask
        value = (-255 / d) * i + 255
        res[edge > 0] = value
        mask = dilation.copy()
    return res


def feathering(lake, bear, mask):
    ones = np.ones(mask.shape, dtype='uint8')
    return bear * mask + (ones - mask) * lake

def laplacian_pyramid(lake, bear, mask, level):
    if level == 4:
        mask = get_d_distance_mat(mask, const_d)
        lake = cv2.blur(lake, (5, 5))
        bear = cv2.blur(bear, (5, 5))
        return feathering(lake, bear, mask / 255)
    lake_blur = cv2.blur(lake, (5, 5))
    lake_lap = lake - lake_blur
    bear_blur = cv2.blur(bear, (5, 5))
    bear_lap = bear - bear_blur
    half_lake = cv2.resize(lake, (-1, -1), fx=0.5, fy=0.5)
    half_bear = cv2.resize(bear, (-1, -1), fx=0.5, fy=0.5)
    half_mask = cv2.resize(mask, (-1, -1), fx=0.5, fy=0.5)
    res = laplacian_pyramid(half_lake, half_bear, half_mask, level + 1)
    res = cv2.resize(res, (-1, -1), fx=2, fy=2)
    mask = get_d_distance_mat(mask, const_d)
    lap = feathering(lake_lap, bear_lap, mask / 255)
    res = res + lap
    return res


lake = cv2.imread('resources/lake4.jpg')
lake = cv2.cvtColor(lake, cv2.COLOR_BGR2RGB)
lake = lake.astype('float32')

bear = cv2.imread('resources/bear1.jpeg')
bear = cv2.cvtColor(bear, cv2.COLOR_BGR2RGB)
bear = bear.astype('float32')

bear = cv2.resize(bear, (384, 320))
lake = cv2.resize(lake, (384, 320))

translation_matrix = np.float32([[1, 0, -100], [0, 1, 60]])
bear = cv2.warpAffine(bear, translation_matrix, (bear.shape[1], bear.shape[0]))

mask = np.zeros(bear.shape, dtype='uint8')
mask = cv2.fillConvexPoly(mask, np.int32([(10, 200), (150, 200), (150, 300), (10, 300)]), (255, 255, 255))


res = laplacian_pyramid(lake, bear, mask, 0)
show_image(res)
