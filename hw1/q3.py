import cv2
from matplotlib import pyplot as plt
import numpy as np


def shift(channel, dx, dy):
    translate = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(channel, translate, (channel.shape[1], channel.shape[0]))
    return dst


def calc_match(channel0, channel1, dx, dy):
    dst = shift(channel1, dx, dy)
    return np.linalg.norm(channel0 - dst)


def pyrDown(channel):
    ret = np.delete(channel, list(range(0, channel.shape[0], 2)), axis=0)
    return np.delete(ret, list(range(0, channel.shape[1], 2)), axis=1)


def find_matching(channel0, channel1):
    if channel0.shape[0] <= 200 and channel0.shape[1] <= 200:
        mx = 0
        my = 0
        diff = calc_match(channel0, channel1, 0, 0)
        for x in range(-200, 200):
            for y in range(-200, 200):
                ndiff = calc_match(channel0, channel1, x, y)
                if ndiff < diff:
                    diff = ndiff
                    mx = x
                    my = y
        return int(mx), int(my)
    pyDown0 = pyrDown(channel0)
    pyDown1 = pyrDown(channel1)
    recursive_ans = find_matching(pyDown0, pyDown1)
    bx = recursive_ans[0]
    by = recursive_ans[1]
    bx *= 2
    by *= 2
    diff = calc_match(channel0, channel1, bx, by)
    for dx in range(-10, 10):
        for dy in range(-10, 10):
            x = dx + bx
            y = dy + by
            ndiff = calc_match(channel0, channel1, x, y)
            if ndiff < diff:
                diff = ndiff
                bx = x
                by = y
    return int(bx), int(by)


image = cv2.imread('resources/melons.tif', -1)

sz = (image.shape[0] / 3)
sz = int(sz)

bChannel = image[0:sz, :]
gChannel = image[sz:2 * sz, :]
rChannel = image[2 * sz:3 * sz, :]

rb_translate = find_matching(rChannel.copy(), bChannel.copy())

bChannel = shift(bChannel.copy(), rb_translate[0], rb_translate[1])

rg_translate = find_matching(rChannel.copy(), gChannel.copy())

gChannel = shift(gChannel.copy(), rg_translate[0], rg_translate[1])

image = np.zeros((rChannel.shape[0], rChannel.shape[1], 3), dtype='uint16')

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image[i, j, 0] = rChannel[i, j]
        image[i, j, 1] = gChannel[i, j]
        image[i, j, 2] = bChannel[i, j]

image = (image / 256).astype('uint8')
plt.imshow(image)
plt.show()
plt.imsave('res04.jpg', image)
