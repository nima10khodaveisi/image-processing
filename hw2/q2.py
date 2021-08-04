import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

answer = []


def normalized_cross_correlation(f, g):
    g_av = g - np.mean(g)
    f_av = f - np.mean(f)
    f_norm = f_av * f_av
    g_norm = g_av * g_av
    one = np.ones((g.shape[0], g.shape[1]), dtype='float32')
    f_norm = cv2.filter2D(f_norm, -1, one)
    ret = (cv2.filter2D(f_av, -1, g_av)) / (np.sqrt(f_norm) * math.sqrt(np.sum(g_norm)))
    return np.abs(ret)


def show_color(match):
    match = match * 128 + 128
    match = match.astype('uint8')

    plt.imshow(match, cmap='gray', vmin=0, vmax=250)
    plt.show()


def template_matching(f, g, p):
    match_r = normalized_cross_correlation(f[:, :, 0], g[:, :, 0])
    match_g = normalized_cross_correlation(f[:, :, 1], g[:, :, 1])
    match_b = normalized_cross_correlation(f[:, :, 2], g[:, :, 2])

    r_threshold = 0.3
    g_threshold = 0.3
    b_threshold = 0.5

    locations = np.where((match_r >= r_threshold) & (match_g >= g_threshold) & (match_b >= b_threshold))

    for loc in zip(*locations[::-1]):
        top_left = (loc[0] - math.floor(p.shape[1] / 2), loc[1] - math.floor(p.shape[0] / 2))
        bottom_right = (loc[0] + math.floor(p.shape[1] / 2), loc[1] + math.floor(p.shape[0] / 2))
        answer.append((top_left, bottom_right))


patch = cv2.imread('resources/patch.png').astype('float32')
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

ship = cv2.imread('resources/Greek_ship.jpg').astype('float32')
ship = cv2.cvtColor(ship, cv2.COLOR_BGR2RGB)

dx = 0.4
while dx <= 1.2:
    resized_patch = cv2.resize(patch, (0, 0), fx=dx, fy=dx)
    resized_patch = cv2.medianBlur(resized_patch, 5)
    template_matching(ship, resized_patch, patch)
    dx += 0.1

ship = ship.astype('uint8')

for (top_corner, bottom_corner) in answer:
    cv2.rectangle(ship, pt1=top_corner, pt2=bottom_corner, color=(0, 0, 255), thickness=1)

plt.imshow(ship)
plt.show()
plt.imsave('res03.jpg', ship)
