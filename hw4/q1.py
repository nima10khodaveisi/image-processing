import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import math

texture_image = cv2.imread('resources/texture1.jpg')
texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

h = texture_image.shape[0]
w = texture_image.shape[1]

# result image is 2500 * 2500
h_result, w_result = 2500, 2500

# patch size and overlap
patch_size = 100
overlap_size = 20
image = np.zeros((h_result, w_result, 3), dtype='uint8')


def normalize(img):
    img = img.astype('float32')
    mx = np.max(img)
    mn = np.min(img)
    if mx == mn:
        img[:, :] = 0
        return img
    val = 255 / (mx - mn)
    img = img * val - mn * val
    img = img.astype('uint8')
    return img


def find_random_template_match(img):
    global texture_image, h, w
    template_match_mask = cv2.matchTemplate(texture_image, img, cv2.TM_CCOEFF_NORMED)
    tmpH = template_match_mask.shape[0]
    tmpW = template_match_mask.shape[1]
    template_match_mask = normalize(template_match_mask)
    mask = np.zeros((h, w), dtype='float32')
    mask[0: tmpH, 0: tmpW] = template_match_mask
    mask[0: h, w - patch_size - 1: w] = 0
    mask[h - patch_size - 1: h, 0: w] = 0
    n = randint(1, 10)
    for tc in range(n):
        min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(mask)
        x = max_loc[0]
        y = max_loc[1]
        mask[y, x] = 0
    return y, x


def find_random_template_match_L_shape(img1, img2, common_image):  # img1 is vertical and img2 is horizontal
    global texture_image, h, w
    template_match_mask1 = cv2.matchTemplate(texture_image, img1, cv2.TM_CCOEFF_NORMED)
    template_match_mask2 = cv2.matchTemplate(texture_image, img2, cv2.TM_CCOEFF_NORMED)
    template_match_mask3 = cv2.matchTemplate(texture_image, common_image, cv2.TM_CCOEFF_NORMED)
    mask1 = np.zeros((h, w), dtype='float32')
    tmpH1 = template_match_mask1.shape[0]
    tmpW1 = template_match_mask1.shape[1]
    mask1[0: tmpH1, 0: tmpW1] = template_match_mask1
    mask2 = np.zeros((h, w), dtype='float32')
    tmpH2 = template_match_mask2.shape[0]
    tmpW2 = template_match_mask2.shape[1]
    mask2[0: tmpH2, 0: tmpW2] = template_match_mask2
    mask3 = np.zeros((h, w), dtype='float32')
    mask3[0: template_match_mask3.shape[0], 0: template_match_mask3.shape[1]] = template_match_mask3
    mask = np.zeros((h, w), dtype='float32')

    mask = mask3.copy()
    mask[0: h, 0: w - overlap_size] += mask1[0: h, overlap_size: w]
    mask[0: h - overlap_size, 0: w] += mask2[overlap_size: h, 0: w]

    mask = normalize(mask)
    mask[0: h, w - patch_size - 1: w] = 0
    mask[h - patch_size - 1: h, 0: w] = 0
    n = randint(1, 5)
    for tc in range(n):
        min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(mask)
        x = max_loc[0]
        y = max_loc[1]
        mask[y, x] = 0
    return y, x


def vertical_mix(image1, image2):  # images should be vertical
    n = image1.shape[0]
    m = image1.shape[1]
    color_img = (image1 - image2) * (image1 - image2)
    dp = np.zeros((n, m), dtype='float32')
    par = np.zeros((n, m), dtype='int8')
    par.fill(-1)
    dp.fill(math.inf)
    dx = [-1, -1, -1]
    dy = [-1, 0, 1]
    res_color_image = np.zeros((n, m), dtype='float32')
    for i in range(n):
        for j in range(m):
            value = color_img[i, j, 0] + color_img[i, j, 1] + color_img[i, j, 2]
            res_color_image[i, j] = value
            if i == 0:
                dp[i, j] = value
                par[i, j] = 0
            else:
                for dirx in range(3):
                    for diry in range(3):
                        nx = i + dx[dirx]
                        ny = j + dy[diry]
                        if 0 <= nx < n and 0 <= ny < m:
                            if dp[i, j] > value + dp[nx, ny]:
                                dp[i, j] = value + dp[nx, ny]
                                par[i, j] = ny
    mask = np.zeros((n, m, 3), dtype='uint8')
    ind = 0
    for i in range(m):
        if dp[n - 1, i] < dp[n - 1, ind]:
            ind = i
    for k in range(ind):
        for channel in range(3):
            mask[n - 1, k, channel] = 1
    cur = n - 1
    while cur >= 0:
        ind = par[cur, ind]
        cur = cur - 1
        for k in range(ind):
            for channel in range(3):
                mask[cur, k, channel] = 1
    # ones = np.ones((n, m, 3), dtype='uint8')
    # plt.imshow(image1)
    # plt.show()
    # plt.imshow(image2)
    # plt.show()
    # plt.imshow(mask * image1 + (ones - mask) * image2)
    # plt.show()
    # res_color_image = normalize(res_color_image)
    # res_color_image = res_color_image.astype('uint8')
    # plt.imshow(res_color_image, vmin=0, vmax=255, cmap='gray')
    # plt.show()
    # mask_color_image = np.zeros((n, m, 3), dtype='uint8')
    # for i in range(n):
    #     for j in range(m):
    #         if mask[i, j, 0] > 0:
    #             mask_color_image[i, j, 0] = 255
    #         else:
    #             for ch in range(3):
    #                 mask_color_image[i, j, ch] = res_color_image[i, j]
    # plt.imshow(mask_color_image)
    # plt.show()
    return mask


def horizontal_mix(image1, image2):
    n = image1.shape[0]
    m = image1.shape[1]
    color_img = (image1 - image2) * (image1 - image2)
    dp = np.zeros((n, m), dtype='float32')
    par = np.zeros((n, m), dtype='int8')
    par.fill(-1)
    dp.fill(math.inf)
    dx = [-1, -1, -1]
    dy = [-1, 0, 1]
    for i in range(m):
        for j in range(n):
            value = color_img[j, i, 0] + color_img[j, i, 1] + color_img[j, i, 2]
            if i == 0:
                dp[j, i] = value
                par[j, i] = 0
            else:
                for dirx in range(3):
                    for diry in range(3):
                        nx = i + dx[dirx]
                        ny = j + dy[diry]
                        if 0 <= nx < m and 0 <= ny < n:
                            if dp[j, i] > value + dp[ny, nx]:
                                dp[j, i] = value + dp[ny, nx]
                                par[j, i] = ny
    mask = np.zeros((n, m, 3), dtype='uint8')
    ind = 0
    for i in range(n):
        if dp[i, m - 1] < dp[ind, m - 1]:
            ind = i
    for k in range(ind):
        for channel in range(3):
            mask[k, m - 1, channel] = 1
    cur = m - 1
    while cur >= 0:
        ind = par[ind, cur]
        cur = cur - 1
        for k in range(ind):
            for channel in range(3):
                mask[k, cur, channel] = 1
    return mask


for i in range(0, h_result - patch_size + 1, patch_size - overlap_size):
    for j in range(0, w_result - patch_size + 1, patch_size - overlap_size):
        best_patch = np.zeros((patch_size, patch_size, 3), dtype='uint8')
        if i == 0 and j == 0:
            randomX = randint(0, h - patch_size)  # check if it's [l, r)
            randomY = randint(0, w - patch_size)
            best_patch = texture_image[randomX: randomX + patch_size, randomY: randomY + patch_size, :].copy()
        elif i == 0:
            overlap_before_image = image[i: i + patch_size, j: j + overlap_size, :].copy()
            x, y = find_random_template_match(overlap_before_image)
            overlap_texture_image = texture_image[x: x + patch_size, y: y + overlap_size, :].copy()
            # plt.imshow(texture_image[x: x + patch_size, y: y + patch_size, :])
            # plt.show()
            verticalMask = vertical_mix(overlap_before_image, overlap_texture_image)
            ones = np.ones((patch_size, overlap_size, 3), dtype='uint8')
            overlap_image = (verticalMask * overlap_before_image) + ((ones - verticalMask) * overlap_texture_image)
            best_patch = texture_image[x: x + patch_size, y:y + patch_size, :].copy()
            best_patch[0: patch_size, 0: overlap_size, :] = overlap_image
        elif j == 0:
            overlap_before_image = image[i: i + overlap_size, j: j + patch_size, :].copy()
            x, y = find_random_template_match(overlap_before_image)
            overlap_texture_image = texture_image[x: x + overlap_size, y: y + patch_size, :].copy()
            horizontalMask = horizontal_mix(overlap_before_image, overlap_texture_image)
            overlap_image = np.ones((overlap_size, patch_size, 3), dtype='uint8')
            overlap_image = horizontalMask * overlap_before_image + (
                    overlap_image - horizontalMask) * overlap_texture_image
            best_patch = texture_image[x: x + patch_size, y: y + patch_size, :].copy()
            best_patch[0: overlap_size, 0: patch_size, :] = overlap_image
        else:
            vertical_before_image = image[i + overlap_size: i + patch_size, j: j + overlap_size, :].copy()
            horizontal_before_image = image[i: i + overlap_size, j + overlap_size: j + patch_size, :].copy()
            common_image = image[i: i + overlap_size, j: j + overlap_size, :].copy()
            before_image = image[i: i + patch_size, j: j + patch_size, :].copy()
            x, y = find_random_template_match_L_shape(horizontal_before_image, vertical_before_image, common_image)
            vertical_overlap_texture_image = texture_image[x + overlap_size: x + patch_size, y: y + overlap_size,
                                             :].copy()
            horizontal_overlap_texture_image = texture_image[x: x + overlap_size, y + overlap_size: y + patch_size,
                                               :].copy()
            current_texture_image = texture_image[x: x + patch_size, y: y + patch_size, :].copy()
            # plt.imshow(before_image)
            # plt.show()
            # plt.imshow(current_texture_image)
            # plt.show()
            verticalMask = vertical_mix(vertical_before_image, vertical_overlap_texture_image)
            horizontalMask = horizontal_mix(horizontal_before_image, horizontal_overlap_texture_image)
            mask1 = np.zeros((patch_size, patch_size, 3), dtype='uint8')
            mask1[0: verticalMask.shape[0], 0: verticalMask.shape[1], :] = verticalMask
            mask2 = np.zeros((patch_size, patch_size, 3), dtype='uint8')
            mask2[0: horizontalMask.shape[0], 0: horizontalMask.shape[1], :] = horizontalMask
            mask = np.zeros((patch_size, patch_size, 3), dtype='uint8')
            mask = np.bitwise_or(mask1, mask2)
            ones = np.ones((patch_size, patch_size, 3), dtype='uint8')
            best_patch = mask * before_image + (ones - mask) * current_texture_image
            # plt.imshow(current_texture_image)
            # plt.show()
            # plt.imshow(mask * 255, vmin=0, vmax=255, cmap='gray')
            # plt.show()
        image[i: i + patch_size, j: j + patch_size, :] = best_patch
        # plt.imshow(best_patch)
        # plt.show()
    # plt.imshow(image)
    # plt.show()

plt.imshow(image)
plt.show()
plt.imsave('res01.jpg', image)
