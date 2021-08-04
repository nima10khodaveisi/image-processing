import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def log(im):
    im[im > 0] = np.log(im[im > 0])
    return im


def scale(img):
    im = img.copy()
    mn = np.min(im)
    mx = np.max(im)
    if mx == mn:
        im = 128
        return im
    m = 255 / (mx - mn)
    im = m * im - m * mn
    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype('uint8')
    return im


def transpose_mult(mat):
    transpose_mat = np.transpose(mat)
    return np.matmul(mat, transpose_mat)


def forward_transform(im):
    im_fft = np.fft.fft2(im)
    shifted_fft = np.fft.fftshift(im_fft)
    return shifted_fft


def inverse_transform(shifted_fft):
    im_fft = np.fft.ifftshift(shifted_fft)
    im = np.fft.ifft2(im_fft)
    im = np.real(im)
    return im


far_image = cv2.imread('q4_01_near.jpg').astype('float32')
far_image = cv2.cvtColor(far_image, cv2.COLOR_BGR2RGB)

close_image = cv2.imread('q4_02_far.jpg').astype('float32')
close_image = cv2.cvtColor(close_image, cv2.COLOR_BGR2RGB)

far_image = cv2.resize(far_image, (far_image.shape[1] * 3, far_image.shape[1] * 3))
close_image = cv2.resize(close_image, (far_image.shape[1], far_image.shape[1]))
plt.imshow(far_image.astype('uint8'))
plt.show()
plt.imsave('q4_04_far.jpg', far_image.astype('uint8'))
plt.imshow(close_image.astype('uint8'))
plt.show()
plt.imsave('q4_03_near.jpg', close_image.astype('uint8'))

height = far_image.shape[0]
width = far_image.shape[1]

gaussian_spatial_low_pass = transpose_mult(cv2.getGaussianKernel(max(width, height), 20))

gaussian_spatial_low_pass = scale(gaussian_spatial_low_pass)

gaussian_spatial_high_pass = 255 - scale(transpose_mult(cv2.getGaussianKernel(max(width, height), 30)))


r = np.std(gaussian_spatial_high_pass)
r = math.floor(r)

plt.imshow(gaussian_spatial_high_pass, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imsave('q4_07_highpass_{}.jpg'.format(r), gaussian_spatial_high_pass, cmap='gray', vmin=0, vmax=255)

s = np.std(gaussian_spatial_low_pass)
s = math.floor(s)

plt.imshow(gaussian_spatial_low_pass, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imsave('q4_08_lowpass_{}.jpg'.format(s), gaussian_spatial_low_pass, cmap='gray', vmin=0, vmax=255)

low_pass_cutoff = 14
for i in range(gaussian_spatial_low_pass.shape[0]):
    for j in range(gaussian_spatial_low_pass.shape[1]):
        mid = (gaussian_spatial_low_pass.shape[0] / 2, gaussian_spatial_low_pass.shape[1] / 2)
        x = (i - mid[0]) * (i - mid[0])
        y = (j - mid[1]) * (j - mid[1])
        if x + y > low_pass_cutoff * low_pass_cutoff:
            gaussian_spatial_low_pass[i, j] = 0

high_pass_cutoff = 5
for i in range(gaussian_spatial_high_pass.shape[0]):
    for j in range(gaussian_spatial_high_pass.shape[1]):
        mid = (gaussian_spatial_high_pass.shape[0] / 2, gaussian_spatial_high_pass.shape[1] / 2)
        x = (i - mid[0]) * (i - mid[0])
        y = (j - mid[1]) * (j - mid[1])
        if x + y <= high_pass_cutoff * high_pass_cutoff:
            gaussian_spatial_high_pass[i, j] = 0


plt.imshow(gaussian_spatial_high_pass, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imsave('q4_09_highpass_cutoff.jpg', gaussian_spatial_high_pass, cmap='gray', vmin=0, vmax=255)

plt.imshow(gaussian_spatial_low_pass, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imsave('q4_10_lowpass_cutoff.jpg', gaussian_spatial_low_pass, cmap='gray', vmin=0, vmax=255)

far_image_final = far_image.copy()

final_image = far_image.copy()

far_image_final_fft = np.zeros((height, width, 3))
close_image_final_fft = np.zeros((height, width, 3))
highpassed_final = np.zeros((width, height, 3), dtype='uint8')
lowpassed_final = np.zeros((width, height, 3), dtype='uint8')


for channel in range(3):
    far_image_channel = np.zeros((height, width), dtype='float32')
    far_image_channel[:, :] = far_image[:, :, channel]

    close_image_channel = np.zeros((height, width), dtype='float32')
    close_image_channel[:, :] = close_image[:, :, channel]

    far_image_fft = forward_transform(far_image_channel)
    print(np.abs(far_image_fft).shape)
    far_image_final_fft[:, :, channel] = (scale(log(np.abs(far_image_fft))))[:, :]

    far_image_fft *= gaussian_spatial_high_pass
    highpassed_final[:, :, channel] = (scale(log(np.abs(far_image_fft))))[:, :]

    close_image_fft = forward_transform(close_image_channel)
    close_image_final_fft[:, :, channel] = (scale(log(np.abs(close_image_fft))))[:, :]
    close_image_fft *= gaussian_spatial_low_pass
    lowpassed_final[:, :, channel] = (scale(log(np.abs(close_image_fft))))[:, :]

    sum_fft = far_image_fft + close_image_fft

    far_image_inverse = inverse_transform(sum_fft)

    far_image_inverse = scale(far_image_inverse)
    final_image[:, :, channel] = far_image_inverse[:, :]

plt.imshow(scale(log(np.abs(far_image_final_fft))))
plt.show()
plt.imsave('q4_06_dft_far.jpg', scale(log(np.abs(far_image_final_fft))))

plt.imshow(scale(log(np.abs(close_image_final_fft))))
plt.show()
plt.imsave('q4_05_dft_near.jpg', scale(log(np.abs(close_image_final_fft))))

lowpassed_final = lowpassed_final.astype('uint8')

plt.imshow(lowpassed_final)
plt.show()
plt.imsave('q4_12_lowpassed.jpg', lowpassed_final)

plt.imshow(highpassed_final)
plt.show()
plt.imsave('q4_11_highpassed.jpg', highpassed_final)

freq_hybrid = 0.5 * highpassed_final.astype('float32') + 0.5 * lowpassed_final.astype('float32')
freq_hybrid = freq_hybrid.astype('uint8')
plt.imshow(freq_hybrid)
plt.show()
plt.imsave('q4_13_hybrid_frequency.jpg', freq_hybrid)

final_image = final_image.astype('uint8')
plt.imshow(final_image)
plt.show()
plt.imsave('q4_14_hybrid_near.jpg', final_image)

final_image = cv2.resize(final_image, (math.floor(width / 4), math.floor(height / 4)))
plt.imshow(final_image)
plt.show()
plt.imsave('q4_15_hybrid_far.jpg', final_image)
