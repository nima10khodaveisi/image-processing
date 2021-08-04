import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('resources/flowers_blur.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image.astype('float32')

unsharp_mask = np.asarray([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
    ],
    dtype='float32'
)

sharp_image = image.copy().astype('float32')
alpha = [20, 20, 20]
unsharp_result = np.zeros((image.shape[0], image.shape[1], 3), dtype='float32')

for channel in range(3):
    curChannel = np.zeros((image.shape[0], image.shape[1]), dtype='float32')
    curChannel[:, :] = image[:, :, channel]
    unsharp_image = cv2.filter2D(curChannel, -1, cv2.flip(unsharp_mask, -1))
    unsharp_result[:, :, channel] += unsharp_image
    sharp_image[:, :, channel] += (alpha[channel] * unsharp_image)


sharp_image[sharp_image > 255] = 255


unsharp_result *= 2.5
unsharp_result += 125


unsharp_output = unsharp_result.astype('uint8')

plt.imshow(unsharp_output)
plt.imsave('res01.jpg', unsharp_output)

result_image = sharp_image.astype('uint8')

result_image = cv2.medianBlur(result_image, 5)

plt.imshow(result_image)
plt.show()
plt.imsave('res02.jpg', result_image)
