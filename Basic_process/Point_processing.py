# Brightness and contrast adjustment
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('', 0)

# Brightness adjustment
img_bright = img + 50
img_bright[img_bright > 255] = 255
img_bright = np.uint8(img_bright)

# Contrast adjustment
img_contrast = img * 1.5
img_contrast[img_contrast > 255] = 255
img_contrast = np.uint8(img_contrast)

# Show image
plt.figure()
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original image')

plt.subplot(132)
plt.imshow(img_bright, cmap='gray')
plt.title('Brightness adjustment')

plt.subplot(133)
plt.imshow(img_contrast, cmap='gray')
plt.title('Contrast adjustment')

plt.show()