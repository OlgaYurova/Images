# -*- coding: utf-8 -*-
"""
@author: Myata
"""


import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

"""Загружаем изображение. Преобразуем в модель RGB"""

image = cv.imread('images/strawberrycake.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
plt.imshow(image_rgb)
plt.show()

#Первый способ:
"""Создаём маску"""

lower_blue = np.array([0,245,100])
upper_blue = np.array([180,255,255])
#lo_square = np.full((10, 10, 3), lower_blue, dtype=np.uint8) / 255.0
#do_square = np.full((10, 10, 3), upper_blue, dtype=np.uint8) / 255.0
'''
plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(lo_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(do_square))
plt.show()'''

"""Находим на изображении цвета подходящие под маску."""

mask = cv.inRange(image_hsv, lower_blue, upper_blue)
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask)
blur = cv.GaussianBlur(result, (7, 7), 0)

plt.figure(figsize=(15,20))

plt.subplot(1, 3, 1)
plt.xticks([]), plt.yticks([])
plt.imshow(image_rgb)
plt.subplot(1, 3, 2)
plt.xticks([]), plt.yticks([])
plt.imshow(mask, cmap="gray")
plt.subplot(1, 3, 3)
plt.xticks([]), plt.yticks([])
plt.imshow(result)
#plt.imshow(blur)
plt.show()


    
#Второй способ:
"""Цветовая сегментация"""

def segment_image(image):
    ''' Attempts to segment the whale out of the provided image '''

    # Convert the image into HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # Set the blue range
    lower_blue = (0, 0, 0)
    upper_blue = (0, 0, 0)

    # Apply the blue mask
    mask = cv.inRange(hsv_image, lower_blue, upper_blue)

    # Set a white range
    light_white = (0, 245, 100)
    dark_white = (255, 255, 255)

    # Apply the white mask
    mask_white = cv.inRange(hsv_image, light_white, dark_white)

    # Combine the two masks
    final_mask = mask + mask_white
    result = cv.bitwise_and(image, image, mask=final_mask)

    # Clean up the segmentation using a blur
    blur = cv.GaussianBlur(result, (7, 7), 0)
    return blur


result = segment_image(image_rgb)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
 
   
#Третий способ:
    
image3 = cv.imread('images/strawberrycake.jpg')
rgb_image3 = cv.cvtColor(image3, cv.COLOR_BGR2RGB)
hsv_image3 = cv.cvtColor(rgb_image3, cv.COLOR_RGB2HSV)
h, s, v = cv.split(hsv_image3)

low_h = 0
high_h = 0.5

mask1 = cv.inRange(h, low_h, high_h)
result1 = cv.bitwise_and(rgb_image3, rgb_image3, mask=mask1)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(rgb_image3)
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.imshow(mask1, cmap='gray')
plt.title('Маска')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[2])
plt.hist(h.reshape(-1), np.max(h), [np.min(h), np.max(h)])
plt.vlines(low_h, 0, 5000, 'r'), plt.vlines(high_h, 0, 5000, 'r')
plt.title('Гистограмма h слоя')
plt.subplot(gs[3])
plt.imshow(result1)
plt.title('Изображение с пикселями выделенного цвета')
plt.show()