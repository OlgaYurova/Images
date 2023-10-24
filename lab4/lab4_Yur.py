# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:50:41 2023

@author: Myata
"""

import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]


img_my = cv.imread('images/rtYCF3oqDcs.jpg')

gray_image1 = cv.cvtColor(img_my, cv.COLOR_BGR2GRAY)

gs = plt.GridSpec(2, 1)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(gs[1])
plt.hist(gray_image1.reshape(-1), 256, range)

plt.show()

# 1 Бинаризация полутоновых изображений

threshold = 190
image = gray_image1

ret, thresh1 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

plt.imshow(thresh1, 'gray', vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])

plt.show()