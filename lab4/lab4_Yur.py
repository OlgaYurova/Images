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


img_my = cv.imread('images/rtYCF3oqDcs.jpg')

gray_image1 = cv.cvtColor(img_my, cv.COLOR_BGR2GRAY)

# 1 Бинаризация полутоновых изображений

threshold = 190
image = gray_image1

ret, thresh1 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

plt.imshow(thresh1, 'gray', vmin=0, vmax=255)
plt.xticks([])
plt.yticks([])

plt.show()