# -*- coding: utf-8 -*-
"""lab2_1"""

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

image1 = cv.imread('images/lenna_bad.png')
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

channels = [0]
histSize = [256]
range = [0, 256]

"""##### 2.1.2.3.2 Эквализация изображения"""

def lut1(gray_image):
    # Вычисляем гистограмму изображения в оттенках серого
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    # Находим сумма элементов массива
    sh = np.sum(hist)
    # Находим накопительную сумму элементов массива
    sum_hist1 = np.cumsum(hist)
    # Выравниваем гистограмму путем деления каждого пикселя на общую сумму гистограммы и умножения на 255
    result_image = 255 * (sum_hist1[gray_image]/sh)
    return result_image


eq_im = lut1(gray_image1)


gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.subplot(gs[1])
plt.imshow(eq_im, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image1.reshape(-1), 256, range)
plt.subplot(gs[3])
plt.hist(eq_im.reshape(-1), 256, range)
plt.show()

