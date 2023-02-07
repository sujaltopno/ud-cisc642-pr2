# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:26:54 2022

@author: sujal
"""


import cv2
import numpy as np

# Validity check of the two images
def validity(left, right, score):
    r1, c1 = left.shape
    r2, c2 = right.shape

    # Validate left image by calculating left - right image disparities
    for i in range(0, r1, 1):
        for j in range(0, c1, 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0

    # Validate left image by calculating right - left image disparities
    for i in range(0, r2, 1):
        for j in range(0, c2, 1):
            if right[i, j] != left[i, j]:
                right[i, j] = 0
    cv2.imshow('Validated Left Image', left)
    cv2.imshow('Validated Right Image', right)
    if score == '1':
        cv2.imwrite('SSD Validated Left Image.jpg', left)
        cv2.imwrite('SSD Validated Right Image.jpg', right)
    elif score == '2':
        cv2.imwrite('SAD Validated Left Image.jpg', left)
        cv2.imwrite('SAD Validated Right Image.jpg', right)
    elif score == '3':
        cv2.imwrite('NCC Validated Left Image.jpg', left)
        cv2.imwrite('NCC Validated Right Image.jpg', right)    
    


# Averaging is performed in the neighborhood to fill the gaps (zeroes)
def averaging(left, right, templateSize, score):
    kernel = np.ones((templateSize, templateSize), np.float32) / (templateSize^2)
    left = cv2.filter2D(left, -1, kernel)
    right = cv2.filter2D(right, -1, kernel)
    cv2.imshow('Averaged Left Image', left)
    cv2.imshow('Averaged Right Image', right)
    if score == '1':
        cv2.imwrite('SSD Averaged Left Image.jpg', left)
        cv2.imwrite('SSD Averaged Right Image.jpg', right)
    elif score == '2':
        cv2.imwrite('SAD Averaged Left Image.jpg', left)
        cv2.imwrite('SAD Averaged Right Image.jpg', right)
    elif score == '3':
        cv2.imwrite('NCC Averaged Left Image.jpg', left)
        cv2.imwrite('NCC Averaged Right Image.jpg', right)