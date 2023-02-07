# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:00:54 2022

@author: sujal
"""

import numpy as np
import cv2
from scores import *
from part2 import *

#getting image name inputs
leftImage = str(input('Enter name of left image (eg. barn_left.ppm): '))
rightImage = str(input('Enter name of right image (eg. barn_right.ppm): '))

#setting number of levels for multi-resolution
levels = int(input('Enter number of levels for multi-resolution (eg. 3): '))

#setting template size
templateSize = int(input('Enter the template size (eg. 4): '))

#setting window size for image
window = int(input('Enter the window size (eg. 100): '))

#function to read original images and make a copy
def OriginalImages():
    #changing images here to test
    left = cv2.imread(leftImage)
    right = cv2.imread(rightImage)
    originalLeft = left.copy()
    originalRight = right.copy()
    return originalLeft, originalRight

#function to create multi-resolution images based on number of levels provided
def resolution(image, levels):
    h, w, c = image.shape
    outputImage = image
    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]
    return outputImage
               
def initImage():
        left, right = OriginalImages()
    
        left = resolution(left, levels)
        right = resolution(right, levels)
        cv2.imwrite('1. Multi-resolution Left image.jpg', left)
        cv2.imwrite('2. Multi-resolution Right image.jpg', right)  
        cv2.imshow('Input left image', left)
        cv2.imshow('Input right image', right)
    
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
        return left, right, templateSize, window
    
    
#stereo matching using Sum of Square Differences
def ssd():  
        #computing disparity maps of the left and right images
        left, right, templateSize, window = initImage()
        leftDisparity = np.abs(disparity_ssd(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
        rightDisparity = np.abs(disparity_ssd(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    
        #scaling disparity maps
        leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        return leftDisparity, rightDisparity
    
#stereo matching using Sum of Absolute Differences
def sad(): 
        #computing disparity maps of the left and right images
        left, right, templateSize, window = initImage()
        leftDisparity = np.abs(disparity_sad(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
        rightDisparity = np.abs(disparity_sad(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    
        #scaling disparity maps
        leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        return leftDisparity, rightDisparity
        
#stereo matching using normalized correlation
def ncc():   
        #computing disparity maps of the left and right images
        left, right, templateSize, window = initImage()
        leftDisparity = np.abs(disparity_ncorr(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
        rightDisparity = np.abs(disparity_ncorr(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    
        #scaling disparity maps
        leftDisparity = cv2.normalize(leftDisparity, leftDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
        rightDisparity = cv2.normalize(rightDisparity, rightDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
    
        return leftDisparity, rightDisparity

#function to select type of score   
def scoreSelect():
        score = input('Select a matching score: 1 for SSD, 2 for SAD, 3 for NCC: ')
    
        if score == '1':
            left, right = ssd()
            cv2.imwrite('3. Left Disparity SSD.jpg', left)
            cv2.imwrite('4. Right Disparity SSD.jpg', right)
            cv2.imshow('Left Disparity SSD', left)
            cv2.imshow('Right Disparity SSD', right)
            validity(left, right, score)
            averaging(left,right,templateSize, score)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
        elif score == '2':
            left, right = sad()
            cv2.imwrite('5. Left Disparity SAD.jpg', left)
            cv2.imwrite('6. Right Disparity SAD.jpg', right)
            cv2.imshow('Left Disparity SAD', left)
            cv2.imshow('Right Disparity SAD', right)
            validity(left, right, score)
            averaging(left, right, templateSize, score)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
        elif score == '3':
            left, right = ncc()
            cv2.imwrite('7. Left Disparity NCC.jpg', left)
            cv2.imwrite('8. Right Disparity NCC.jpg', right)
            cv2.imshow('Left Disparity NCC', left)
            cv2.imshow('Right Disparity NCC', right)
            validity(left, right, score)
            averaging(left, right, templateSize, score)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print ("Select a valid matching score")

scoreSelect()
