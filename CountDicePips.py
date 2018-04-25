# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:49:01 2018

@author: Allen Askay
"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def CountDicePips(inputPath,outputPath,image):
    """ Count the number of pips on one or more dice in an image file.
    
    Parameters:
        inputPath (string): Path to the source image file.
        outputPath (string): Path to the image output location.
        image (string): Name of the image to process.
        
    Returns:
        Does not return a value.
        The function outputs a copy of the original image annotated with the
        sum of all pips found on all dice as well as the number of pips on
        each dice next to the dice.
    """
    # Read in an image and correct for colorspace
    path = os.path.join(inputPath,image)
    if os.path.isfile(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("No image file found")
        return
    
    # Create a greyscale version of the image        
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     
    # Threshold the image
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Get the contours of the dice
    _,contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    cnts = sorted(contours,key=cv2.contourArea,reverse=True)

    sum_dots = 0
    for c in cnts:
        # Approximate the contours found above and find the area
        approx = cv2.approxPolyDP(c,0.04*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c)

        if ((len(approx)==4) & (area > 1000)): # square contour
            cv2.drawContours(img,[c],0,(0,255,0),3)

            # Find the set of points that 'cover' the die to find the ROI 
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get the region of interest (for the particular die)
            center = ComputeContourCenter(c)
            size = ComputeBoxSize(box)            
            roi = cv2.getRectSubPix(gray,size,center)
            
            # Count the pips on the die
            params = cv2.SimpleBlobDetector_Params()
            
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.5
            
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.75
            
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(roi)            
            num_dots = len(keypoints)
            
            # Calculate an offset for labeling the number of pips by each die
            locationX = int(center[0]+size[0]/2)
            locationY = int(center[1]+size[1]/2)
            location = (locationX,locationY)
            
            # Label the image and accumulate the total number of pips
            cv2.putText(img,str(num_dots),location,0,1.75,(0,255,0), 3, cv2.LINE_AA)
            sum_dots += num_dots
    
    text = 'Sum: ' + str(sum_dots)             
    cv2.putText(img,text,(25,50),0,1.75,(0,255,0), 4, cv2.LINE_AA)
    
    # Plot the result
    plt.imshow(img),plt.show()
    
    outputName = 'output_'+image    
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(outputPath,outputName),img)
      

def ComputeContourCenter(contour):
    """ This function takes in an OpenCV contour object
    and returns the center of the contour by using the
    OpenCV moments function.
    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX,cY)
    return center

def ComputeBoxSize(box):
    """ This function takes in an OpenCV boxPoints
    array and returns the size of the box."""
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    size = (x2-x1, y2-y1)
    return size
