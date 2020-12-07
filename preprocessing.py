# -*- coding: utf-8 -*-
"""
 
 Title: Pre Processing
 Author: Tom-Morten Theiß
 Created: 2019-12-7

 Information: Functions for Image Preprocessing

"""

import cv2
import numpy as np
import constants as const

#function for transformation of an image from rgb (bgr) to grayscale
def rgb2gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#function to load the previously computed  parameters for the calibration and bev transformation
def loadPPData(directory):
    
    mtx = np.load(directory+'/calib_mtx.npy')
    dist = np.load(directory+'/calib_dist.npy')
    bev_matrix=np.load(directory+'/bev_matrix.npy')
    
    ppData = (mtx,dist,bev_matrix)
    camMat = ppData[0] / 2
    camMat[2,2] = 1
    maps = cv2.initUndistortRectifyMap(ppData[0],ppData[1],None,camMat,(640,360) ,cv2.CV_32FC1)
    return (camMat,ppData[1],ppData[2]), maps
    

#function for image calibration (remove distortion)
def undis_img(img,ppDats, maps):
    # return cv2.undistort(img, ppData[0], ppData[1])
    return cv2.remap(img,maps[0],maps[1],cv2.INTER_LINEAR)



#function to execute the image preprocessing steps calibration and birds eye view (bev) transformation
def bev(img,ppData,maps):
    
    ## Undistortion
    img = undis_img(img,ppData,maps)
    
    ## BEV
    # Bildausgangsgröße hinten in Klammer
    bev = cv2.warpPerspective(img,ppData[2],(const.IMAGE_WIDTH,const.IMAGE_HEIGHT),borderMode=cv2.BORDER_REPLICATE)
    return bev


#function to extract a specific image section (for sas detection)
def shapeImage(img):
    
    return img[const.X_START:const.X_STOP,const.Y_START:const.Y_STOP]


#function for sobel edge detection in different directions
def sobelEdgeDetection(img,direction=0):
    
    if direction == 1:
        #horizontal sobel-edge detection
        sobelImg = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=const.EDGE_SIZE)
        # cv2.imshow("Camera",sobelImg)
        # cv2.waitKey(1)

    elif direction == 2:
        #vertical sobel-edge detection
        sobelImg = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=const.EDGE_SIZE)
    else:
        #bidirectional sobel-edge detection
        sobelImg = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=const.EDGE_SIZE)

    
    # sobelImg = (np.array(sobelImg)>const.SOBEL_THRESHOLD)
    return sobelImg