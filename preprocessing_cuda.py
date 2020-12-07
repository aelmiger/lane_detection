# -*- coding: utf-8 -*-
"""
 
 Title: Pre Processing CUDA
 Author: Anton Elmiger
 Created: 2020-5-4

 Information: Functions for Image Preprocessing on CUDA Cores

"""

import cv2
import constants as const


#creating a CUDA Mat from Image
def create_cuda_mat(img):
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-capable device is detected")

    cuda_mat = cv2.cuda_GpuMat()
    cuda_mat.upload(img)
    return cuda_mat

#blur a CUDA Mat
def blur(cuda_mat,blur_strength = 5):
    blur_filter = cv2.cuda.createGaussianFilter(cuda_mat.type(), -1, (blur_strength, blur_strength), 16)
    return blur_filter.apply(cuda_mat)

#convert a CUDA Mat to Gray
def convert_to_gray(cuda_mat):
    return cv2.cuda.cvtColor(cuda_mat, cv2.COLOR_RGB2GRAY)

#convert a CUDA Mat to binary with threshold
def convert_to_binary(cuda_mat, threshold=150):
    retval, cuda_mat = cv2.cuda.threshold(cuda_mat, threshold,255,cv2.THRESH_BINARY)
    return cuda_mat

#undistort a CUDA Mat (Maps have to be computed in advance)
def undistort_img(cuda_mat,cuda_map1, cuda_map2):
    return cv2.cuda.remap(cuda_mat, cuda_map1, cuda_map2, cv2.INTER_LINEAR)

def bev(cuda_mat, cuda_bev_mat):
    cuda_mat1 = cv2.cuda_GpuMat()
    cuda_mat1 = cv2.cuda.warpPerspective(cuda_mat, cuda_bev_mat, (const.IMAGE_WIDTH,const.IMAGE_HEIGHT))
    return cuda_mat1

def create_sobel_filter(direction = 0):
    if direction == 1:
        #horizontal sobel-edge detection
        sobel_filter = cv2.cuda.createSobelFilter(cv2.CV_8U,cv2.CV_8U, 1, 0, ksize=const.EDGE_SIZE)

    elif direction == 2:
        #vertical sobel-edge detection
        sobel_filter = cv2.cuda.createSobelFilter(cv2.CV_8U,cv2.CV_8U, 0, 1, ksize=const.EDGE_SIZE)

    else:
        #bidirectional sobel-edge detection
        sobel_filter = cv2.cuda.createSobelFilter(cv2.CV_8U,cv2.CV_8U, 1, 1, ksize=const.EDGE_SIZE)
    # return sobel_filter.apply(cuda_mat)
    return sobel_filter
