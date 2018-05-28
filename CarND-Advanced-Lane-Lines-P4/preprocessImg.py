'''
=========================================================================================================
This file contains the procedure for preprocessing the image
1. Camera Calibration and Distortion Correction
2. Perspective transform from the bird eye
3. Color & Gradient Threshold
=========================================================================================================
'''

import pickle
import numpy as np
import cv2
import collections
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Camera Calibration and Distortion Correction
def undistort(img):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open('examples/wide_dist_pickle.p','rb'))
    objpoints = dist_pickle['objpoints']
    imgpoints = dist_pickle['imgpoints']
    
    # Compute the camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    # Use the opencv undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

# Perspective transform from the bird eye
def unwarp(img):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    
    # For source points, select the appropriate region of the road
    src = np.float32(np.array([[1280,720], [0,720], [546,460], [734,460]]))
    # For destination points
    dst = np.float32(np.array([[1280,720], [0,720], [0,0], [1280,0]]))
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using opencv warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    # Return the resulting image and matrix
    return warped, M

# Color & Gradient & histogram equalization Threshold
# Apply morphology to "fill the gaps" in binary image
def threshold(img, s_thresh=(170,255), sx_thresh=(35,100)):
    # Convert to HLS color space and seperate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x 
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Apply histogram equalization to an input frame, threshold it and return the (binary) result
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    _, th = cv2.threshold(eq, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    seq_binary = np.zeros_like(s_channel)
    seq_binary[th > 0] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    color_binary = np.dstack((seq_binary, sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (seq_binary == 1)] = 1
    
    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    return color_binary, combined_binary, closing