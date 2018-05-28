'''
========================================================================================================
This file contains the procedure for detecting lane lines
1. Detect the lane lines
2. Search within a specific window around the previous location
3. Draw the lane lines
========================================================================================================
'''

import numpy as np
import cv2

# Detect Lane Lines
def detection(binary_warped, left_lane, right_lane):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # update the line status
    left_lane.clearBuffer()
    right_lane.clearBuffer()
    left_lane.update(nonzerox, nonzeroy, left_lane_inds)
    right_lane.update(nonzerox, nonzeroy, right_lane_inds)
    
    return left_lane, right_lane


# Look-Ahead Filter:
# Skip the sliding windows step once you know where the lines are
# search within a window around the previous detection
def lookAhead(binary_warped, left_lane, right_lane):
    left_fit = left_lane.average_fit
    right_fit = right_lane.average_fit
    
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # update the line status
    left_lane.update(nonzerox, nonzeroy, left_lane_inds)
    right_lane.update(nonzerox, nonzeroy, right_lane_inds)
    
    return left_lane, right_lane


# Compute the offset from center of the infered lane lines
# We assume that the camera capturing the video is mounted in the midpoint of car. Thus, we can compute the deviation
# from the lane center as the distance between the midpoint of frame and that of the lane detected. 
def compute_offset_from_center(left_lane, right_lane, img_width=1280):

    midpoint = img_width / 2
    left_bot = np.mean(left_lane.allx[left_lane.ally > 0.95 * left_lane.ally.max()])
    right_bot = np.mean(right_lane.allx[right_lane.ally > 0.95 * right_lane.ally.max()])
    lane_center = (left_bot + right_bot) / 2
    offset_pix = abs((left_bot + right_bot)/2 - midpoint)
    offset_meter = left_lane.xm_per_pix * offset_pix
    
    return offset_meter


def draw(left_lane, right_lane, undist, M):
    left_fit = left_lane.average_fit
    right_fit = right_lane.average_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, 719, 720)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, np.mat(M).I, (warp_zero.shape[1], warp_zero.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # add text (curvature and offset info) on the upper right of the blend
    # mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    lane_curve = (left_lane.measure() + right_lane.measure()) / 2
    offset = compute_offset_from_center(left_lane, right_lane)
    cv2.putText(result, 'Curvature radius: {:.02f}m'.format(lane_curve), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'offset: {:.02f}m'.format(offset), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    return result