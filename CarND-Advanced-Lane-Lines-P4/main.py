# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import argparse
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from preprocessImg import *
from Detection import *
from Lane import Line

# Define the pipeline
def pipeline(img):
    global detected, left_lane, right_lane
    undist = undistort(img)
    warped, M = unwarp(undist)
    color_binary, combined_binary, closing = threshold(warped)
    if not detected:
        left_lane, right_lane = detection(closing, left_lane, right_lane)
        detected = True
    else:
        left_lane, right_lane = lookAhead(closing, left_lane, right_lane)
    result = draw(left_lane, right_lane, undist, M)
    return result


if __name__=='__main__':
    parser = argparse. ArgumentParser(description='Detect the lane lines')
    parser.add_argument('--mode', type=str, help='Choose the mode of processing image or video', default='img')
    parser.add_argument('--path', type=str, help='The path of image or video', default='test_images/test2.jpg')
    args = parser.parse_args()
    
    # Define line class
    global left_lane, right_lane
    left_lane, right_lane = Line(), Line()
    
    global detected
    detected = False
    
    if args.mode=='img':
        img = mpimg.imread(args.path)
        result = pipeline(img)
        cv2.imwrite('output_images/testpy.jpg',result[:,:,::-1])
    elif args.mode=='video':
        output = 'project_output.mp4'
        original_clip = VideoFileClip(args.path)
        output_clip = original_clip.fl_image(pipeline)
        output_clip.write_videofile(output, audio=False)
        
    
