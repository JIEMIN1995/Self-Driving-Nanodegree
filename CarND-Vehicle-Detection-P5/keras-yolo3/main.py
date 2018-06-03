'''
=====================================================================================
How to download the pretrained weights, please check the readme file in keras-yolo3.
=====================================================================================
'''

# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy
import argparse
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from yolo import *
from PIL import Image


# Define the pipeline
def pipeline(arr):

    img = Image.fromarray(arr)
	
    yolo = YOLO()
    result = yolo.detect_image(img)
    result = numpy.array(result)		
    # yolo.close_session() 

    return result


if __name__=='__main__':
    parser = argparse. ArgumentParser(description='Detect the lane lines')
    parser.add_argument('--mode', type=str, help='Choose the mode of processing image or video', default='video')
    parser.add_argument('--path', type=str, help='The path of image or video', default='../project_video.mp4')
    args = parser.parse_args()
    
    if args.mode=='img':
        img = Image.open(args.path)
        yolo = YOLO()
        result = yolo.detect_image(img)
        yolo.close_session()
        result.save('../output_images/testpy.jpg')
    elif args.mode=='video':
        output = '../project_output_subclip.mp4'
        yolo = YOLO()
        original_clip = VideoFileClip(args.path).subclip(6,7)
        output_clip = original_clip.fl_image(yolo.detect_image)
        output_clip.write_videofile(output, audio=False)
        yolo.close_session()
        
    
