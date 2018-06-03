# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Apply a deep neural network called YOLO v3 to detect vehicles in this project. Deploy the convolutional neural network that YOLO makes use of. 
* Predict across different scales to solve the porblem of missing small object. 
* Output processing: thresholding by object confidence and non-maximum suppression. 
* Run YOLO on a video stream. 
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/architectures.png
[image2]: ./output_images/predictions.png
[image3]: ./output_images/scales.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./output_images/testpy.jpg
[video1]: ./project_video.mp4

---
### **README**

### **YOLO v3**

#### 1. Explain the architecture of YOLO and interpret the output.

You can check the code of constructing the body of darknet in the file of 'keras-yolo3/yolo3/model.py'. YOLO makes use of fully convolutional neural network which has 75 convolutional layers with skip connections and upsampling layers. And the architecture of darknet is shown below:

![alt text][image1]

Let's consider the example above, the stride of network is 32. Thus, if the input image is 256&times;256, the output image is 8&times;8. It means that we can divide the input image into 13&times;13 cells. 

Note that the output of YOLO is still a feature map. You can interpret this prediction map is that each cell can predict a fixed number of bounding boxes. We can get S&times;S&times;B&times;(5+C) entries in the feature map. B represents the types of anchor boxes. The number 5 means the coordinates including the center coordinates, the dimensions and the objectness score. C means the number of classes and each value represents the corresponding class confidence. The figure below illustates the structure of predictions. 

![alt text][image2]

#### 2. Prediction across different scales. 

YOLO makes predictions across 3 different scales with stride of 32, 16, 8 respectively. In detail, the darknet downsample the image until the first detection layer. If the size of input image is 416&times;416, then scale of first detection layer is 13&times;13. Furthermore, other two scales are obtained by upsampling by a factor of 2 step by step. As a result, we can predict over 3 different scales. The example is shown below: 

![alt text][image3]

Comparing to the old versions of YOLO, it turns out that the upsampling can effectively help the network learn fine-grained features. 

#### 3. Output processing

For an image of size 416&times;416, YOLO can predict (13&times;13 + 26&times;26 + 52&times;52) &times; 3 = 10647 bounding boxes. So we need to judge which bounding box correctly predict the object. There are two steps here: thresholding by object confidence and non-maximum suppression. 

First, we filter out the bounding boxes of which the objectness score is below a threshold. The objectness score reflect the possibility that there exists a object in the current box. 

Second, non-maximum suppression (NMS) intends to cure the repeated detections of the same object. It will compute the intersection of union (IoU) of overlapping bounding boxes and pick one of them with the highest confidence. 

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)

If you cannot open the link, please check the local file './project_output.mp4'. 

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The false positives rate can be suppressed by setting the threshold of object confidence. You can check the code of line 26 in the file 'keras-yolo3/yolo.py'. But the high threshold may leads to missing detections. Thus, I have to trade off between the recall and and false positive. Luckily, YOLO perform fairly good in this video. The false positive rate can be almost zero if we set a appropriate threshold. 

Besides that, the overlapping bounding boxes is an common issue in the task of object detection. It has been widely accepted that the NMS is an useful method to address such problems. The NMS has been provided by tensorflow, you can check the code in line 217 in the file 'keras-yolo3/yolo3/model.py'. The IoU here is set to the default value of 0.5. 

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Compared with the traditional techniques, the deep learning is definitely powerful in such computer vision tasks. Although there exist various conditions of shadow, occlusion and distance in this video, the YOLO still perform well. That's really amazing! 

The biggest problem in this project are the small size of cars occuring in the distance and vechicles under partial occlusion driving in the left lane. We even hardly recognize them with eyes. To alleviate that situation, I input the image with higher resolution of 1280&times;960. The higher resolution bring more information about small objects, which is a practical trick in object detection task. 

