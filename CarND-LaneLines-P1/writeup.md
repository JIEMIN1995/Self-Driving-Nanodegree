# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image0]: ./examples/grayscale.jpg "Grayscale"
[image1]: test_images_output_segment/solidYellowCurve.jpg
[image2]: test_images_output_segment/solidWhiteRight.jpg
[image3]: test_images_output/solidYellowCurve.jpg
[image4]: test_images_output/solidWhiteRight.jpg


### Reflection

### 1. Describe my pipeline and the modification in draw_lines() function.

My pipeline consisted of 5 steps:
    
    1. Convert the original image to grayscale so as to easily compute the gradients of images. And then apply a gaussian smoothing on that. Because the gassian smoothing can effectively address the issue of noise occuring. 
    
    2. Define our parameters for Canny and apply it to detect the line segments. 
    
    3. Define our four sided polygon to select region of interest(ROI). 

    4. Define the Hough transform parameters to detect the lane edges on image. 
    
    5. Draw the lines detected on image. 

You can see the results below:
![alt test][image1]
![alt test][image2]

As the figures clearly show, the pipeline can only detect the edge segments. Thus I modify the code in draw_lines() function as below:
    
    1. Calculate the slope for each line outputed by Hough Transform.
    
    2. Filter out the slope between the range [-0.2, 0.2]. Since the lane edges on both sides impossibly occur in that range. Furthermore, the trivial slope may lead to infinity result in the later step. 
    
    3. Match the line segments that may belong to the same lane edge by checking whether the slope of two line segments are similar. If the difference is below threshold, we can judge that the two segments both belong to the same line. We set the threshold value to 0.15 here. 
    
    4. Average the position and slope of segments belonging to the same line. 
    
    5. Extrapolate the lines with the mean of position and slope to the top and bottom of the lane. 

After that, you can see two solid and single line on image:
![alt test][image3]
![alt test][image4]


### 2. Identify potential shortcomings with your current pipeline

This pipeline rely on the gradients change to detect the lane line. Thus, it's easy to be influenced by the illumination, shadow. And I guess the weather may also influnce the performance. For example, the rain or snow dropping in the air may be recognized as line. 

Another shortcoming is that the line is still not solid enough. 


### 3. Suggest possible improvements to your pipeline

With respect to the first shortcoming stated above, I think it's hard to perfectly address those problems using the current pipeline. Since the realistic condition is too complex. But we can add some experience in the pipeline. For example, we only preserve the lines of which slopes are within a specific range. And two lane lines should occur in left side and right side respectively. Furthermore, the absolute value of slopes of two lines should be similar. With such limitations, much noise can be filtered out. 

Secondly, a suppression mechanism should be used to prevent the line change dramatically. As a result, the lines drawn on image are steady or move smoothly. 