# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1_undistort.jpg "Road Transformed"
[image3]: ./output_images/color_binary.jpg "Thresholded Binary Example"
[image4]: ./output_images/combined_binary.jpg "Binary Example"
[image5]: ./output_images/warped.jpg "Warp Example"
[image6]: ./output_images/color_fit_lane.jpg "Fit Visual"
[image7]: ./output_images/histogram.jpg "histogram"
[image8]: ./output_images/lookAhead.jpg "look ahead"
[image9]: ./output_images/testpy.jpg "result"
[video1]: ./project_output.mp4 "Video"

---

# **README**


### **Camera Calibration**

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for obtaining the objpoints and imgpoints is contained in lines 18 through 32 of the file called 'calibration.py'.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

And then the code for distortion correction is contained in lines 18 through 29 of the file 'preprocessImg.py'. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]

### **Pipeline (single images)**

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in lines 32 through 47 in the file `preprocessImg.py`.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

src = np.float32(np.array([[1280,720], [0,720], [546,460], [734,460]]))

dst = np.float32(np.array([[1280,720], [0,720], [0,0], [1280,0]]))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280, 720     | 1280, 720     | 
|    0, 720     |    0, 720     |
|  546, 460     |    0, 0       |
|  734, 460     | 1280, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient and histogram threshold to generate a binary image. And then apply a light morphology to "fill the gaps" in the binary image. The codes for these steps are contained in lines 51 through 87 of the file called 'preprocessImg.py'. More details are explained below:

In the step of color threshold, convert the original image to HLS color space and seperate the S channel, and then threshold the S channel. This method can perform well detecting the yellow lane lines, but often miss the white lines. 

The gradient threshold utilize the characteristic of direction of the lane. It computes derivation in x direction. It can alleviate the issue of detecting white lines to some degree. 

For the lane lines that are in the distance, they are unclear and blurry. The histogram equalization can effectively magnify the contrast of images. As a result, it can help us detect the lane lines in the distance. 

To display the result of each method, I stack the result of S channel, x sobel, histogram equalization in red, green, blue channels respectively. It is illustrated below:

![alt text][image3]

After that, apply a light morphology to remove noises. The finnal binary result is shown below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

As the slides introduce, I implement sliding windows to identify lane-line pixels. Firstly, sum up pixel values along y axis on thresholded binary images. And a hitogram is plotted below:

  ![alt text][image7]

We can easily find the lane lines by checking two peaks in histogram. You can check the code of this step in function detection() in the file of 'Detection.py'. The coordinates x of two peaks in the histogram can be selected as starting points for searching. After determining the starting positions, we search from bottom to top with sliding windows. While sliding the window, it count the number of nonzero pixels within the window. If the number is greater than a specific number, then center the window on the mean position of lane pixels detected and store the locations of  these pixels. Repeat this procedure to step through the windows one by one. Finally, extract all the left and right line pixel positions and fit second order polynomial. The result is displayed below:

   ![alt text][image6]

Once we know where the lines are, we don't need to do a blind search again in the next frame. Thus, I search in a margin around the previous line position. You can check code of this step in function lookAhead() in the file of 'Detection.py'. The search range is visualized below:

   ![alt text][image8]

Furthermore, I defined a Line() class in the file of 'Lane.py' to keep track of all lane lines detected from frame to frame. The class record the x and y positions of current pixels detected and polynomial of recent frames. To smooth the detections, I average over n past measurements to obtain the stable lane positions. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code of calculating curvature of the lane is included in measure() function in Lane class. The equation for radius of curvature is shown below:
#### **$$ R_{curve} = \frac {(1 + (2Ay + B)^2)^{3/2}}{|2A|} $$**
But the calculation is just based on pixel values. Thus I need to repeat this calculation after converting our x and y values to real world space. Here we assume that in real world the lane is about 30 meters long and 3.7 meters wide. After that, we can infer that each pixel represents 30/720 meters in y dimension and 3.7/700 meters in x dimension respectively. Finally, I average the curvature of left and right lane to get a more stabel result. 

The code of calculating the position of vehicle is included in compute_offset_from_center function in the file of 'Detection.py'. We assume that the camera capturing the video is mounted in the midpoint of car. Thus, we can compute the deviation from the lane center as the distance between the midpoint of frame and that of the lane detected. Find the pixels detected on the bottom and compute their mean position, the answer of which can be considered as position of left and right lanes. Then the distance between the midpoint of lane and that of frame can be easily obtained as offset. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 125 through 159 in my code in `Detection.py` in the function `draw()`.  

Here is an example of my result on a test image:

![alt text][image9]

---

### **Pipeline (video)**

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

If you cannot open video, please see the local file (./project_output.mp4)

---

### **Discussion**

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The bigges problem is the detection of lane lines. It can be easily influenced by the shadow and illumination. Except that, the right line is always discontinuous. Thus, a reasonable combination of threshold methods is necessary. Except the color & gradient thresholds suggested, I tried a lot of other techniques. And the histogram equalisation is definitely an useful trick. And the application of smoothing effectively address the issue of shadow. 

However, this pipeline cannot work in challenge video. Because the lane lines don't always exist in each frame, which means that it cannot work only detecting lines. Maybe we can also consider the direction of road. 
