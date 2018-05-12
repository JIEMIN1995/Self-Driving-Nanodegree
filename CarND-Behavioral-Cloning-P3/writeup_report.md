# **Behavioral Cloning Project**

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidiaPipeline.png "Model Visualization"
[image2]: ./examples/track1.jpg 
[image3]: ./examples/center.jpg "Recovery Image"
[image4]: ./examples/left.jpg "Recovery Image"
[image5]: ./examples/right.jpg "Recovery Image"
[image6]: ./examples/left.jpg "Normal Image"
[image7]: ./examples/flip.jpg "Flipped Image"
[image8]: ./examples/model_architectures.png "Model Architecture"
[image9]: ./examples/bn.png "Batchnorm layer"

## **Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implement the Nvidia pipeline in my model. It contains the kernels with size of 3x3 and 5x5 and depths between 24 and 64.

The model includes RELU lyaers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Furthermore, the cropping layer is added to select the region of roads from original image. It aims at removing the unrelated information from the background such as hills or sky. 

![alt text][image8]

#### 2. Attempts to reduce overfitting in the model

The model contains batchnorm layers after each convolutional layer in order to reduce overfitting. It is a fairly effective method.  

![alt text][image9]

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, track 1 and track 2.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try suggested and classical models. 

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it is classical and has been proved that work well in Mnist datasets. But I don't expect it can perform well and directly pass the track1.

As I expected, LeNet is too shallow and unable to fit training data well. Then, I implemented the Nvidia pipeline as suggested. This network has deeper depths being more powerful, and is even implemented in a real car. Thus, I used it as my backbone.

In order to evaluate how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add the Batchnormlization after each convolutional layer to rescale the data. It is a fairly effective method to address the issue of overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as sharp turns or bridges. To improve the driving behavior in these cases, I record more videos especially in such conditions. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-76) which is the same as the Nvidia pipeline except the batchnorm layers. The architecture is shown below:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to steer if the car drifts off to the left or the right. These images show what a recovery looks like starting from the center, left, right side repectively :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would balance the ratio of training images with right turns and left turns. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 98658 number of data points. I then preprocessed this data by normalizing and cropping the region of roads. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the lowest validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
