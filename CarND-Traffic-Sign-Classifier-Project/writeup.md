# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/samples.png "Samples"
[image2]: ./examples/distribution.png "Distribution"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./NewImage/newSample.png "Traffic Sign 1"
[image5]: ./NewImage/2.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows a random sample for each class from training dataset.

![alt text][image1]

The bar below demonstrates the distribution of training, validation, testing dataset. From the legend in the upper right corner on figure, you can see that the red, green, blue represents training, validation and testing dataset distribution respectively. 

We can see that the distribution of training dataset is uneven, which may leads to bias to those much more common classes. But the good news is that the three datasets almost keep the same distribution. Thus the problem like data shift won't occur while evaluating the trained model. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because we can recognize the traffic signs with only shape and edges. The grayscale can provide the gradients change and contour information. Furthermore, it keeps less channel (only 1), compared with the rgb image of 3 channels. It reduce the compution burden.

Mean subtraction is the most common trick for preprocessing data. This technique has good geometric interpretation that the data points distribute around the origin along each dimension, which makes it easier to find the relationships among features along each dimension. After that, normalize the data so that different dimensions are locate in the same scale. Otherwise, the problem of vanishing gradient and exploding gradient may occur. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected 1		| inputs 400,  outputs 128						|
| Dropout               |                                               |
| Fully connected 2		| inputs 128,  outputs 43						|
| Softmax				|        									    |

Compared with the original LeNet model, I remove the third fc layer and add the dropout layer after fc1 layer. This is a common trick. Two fc layers are enough to do the classification and dropout can effectively alleviate the overfitting. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I still use the adam and keep the learning rate of 0.001. Much experiments and papers prove that Adam has good performance. The batch size is set to 32, which is mainly restricted by the hardware. Another key point is to shuffle the training in each epoch. It can make the model see different order of batches of data at each epoch so as to improve the robustness. The epoch is set to 30 that is enough for model to fit the training data. And the keep probability of dropout is set to default value of 0.5.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.950 
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  At first, I still kept the LeNet architecture. Although this technique is old, Lenet is     classical and achieved the state of art in 2012. It is a good choice for the initial architecture that can be considered as the baseline. 

* What were some problems with the initial architecture?

  There is no need to apply 3 fc layers and the overfitting is kind of severe. 

* How was the architecture adjusted and why was it adjusted? 

  I remove the third fc layer. I think two fc layers are enough to do the classficaition job for such a small dataset. And the dropout layer is added after fc1 layer to alleviate the overfitting. 

* Which parameters were tuned? How were they adjusted and why?

  The weights and bias of conv and fc layers are tuned. They were updated by the backpropagation to find a better solution for the objective function of cross entropy. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  Convolution layer has been proved that perform well for processing image. Two key features  of conv layer are the exploitation of local correlation and mechanism of sharing weights. Furthermore, as the architecture go deeper, it can extract more fine-grained features. That is fairly great for helping recognizing the item and understanding the image. 
  
  Dropout randomly select part of features and discard the rest. This technique is simple and improve the uncertainty in model, which promote the robustness as well. From the intuition behind, we don't need to keep all the features that may contain much useless information. Thus, we only randomly keep some of them to filter out those redundant information. At the same time, it plays a role of feature selection. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

These five images all have good clarity. But the fourth one of 80km/h speed limit may be more difficult because the sign just takes up part of image. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| Speed limit (20km/h)							| 
| Stop       			| Stop 							            	|
| No entry				| No entry										|
| Speed limit (80km/h)	| Speed limit (80km/h)				 			|
| Ahead only		    | Ahead only								    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.6%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (20km/h) sign (probability of 1.0), and the image does contain a Speed limit (20km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (20km/h)  						| 
| 0.00     				| Wild animals crossing						    |
| 0.00					| Speed limit (30km/h) 						    |
| 0.00	      			| Bicycles crossing				 			    |
| 0.00				    | Right-of-way at the next intersection		    |

For the second image, the model is relatively sure that this is a stop sign (probability of 0.86), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.86         			| Stop sign   						    		| 
| 0.12     				| Speed limit (70km/h)					    	|
| 0.01					| Speed limit (120km/h)						    |
| 0.01	      			| Speed limit (30km/h)				 	    	|
| 0.01				    | Speed limit (80km/h) 						    |

For the third image, the model is relatively sure that this is a No entry (probability of 1.00), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry   			    					| 
| 0.00     				| No passing									|
| 0.00					| Speed limit (20km/h)		    				|
| 0.00	      			| Speed limit (30km/h)			    	    	|
| 0.00				    | Speed limit (50km/h)				    		|

For the fourth image, the model is relatively sure that this is a Speed limit (80km/h) (probability of 0.77), and the image does contain a Speed limit (80km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.77         			| Speed limit (80km/h)		    		    	| 
| 0.22     				| Speed limit (50km/h)			    			|
| 0.01					| Speed limit (60km/h)			        		|
| 0.00	      			| Speed limit (100km/h)					    	|
| 0.00				    | Wild animals crossing						    |

For the fifth image, the model is relatively sure that this is a Ahead only sign (probability of 1.0), and the image does contain a Ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only  			    					| 
| 0.00     				| Priority road				    				|
| 0.00					| Children crossing				    			|
| 0.00	      			| Road work					 		      		|
| 0.00				    | Speed limit (60km/h)           		    	|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


