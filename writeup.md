#**Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup/samples.png "Train samples" 
[image2]: ./writeup/freqs.png "Dataset structure"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./test-web/1.jpg "Speed limit (30km/h) "
[image5]: ./test-web/14.jpg "Stop"
[image6]: ./test-web/17.jpg "No entry"
[image7]: ./test-web/28.jpg "Children crossing"
[image8]: ./test-web/38.jpg "Keep right"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. With this project i've reached about 97% acuracy on validation dataset and about 94% on test dataset. 5 images found on web wereclassified with 80% accuracy (4 of 5).
I started with lenet architecture wich gave about 87%. Then i implemented preprocessing (grayscaling and local normalization) wich lift up accuracy to  about 89%. Then i tried to go wide with classification weights with no luck. Then idecided to increase number of features and go little deep: i doubled number of features on both convolutional layers and added one partially connectected classification level, which gave the final result. I hadn't ability to train nn on GPU, so i used my laptop (CPU:Intel T6600 2.2GHz, RAM:4Gb), so learning cycle took about 30 minutes. 
The submission includes the my [project code](https://github.com/amakurin/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. I used the numpy library to calculate summary statistics of the traffic
signs data set (second code cell):

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. I used numpy and matplotlib to obtain exploratory visualization of the data set.
The code for this step is contained in the third code cell of the IPython notebook. 

Here is an exploratory visualization of the data set. 
I've plotted one sample for each label to understand what i am doing:
![alt text][image1]

and bar chart of frequencies of labels to figure out dataset structure:
![alt text][image2]

Bar chart shows that number of samples significantly differs from label to label. I used this to improve accuracy later.

###Design and Test a Model Architecture

####1. The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because rough analisis of samples shows that structure of image is way more informable that coloring. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]


As a last step, I normalized the image data to get rid of big numbers hoping that it will increase learning accuracy and speed.

####2. Provided dataset already included train, validation, and test data so i used it. Analisis of dataset structure shows huge difference in frequences for different labels. So when i stuckedon 89% accuracy i decided to generate additional samles for all labels with frequency less then mean. I implemented this by copying existing samples and adding distortions in rotation and scaling.

I used uniformly distributed random values for:
* angle [-10,10] degrees
* translation of rotation center [0,6] as distance from origin
* scale factor [0.8, 1.2]

The code for data generation is contained in the forth and fifth code cells of the IPython notebook.  
With this procedure i increased size of training dataset from 34799 to 51448.

To cross validate my model, iused validation set included inprovided data.

My final training set had 51448 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 					|
| Classifier 			| 800 -> 400									|
| Dropout				| train prob 0.5, eval prob 1.0					|
| RELU					|												|
| Classifier			| 400 -> 120									|
| RELU					|												|
| Classifier			| 120 -> 84										|
| RELU					|												|
| Classifier			| 84 -> 43										|
| RELU					|												|

Additional layer of softmax cross entropy used to calculate loss.


####4. The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used default lenet hyperparameters:
* learning rate 0.001
* number of epochs 10
* batch size 128

####5. The code for calculating the accuracy of the model is located in the seventh and eight cells of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.975 
* test set accuracy of 0.942

Based on lectures and paper of Pierre Sermanet and Yann LeCun first i chose plain lenet model which gave me 87%.
Grayscaling and normalization gave 89%.
Sample generation gave 92%.
Then i noticed that learning process don't have enough progress: EPOCH1 80%, EPOCH10 92%. I thought that maybe my model is not capacious enough. I doubled features for both convolutional layers, added one classification layer and dropout hoping that it speed up learning and increase robustness. This gave 97.5% accuracy on validation set, and 94% on test set - final result. This step didn't change learning rate mush, but gave more accuracy.

###Test a Model on New Images

####1. I downloaded five images using goggle picture search "German traffic signs":

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All images have some watermarks on it, "No entry" and "Keep right" have custom view angle that could be a problem during classification.

####2. The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30km/h 				| 30km/h   										| 
| Stop					| Stop 											|
| No entry				| Turn left ahead								|
| Children crossing		| Children crossing								|
| Keep right			| Keep right      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94% taking an account small number (5) of experiments.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code plotting softmax probabilities is located in the 12th cell of the Ipython notebook.

For the first image, the model is completely sure that this is a "speed limit 30" (probability of 0.98), and the image does contain a "speed limit 30" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Speed limit (30km/h)							| 
| .012     				| Speed limit (20km/h) 							|
| 4e-05					| End of speed limit (80km/h)					|
| 9e-06					| Speed limit (80km/h)			 				|
| 3e-06					| Speed limit (50km/h) 							|


For the second image, the model is completely sure that this is a "Stop" sign (probability of 0.99), and the image does contain a "Stop" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| 1e-03    				| Speed limit (30km/h) 							|
| 1e-03					| No vehicles									|
| 9e-04					| No entry						 				|
| 7e-04					| Speed limit (20km/h) 							|

For the third image, the model shows some uncertainty about what is it so winner (probability of 0.5) is "Turn left ahead" which is wrong, and true answer "No entry" is on third place with probability .09. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .5         			| Turn left ahead   							| 
| .25    				| Priority road 								|
| .09					| No entry										|
| .05					| No passing									|
| .04					| Ahead only 									|

My guess the reason is view angle. And i think this can be improve by adding perspective distortion to training data.

For the forth image, the model is completely sure that this is a "Children crossing" sign (probability close to 1.0), and the image does contain a "Children crossing" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >.99         			| Children crossing								| 
| 3e-09    				| Right-of-way at the next intersection 		|
| 2e-09					| Beware of ice/snow							|
| 2e-10					| Traffic signals				 				|
| 3e-11					| Dangerous curve to the left					|

For the forth image, the model is completely sure that this is a "Keep right" sign (probability close to 1.0), and the image does contain a "Keep right" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| >.99         			| Keep right									| 
| 2e-19    				| Yield 										|
| 2e-20					| Stop											|
| 6e-21					| End of all speed and passing limits			|
| 1e-21					| Slippery road									|
