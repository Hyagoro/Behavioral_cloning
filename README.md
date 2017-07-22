# **Behavioral Cloning Project** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The result of this project (Youtube video) can be found [here](https://youtu.be/8h61V5bXU64).

[//]: # (Image References)

[image2]: ./readme_data/center_2017_07_19_21_38_51_955.jpg "Center sample"
[image3]: ./readme_data/Figure_1.png "Augmented Image 1"
[image4]: ./readme_data/Figure_2.png "Augmented Image 2"
[image5]: ./readme_data/Figure_3.png "Augmented Image 3"
[image6]: ./readme_data/left_2017_07_19_21_53_34_198.jpg "Left Image"
[image7]: ./readme_data/right_2017_07_19_21_43_08_971.jpg "Right Image" 

---

#### Data structure

`model.py` - The script used to create and train the model.

`drive.py` - The script to drive the car.

`model.h5` - The model weights.

### Model Architecture and Training Strategy

My model is based on NVIDIA self driving car architecture  : [Nvidia self driving car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 

It consist of five convolutionnal layers (filters : 24, 36, 48, 64, 64), 3 strided convolutional layers with 2x2 strides ans 5x5 kernel and 2 non strided with 3x3 kernel.

The normalization is not included in my architecture (see bellow).

The model includes RELU layers to introduce nonlinearity. 

#### Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and a lot of curves to compensate long straigh forward lines. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### Solution Design Approach

The idea was to train my model one track one, reduce overtfitting, and then add data from track two to generalize the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model adding two dropout layers, one before the flatten layer with 0.2 keep propability and the second before the last full connected layer with 0.5 keep probability.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I just add some new data, specially curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

Here is a visualization of the architecture :

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_2 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================


#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also used the two secondary camera (right and left) with steering correction of 0.2.

![alt text][image6]
![alt text][image7]

There is steps for preprocedding images:
- Crop the top (65 px) and the bottom (20 px) to remove the car and le landscape.
- Apply Gaussian blur to remove for exemple concrete texture complexity
- Add random bright to learn the track with more light or shadow
- And finaly normalize the image by dividing by 255 minus 0.5.

![alt text][image3]
![alt text][image4]
![alt text][image5]

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.
