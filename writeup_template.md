# **Behavioral Cloning** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior on Track 1 and Track 2
* Build a convolution neural network in Keras that predicts steering angles from images
* Choose a proper way of image preprocessing 
* Train and validate the model with a training and validation set originated from Track 1 and Track 2
* Test that the model successfully drives around both tracks without leaving the road
* Test that the model successfully generalizes to a previous version of Track 2 that wasn't used for training
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "CNN architecture"
[image2]: ./examples/track1HSVYUV.PNG "Track 1 HSV YUV"
[image3]: ./examples/track2HSVYUV.PNG "Track 2 HSV YUV"
[image4]: ./examples/tracksSV.png "Track 1&2 SV"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[NVIDIA]: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes 3 folders **tracks**, **tracks30**, **track1fast** with the following files in each:
* model.py containing the script to create and train the model
* model.ipynb using model.py to run the pipeline step by step and build various plots
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

**tracks** is a main folder, while **tracks30** and **track1fast** are for reference when presenting complimentary results.

Folders **tracks30** and **tracks** correspond to the same common model trained during 30 and 5 epochs respectively
using data for both Track 1 and Track 2. They additionally include these videos:
* track1_video.mp4 containing video recording of a vehicle driving autonomously around the Track 1
* track2_video.mp4 containing video recording of a vehicle driving autonomously around the Track 2
* track2prev_video.mp4 containing video recording of a vehicle driving autonomously around the older version of Track 2

Folder **track1fast** corresponds to the same model trained during 5 epochs using data from Track 1 only and a parameter
tuned for faster drive. It additionally includes this video:
* track1fast_video.mp4 containing video recording of a vehicle driving autonomously around the Track 1 with a speed equals 24

#### 2. Submission includes functional code
Using the Udacity provided simulators and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
in each of the folders **tracks**, **tracks30** and **track1fast**.
For Track 1 and Track 2 current version of **windows_sim.exe** was used and **Default Windows desktop 64-bit.exe** for older Track 2. 

#### 3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The code consists of three parts:
* Initialization of parameters (**lines 1-13**)
* Definition of the class BehavioralCloning encapsulating the solution pipeline (**lines 14-190**)
* The code instantiating the model and running the training/validation process (**lines 192-197**)

Additionally the **model.ipynb** lets run the pipeline step by step and get visualisations used in this report.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've implemented the model in the ```__defineCNN``` method (**lines 46-61**) as described in [NVIDIA article][NVIDIA] 

![alt text][image1]

The data is normalized in the model using a Keras lambda layer (**line 18**). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (**model.py lines 57**). 

The model was trained and validated on different data sets (**code lines 158-167**). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the Track 1 and Track2.
Additionally the model was tested on an older version of Track 2 that wasn't used for training\validation.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (**model.py line 171**).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road on both Track 1 and 2 and consists of Track 1 data provided in the course resources and Track 2 data recorded by myself through the simulator. On Track 2 I used to follow central lane line and didn't perform any special movements for central recovery training since recordings from side cameras were used for this task. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it as simple as possible so that I have a more clear sence of how things work introducing small updates if necessery.

My first step was to implement all parts presented in the Behavioral Cloning lesson and use a convolution neural network model similar to the one developed by [NVIDIA][NVIDIA] as I was curious about how this model might work in a simulated environment.

The next step was to introduce an appropriate way of image preprocessing since as I learned from my previous projects it has a huge influence. I decided to try moving to other color spaces so I picked a few representative images from Track 1 and Track 2 and outputed them by layers of HSV and YUV colorspaces:

*Track 1*
![alt text][image2]

*Track 2* 
![alt text][image3]

As it can be seen from these images the S layer of HSV is the best to distinguish the road in a quite common way for various road textures. At the same time the V layer of HSV looks to be the best to distinguish lane lines.

![alt text][image4]


Experiments showed that for Track 1 representing the image as a single S layer was enough same as V layer was enough for Track 2, but neither of them was enough for both Track 1 and Track 2 so I decided to represent the input images in 2 layers - S and V. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
