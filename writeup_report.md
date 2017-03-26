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
[image5]: ./examples/epochs30.png "Loss vs 30 epochs"
[image6]: ./examples/epochs5.png "Loss vs 5 epochs"
[image7]: ./examples/initialSteeringHist.png "Initial steeering histogram"
[image8]: ./examples/undersampledSteeringHist.png "Undersampled steeering histogram"
[image9]: ./examples/augmentedSteeringHist.png "Augmented steeering histogram"


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

Figure shows the network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The data is normalized in the model using a Keras lambda layer (**line 18**). 

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

My first step was to implement all parts presented in the Behavioral Cloning lesson and use a convolution neural network model similar to the one developed by [NVIDIA][NVIDIA] as I was curious about how this model might work in a simulated environment. Following parts were implemented:
* Data loading using Pandas (**line 40**)
* NVIDIA CNN (**lines 47-61**)
* Normalization and cropping (**lines 48-50**)
* Augmenting data by flipping (**lines 130-137**)
* Creating adjusted steering measurements for the side camera images based on ```correction``` parameter (**lines 106-114**) 
* Compiling the model using adam optimizer (**line 171**)
* Training and validating the model using generators (**lines 145-155, 173-179**)
* Outputting Training and Validation Loss Metrics (**lines 182-190**)

Fortunatelly there was no necessity to tune ```correction = 0.2``` parameter that was quite good for the default ```speed = 9``` in the **drive.py**.

The next step was to introduce an appropriate way of image preprocessing since as I learned from my previous projects it has a huge influence. I decided to try moving to other color spaces so I picked a few representetive images from Track 1 and Track 2 and outputed them by layers of HSV and YUV colorspaces:

*Track 1*
![alt text][image2]

*Track 2* 
![alt text][image3]

As it can be seen from these images the S layer of HSV is the best to distinguish the road in a quite common way for various road textures. At the same time the V layer of HSV looks to be the best to distinguish lane lines both under the light and in shadows.

![alt text][image4]

Experiments showed that for Track 1 representing the image as a single S layer was enough same as V layer was enough for Track 2, but neither of them was enough for both Track 1 and Track 2 so I decided to represent the input images in 2 layers - S and V (**model.py lines 117-119**)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
Running the training\validation process on 30 epochs gave the following loss values:

![alt text][image5]

From the plot above we can see that the model becomes more and more overfitted with epochs. 
Corresponding model and videos can be found in **tracks30** folder.
Though the car keeps in both Track 1 and 2, it fails on old Track 2 hitting the fence in about half a way. 
Another probable sign of overfitting is a noticable shaking which didn't occur for low number of epochs.
My current best thinking is that our model is too complex for our data and that there is some space for simplifications of the model.
Also training loss decreesing looks to be somewhat slow.

Aiming to achieve a better generalization I decided to try out number of epochs equal 5. As expected after previous plot,
training and validation losses are closer and we can expect a better generalization.

![alt text][image6]

Corresponding model and results are placed in **tracks** folder and we can now see the vehicle is able to drive autonomously around each track without leaving the road.

The final step was to find out what if I want the car to drive faster. In this case we need to tune ```correction``` parameter down
since the car is now able to reach the center of the road faster and too steep angle of restore will throw the car from one side to another. Setting ```correction = 0.01``` let the car drive at a ```speed = 24``` on Track 1.


#### 2. Final Model Architecture

The final model architecture (**model.py lines 47-61**) consists of a convolution neural network described above with additional
dropout inserted between convolutional and classification layers. The input image is first converted to HSV, then S and V planes are passed to the network. Steering angle is a single output of the model.

#### 3. Creation of the Training Set & Training Process

For Track 1 I've used the data provided in resources. To capture good driving behavior on Track 2, I first recorded 6 laps in forward direction and 6 laps in backward direction and randomly picked half of the images that were then joined to Track 1 data.
A histogram for steering angles on this stage looks like this:

![alt text][image7]

As we can see this data would bias the model towards -1, 0 and 1 angles so we need to undersample these values (**lines 82-103**).
The undersampling depends on m_frac, zero_frac and p_frac parameters for -1, 0, 1 respectively that specify fraction of values to be left during undersampling. For our model ```zero_frac = 0.25```, ```m_frac = 0.6```, ```p_frac = 0.6```.
A histogram for steering angles on undersampling looks like this:

![alt text][image8]

The next step is to augmemnt the data with side cameras images and corresponding adjusted steering measures so that the model learns to recover back to the center of the road (**lines 106-114**):

![alt text][image9]

After the collection process, I had 50538 data points.

Additionally flipped images are added to train and validation data. This augmentation is performed on a batch level for images already loaded into the memory so that we can avoid doubling of disk read operations (**lines 130-137**).

I finally randomly shuffled the data set and put 20% of the data into a validation set (**lines 95, 159**). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The acceptable number of epochs was 5 as evidenced by video recordings from all three tracks mentioned. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Opportunities for improevment

* Simplifying the model to reduce overfitting
* Introducing batch normalization to make training loss decreese faster
* Automatic adjustment of a ```correction``` parameter according to speed and road geometry
* Implementing additional ways to augment the data
