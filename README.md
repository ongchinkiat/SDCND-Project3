## Self-Driving Car Engineer Nanodegree

## Project 3: Behavioral Cloning

## Introduction

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Project Files

The project files are:

1. Script to create and train the model: model.py

2. Script to drive the car in autonomous mode: drive.py

3. Project writeup: README.md

4. Trained model file: model.h5

5. Simulation run video: video.mp4

### Functional Code

Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing

```
python3 drive.py model.h5
```

### Model script code usable and readable

The model.py script can be executed by running the command

```
python3 model.py
```

The model.py file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### Model Architecture

The model I used is based on the NVIDIA Architecture. It consist of 5 convolution layer followed by 5 fully connected layers. (model.py lines 210 - 234)

RELU layers were added between the fully connected layer to introduce nonlinearity. (model.py lines 227, 230)

Before the convolution layers, the image went through 3 pre-processing steps:

1. Cropping to remove the "unwanted" top and bottom parts of the image, which doesn't help in our training goals.

2. Normalize the image data to between -1 and 1.

3. Reduce the image size by half the width and half the height using Max Pooling.



### Reduce overfitting

Dropout layer was used between the first and second fully connected layer to reduce overfitting of the model. (model.py lines 228)

The data is also split into Training and Validation sets. (model.py lines 167)

### Model Summary

This is the summary of the model:

| Layer (type)  | Output Shape  | Param #   |
| ----- | ----- | ----- |
| cropping2d_1 (Cropping2D) |   (None, 90, 320, 3)    |    0         |
| lambda_1 (Lambda) (Normalize)  |        (None, 90, 320, 3) |       0        |
| max_pooling2d_1 (MaxPooling2D) | (None, 45, 160, 3)  |      0         |
| conv2d_1 (Conv2D)   |         (None, 21, 78, 24)  |      1824      |
| conv2d_2 (Conv2D)  |          (None, 9, 37, 36)  |       21636     |
| conv2d_3 (Conv2D)  |          (None, 5, 33, 48) |        43248     |
| conv2d_4 (Conv2D)        |    (None, 3, 31, 64)  |       27712     |
| conv2d_5 (Conv2D) |           (None, 1, 29, 64) |        36928    |
| flatten_1 (Flatten)  |        (None, 1856) |             0    |     
| dense_1 (Dense)  |            (None, 1000)         |     1857000   |
| activation_1 (Activation) (RELU) |   (None, 1000)  |            0         |
| dropout_1 (Dropout) |         (None, 1000)       |       0   |      
| dense_2 (Dense)   |           (None, 200)  |             200200    |
| activation_2 (Activation) (RELU) |  (None, 200)  |             0         |
| dense_3 (Dense) |             (None, 50)              |  10050     |
| dense_4 (Dense) |             (None, 10)  |              510       |
| dense_5 (Dense)  |            (None, 1)              |   11     |

### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. (model.py line 237)

The model is trained using these hyperparamters:

1. epochs = 5
2. batch size = 32
3. keep probabilities = 0.5
4. optimizer = AdamOptimizer

### Appropriate training data

I started with just the Udacity provided training data set to build up my model architecture.

With a simple architecture of 2 fully connected layers, the model managed to keep the car in the center but fail to make even the first turn.

More layers were added, and I finally settled down on the 5 convolution layer followed by 5 fully connected layers architecture described above.

To improve performance, I added the left camera images, with the steering angle added a fix bias of +0.35 (model.py line 41-47). The right camera images were also added, with the steering angle added a fix bias of -0.35 (model.py line 49-55). For the middle camera, images with center angle < -0.15 or center angle > -0.15 were duplicated 2 more time, to add emphasis on these large steering samples. (model.py line 59-65)

With these improvements, the model managed to drive the car pass the bridge, but fail to pass the sharp bend after the bridge.

More training data were recorded using the Unity simulator in training mode, especially in those areas where the car goes off the road during testing.

Two more training data set were used:

1. The set "turnafterbridge" consist of 958 images recording the car going through the sharp turn after the bridge. (model.py line 70-115)

These are some images from the "turnafterbridge" data set:

![Image 1](https://github.com/ongchinkiat/SDCND-Project3/raw/master/turnafterbridge-center-1.jpg "Image 1")

![Image 2](https://github.com/ongchinkiat/SDCND-Project3/raw/master/turnafterbridge-center-2.jpg "Image 2")

![Image 3](https://github.com/ongchinkiat/SDCND-Project3/raw/master/turnafterbridge-center-3.jpg "Image 3")


2. The set "twoturns" consist of 2368 images recording the car going through the 2 turns after the sharp turn after the bridge. (model.py line 118-163)

These are some images from the "twoturns" data set:

![Image 1](https://github.com/ongchinkiat/SDCND-Project3/raw/master/twoturns-center-1.jpg "Image 1")

![Image 2](https://github.com/ongchinkiat/SDCND-Project3/raw/master/twoturns-center-2.jpg "Image 2")

![Image 3](https://github.com/ongchinkiat/SDCND-Project3/raw/master/twoturns-center-3.jpg "Image 3")

The car went onto the ledge on the 2nd turn after the bridge. This was fixed by increasing the bias for the left and right camera from 0.35 to 0.45.

With the changes, the model managed to drive the car around the track without leaving the road.

This is the training and validation accuracy and loss graph:

![Loss Image](https://github.com/ongchinkiat/SDCND-Project3/raw/master/model-loss.jpg "Loss Image")


### Simulation

A video of the simulation run is in the file video.mp4.

A simulator screen capture video is also uploaded on YouTube. This video is from a previous run using a older model. In this run, the car went onto the ledge on the 2nd turn after the bridge.

Video URL: https://youtu.be/w9Tr5YNHzx4

<a href="http://www.youtube.com/watch?feature=player_embedded&v=w9Tr5YNHzx4" target="_blank"><img src="http://img.youtube.com/vi/w9Tr5YNHzx4/0.jpg"
alt="CarND" width="240" height="180" border="1" /></a>
