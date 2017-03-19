**Behavioral Cloning**

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./output_images/data_distribution_lane_driving_track01.png "Data Distribution"
[image2]: ./output_images/data_distribution_recovery_track01.png
[image3]: ./output_images/lane_driving_track01_turning_left.png
[image4]: ./output_images/recovery_driving_track01_hard_left.png
[image5]: ./output_images/recovery_driving_track01_hard_right.png
[image6]: ./output_images/data_augmentation_all_cameras.png
[image7]: ./output_images/data_augmentation_brightness.png
[image8]: ./output_images/data_augmentation_flip.png
[image9]: ./output_images/data_augmentation_transform.png
[image10]: ./output_images/data_augmentation_stripe.png
[image11]: ./output_images/loss_of_train_dataset.png
[image12]: ./output_images/loss_of_test_dataset.png

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:

* `drive.py` for driving the car in autonomous mode
* `explorer.py` data exploration snippet
* `explorer.ipynb` data exploration jupiter notebook
* `model.h5` containing a trained convolution neural network model
* `model.py` containing the script to create and train the model
* `output_images/` folder containing output images
* `parameters.py` containing the global parameters
* `video.py` for capturing video
* `video.mp4` autonomous driving on track one
* `video_track2.mp4` autonomous driving on track two
* `writeup_report.md` summarizing the results
* `tools/data_explorer.py` tools for data exploration
* `tools/data_generator.py` tools for data generation
* `tools/data_preprocessor.py` tools for data pre-processing
* `tools/data_provider.py` tools for providing data
* `tools/network.py` neural network definition

#### 2. Submssion includes functional code

Using the Udacity provided simulator and `drive.py` file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submssion code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline that was used for training and validating the model. It uses the following python classes:

* `tools/data_provider.py` which provides the training and test data from data sets
* `tools/data_generator.py` which generates additional data by adjusting image brightness, flips the images, creates image transformation and applies strips of different shadows to the the image
* `tools/data_preprocessor.py` which re-sizes and crops the images
* `tools/network.py` which contains convolutional neural network definition

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Regression model was used. It is made of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (`tools/network.py` lines 12-37). The model also includes fully connected layers with RELU activations to introduce non-linearity ('tools/network.py' lines 39-59).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers (30%) in every layer in order to reduce overfitting ('tools/network.py' lines 12 and 59). L2 weights regularization (0.001) was also used in largest fully connected layer ('tools/network.py' line 41) .

Training, validation and test data sets were used while training the model to ensure that the model was not overfitting.(`model.py` lines 21-22, lines 27-34).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 20).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The combination of center lane driving and recovering from the left and right sides of the road on both tracks was used.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to compare the project to traffic sign classification project where convolutional neural network was used. Similar kind of network was used for this project, but output of the model was modified in order to make it a regression model. I was also inspired by [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

Loss of training, validation and test data set was monitored during the training. The parameters were modified to ensure that loss of both training and validation dataset was dropping. In order to prevent overfitting:

* model complexity was reduced (i.e. NVidia's largest fully connected layer was reduced from 1164 to 300)
* dropout layers were introduced (0.3) in all layers of the model
* L2 wight regularization was used (0.001) in the largest fully connected layer
* short training, only 5-9 EPOCS were used

The final test was to run the simulator to see how well the car was driving around both tracks. There were a few spots where the vehicle fell off the track (i.e. bridge, off-road section, tight curves on advanced track). To improve the driving behavior in these cases, aditional lane and recovery driving data was recorded in those sections of the road and used to train the model.

At the end of the process the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`network.py`, `create_convolutional_nvidia_style_modified()`) consisted of a convolution neural network with the following layers and layer sizes:

* input: 160 x 320 x 3
* lambda: output 160 x 320 x 3;
* cropping: output 65 x 320 x 3
* layer 1: convolutional; kernel 5 x 6; stride 2 x 4; output 33 x 80 x 24; dropout 30%;
* layer 2: convolutional; kernel 5 x 5; stride 2 x 2; output 17 x 40 x 36; dropout 30%;
* layer 3: convolutional; kernel 5 x 5; stride 2 x 2; output  9 x 20 x 48; dropout 30%;
* layer 4: convolutional; kernel 3 x 3; stride 1 x 1; output  9 x 20 x 64; dropout 30%;
* layer 5: convolutional; kernel 3 x 3; stride 1 x 1; output  9 x 20 x 64; dropout 30%;
* layer 6: fully connected; output 300 x 1; dropout 30%;
* layer 7: fully connected; output  75 x 1; dropout 30%;
* layer 8: fully connected: output  40 x 1; dropout 30%;
* layer 9: fully connected: output  10 x 1; dropout 30%;
* output: fully connected: output    1 x 1;

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, three laps of center lane driving were recorded on each track. Two laps of recovery driving on both tracks were added.

Training data was divided into 49 sections:

* [-24] to [-1] for left steering angle
* [0] for straight driving
* [1] to [24] for right steering angles

In order to ensure uniform distribution filters were applied for lane driving `parameters.py` (lines: 37 to 56)

* [-1] random 60%,
* [0] random 10%,
* [1] random 60%

And the following filters for recovery driving:

* from section [-5] to [5] random 10% of data was used.

Figure below shows example of manipulated data distribution for lane driving on track one.

![alt text][image1]

Random examples of turning left during center lane driving on track one are shown on the figure below:

![alt text][image3]

Then recovery driving was recorded on both tracks where vehicle recovers from the left and right side of the road back to the center. Example of manipulated data distribution for recovery driving is shown below.

![alt text][image2]

Random examples of recovery from the right back to the center...

![alt text][image4]

And recovery examples from the left back to center...

![alt text][image5]

In order to generate more data points the following data augmentation techniques were used (`tools/data_generation.py`):

* center and both left and right camera images were used and steering angle updated
* random brightness of the image
* random flip of the image and steering angles updated to make dataset symmetrical
* random horizontal translation of the image and steering angles updated
* random vertical translation of the image to improve driving in hilly conditions
* random stripes of shadows to improve behaviour where shadows are casted

Examples of steering wheel adjustment when using left and right camera:

![alt text][image6]

Example of random brightness:

![alt text][image7]

Examples of random flip of the image and steering wheel angle adjustment:

![alt text][image8]

Example of random vertical and horizontal transformation of the image and steering wheel angle adjustment:

![alt text][image9]

And example of random stripes:

![alt text][image10]

Data generation process created 5 times more data points.

The data was randomly shuffled and split into:

* dataset used for training 85% of which 80% was used for training and 20% for validation and
* dataset used for testing 15%. Test data set didn't include generated images.

Each batch contained 2.000 random data points from which data generator created additional 10.000 data points.

During the training loss on training and validation dataset was monitored in order to insure that it was decreasing for both sets of data. EPOCS parameter was accordingly adjusted in order to prevent overfitting. Figure below shows loss of training and validation dataset:

![alt text][image11]

And loss of test dataset:

![alt text][image12]

Models were saved and evaluated with driving simulator on every 3 batches.

Example of autonomous driving on [track one](./video.mp4) ond [track two](./video_track2.mp4) was recorded into a video.

### Discussion

The best project by far on CarND so far. Spent over $200 on AWS. It might be time to get that NVidia Titan GPU ;-)

Two different approaches were used for the two models that I trained for this project. For the first model classification model was used and for the second one regression. Both of them were able to drive the vehicle around both tracks however the latest regression model performs better. It achieves faster speeds as its more fine tuned and vehicle doesn't oscillate as much.

