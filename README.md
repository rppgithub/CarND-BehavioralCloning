# CarND-BehavioralCloning
Project 3 of Udacity Nanodegree
##Instructions
The solutions contains the following files:
* model.py - Contains the code to generate the model and train the model 
* utils.py - Contains utility functions
* preprocess.py - Function for pre-processing the image
* drive.py      - Modified drive.py
  * For Track 1 -- use throttle of .2
  * For Track 2 -- Uncomment line 58 and 59. When speed goes below 5mph, increases throttle to .5
 * model.json - json file
 * model.h5   - weights file
 1. Run python model.py -- This creates model.json and model.h5
 2. For Track 1 run python drive.py model.json
 3. For Track 2 Uncomment Line 58 and 59. Run python drive.py model.json
 4. Data is stored in the data directory.
 
 
##Data Set
 * I am using the set of data provided by Udacity with the following changes:
   * Kept only the S channel in the HSV color space and reduced the size to 16 rows and 32 columns. This idea was     borrowed from https://github.com/xslittlegrass/CarND-Behavioral-Cloning
   * Added a small offset ( +/- .3 ) to left and right images respectively. I started with .4 for the left offset and .8 for the right offset and scaled down after observing behavior on the tracks
   * I randomly flip images to augment the data set
   * Ideas that I discarded after running it on the tracks:
     - Cropping the image to discard the bonnet of the car which is visible in the image
     - Shearing the image
     - Increasing brightness
     
## Architecture
  * This model uses 497 parameters
  * Normalization 
  * Convolution with a 3 x 3 kernel and relu activation
  * Max Pooling with a 4 x 4 kernel
  * Applied Dropout of .25 to prevent overfitting
  * Dense layer with 1 neuron with tanh activation
  
  
  | Layer           | Output Shape                  | Param
  |-------          |--------------                 |------
  |Normalization     | (None,16,32,1)                |  0
  |Conv2D           | (None,14,30,16)               | 160
  |MaxPooling2D     | (None, 3,7,16)                | 0
  |Dropout (.25)    | (None, 3,7,16)                | 0
  |Flatten          | (None, 336)                   | 0
  |Dense            | (None, 1)                     | 337

## Training Strategy
   * Shuffled the data and perfom an 80/20 split of the train/test data. Don't use validation data since validation is performed on the track
   * Used Adam optimizer with default learning rate and mean square error
   * Used fit generator with 10 epochs.
## Approach
  * Used ideas from Nvidia paper and comma.ai to augment images. Also from Dr. Vivek Yadav learnt how to augment images. 
  * Although the Nvidia model worked without any changes , I found that it took a while to train it on my laptop. My mentor suggested reading Mohan Karthik's paper @mohankarthik
  * Another student Mengxi Wu wrote about the tiny model with just a few parameters. I borrowed the idea of using just the S channel in the HSV color space. This works for both the tracks. I experimented by adding a slight offset to the left and right steering through a set of trial and error. Also, flipped images to augment.
  * Driving on track 2, noticed that the car had difficulty going up-hill. I modified drive.py to set throttle to .5 whenever speed decreased to less than 5 mph. With this change, the car is now able to finish track 2. Track 1 worked with throttle set to .2.
  * Many thanks to all of the people on the forums and the keras documentation.
     
