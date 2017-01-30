# Test model 
#      left turn .3 
#      right turn .3
#      epochs 10
#      augment with flip 
#      Passed
#      Track 1 throttle .2
#      Track 2 throttle .5 when speed < 5mph
# coding: utf-8

# In[ ]:

import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from   preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam


# In[ ]:

#img_rows = 16
#img_cols = 32
#lines=[]
#X_train=[]
#y_train=[]
#batch_size=200
#nb_epoch=5
#csvfile = 'data/driving_log.csv'
data_folder = 'data/'


# In[ ]:

def read_file(csvfile,lines):

    with open(csvfile,'rt') as f:
         reader = csv.reader(f)
         for line in reader:
             lines.append(line)
    remove_labels = lines.pop(0)

    return lines



# In[ ]:


# In[ ]:

def random_shear(image,steering,shear_range=100):
    rows,cols = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    
    return image,steering


# In[ ]:

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
       image,steering=cv2.flip(image,1),-steering
    return image,steering


# In[ ]:

def augment_brightness_camera_images(image):
       image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
       random_bright = .25+np.random.uniform()
    #print(random_bright)
       image1[:,:,2] = image1[:,:,2]*random_bright
       image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
       return image1


# In[ ]:

def augment_images(image, steering):
    #image = augment_brightness_camera_images(image)
    image,steering = random_flip(image,steering)
    return image,steering


# In[ ]:

def read_next_image(i,X_train,y_train):
    img = plt.imread(data_folder + (X_train[i].strip()))
    steering = float(y_train[i])
    return img, steering


# In[ ]:

def load_data(X,y,lines,data_folder,left_offset,right_offset):

    for i in range(len(lines)):
        X.append(lines[i][0])
        y.append(float(lines[i][3]))

    for i in range(len(lines)):
        X.append(lines[i][1])
        y.append(float(lines[i][3]) + left_offset)
    
    for i in range(len(lines)):
        X.append(lines[i][2])
        y.append(float(lines[i][3]) - right_offset)


    return X,y
 
    

# In[ ]:

def generate_training_data(X_train, y_train,img_cols,img_rows):
        i               = np.random.randint(0,len(y_train))
        image, steering = read_next_image(i,X_train,y_train)
        
        image  = image_preprocessing(image,img_cols,img_rows)
        
        #image,steering = augment_images(image,steering)
        
        
        return image, steering

def generate_batch(X_train,y_train,img_cols,img_rows,batch_size = 32):
    
    batch_images = np.zeros((batch_size, img_rows, img_cols, 1))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_data(X_train, y_train,img_cols,img_rows)
            batch_images[i_batch] = x.reshape((img_rows,img_cols, 1))
            batch_steering[i_batch] = y
            
            
        yield batch_images, batch_steering

def create_model(img_rows,img_cols):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(16, 3, 3,border_mode='valid',input_shape=(16,32,1),activation='relu'))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1,activation='tanh'))
    model.summary()
    return model
