# Test1 model1 left turn .45 right turn .4 passed
# Test2 model2 right turn .4 failed
# Test3 model2 right turn .6 failed
# Test4 model3 right turn .8 failed
# Test5 model4 left turn .45 right turn .8 failed
# Test6 model5 left turn .45 right turn .8 failed
# Test7 model6 left turn .3 right turn .8 failed
# Test8 model7 left turn .3 right turn .8 passed with 10 epochs
# Test9 model7 
#      left turn .3 
#      right turn .8
#      epochs 10
#      augment with flip 
#      Passed
#      Submit this
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
    discard_labels = lines.pop(0)

    return lines



# In[ ]:


# In[ ]:

def image_trim(img):
    trimed = img#[20:140]
#     resized = cv2.resize(img,(32,16))
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(32,16))
    return resized

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

def read_next_image(m,X_train,y_train):
    img = plt.imread(data_folder + (X_train[m].strip()))
    steering = float(y_train[m])
    return img, steering


# In[ ]:

def load_data(X,y,lines,data_folder,left_offset,right_offset):

    for i in range(len(lines)):
        #img = plt.imread(data_folder + (lines[i][0]).strip())
        #X.append(image_preprocessing(img))
        #X.append(img)
        X.append(lines[i][0])
        y.append(float(lines[i][3]))

    for i in range(len(lines)):
        #img = plt.imread(data_folder + (lines[i][1]).strip())
        #X.append(image_preprocessing(img))
        #X.append(img)
        X.append(lines[i][1])
        y.append(float(lines[i][3]) + left_offset)
    
    for i in range(len(lines)):
        #img = plt.imread(data_folder + (lines[i][2]).strip())
        #X.append(image_preprocessing(img))
        #X.append(img)
        X.append(lines[i][2])
        y.append(float(lines[i][3]) - right_offset)


    return X,y
 
    

# In[ ]:

def generate_training_data(X_train, y_train,img_cols,img_rows):
        m               = np.random.randint(0,len(y_train))
        image, steering = read_next_image(m,X_train,y_train)
        
        image  = image_preprocessing(image,img_cols,img_rows)
        
        image,steering = augment_images(image,steering)
        
        
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

def create_model3(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
    model.add(MaxPooling2D((2,2),(2,2),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(8,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(4,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='elu'))
    model.summary()

    return model


def create_model4(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
    model.add(MaxPooling2D((2,2),(2,2),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='relu'))
    model.summary()

    return model

def create_model5(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
    model.add(MaxPooling2D((2,2),(2,2),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16,activation='elu'))
    model.add(Dense(8,activation='elu'))
    model.add(Dense(4,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='elu'))
    model.summary()

    return model

def create_model6(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16,activation='elu'))
    model.add(Dense(8,activation='elu'))
    model.add(Dense(4,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='elu'))
    model.summary()

    return model

def create_model7(img_rows,img_cols):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(16, 3, 3,border_mode='valid',input_shape=(16,32,1),activation='relu'))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1,activation='tanh'))
    model.summary()
    return model


def create_model2(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(16,32,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(16,32,1), activation='relu'))
    model.add(MaxPooling2D((2,2),(2,2),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16,activation='elu'))
    model.add(Dense(8,activation='elu'))
    model.add(Dense(4,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='elu'))
    model.summary()

    return model

def create_model(img_rows,img_cols):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(img_rows,img_cols,1)))
    model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(img_rows,img_cols,1), activation='relu'))
    model.add(MaxPooling2D((4,4),(4,4),'valid'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()
    return model
