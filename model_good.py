#Test 1 Passed
#create_model
#Test2 testing
#create model2

import numpy as np
import keras
import matplotlib.pyplot as plt
import csv
import cv2
from   utils import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

img_rows = 16
img_cols = 32
lines=[]
X_train=[]
y_train=[]
batch_size=200
nb_epoch=10
csvfile = 'data/driving_log.csv'
data_folder = 'data/'
left_offset  = .3
right_offset = .8


# read file
lines=read_file(csvfile,lines)

print (len(lines))
# load data
X_train,y_train = load_data(X_train,y_train,lines,data_folder,left_offset,right_offset)


print (len(X_train))
print (len(y_train))

#shuffle data
X_train, y_train = shuffle(X_train, y_train)
# split train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)

#generate training data in batches 
data_generator = generate_batch(X_train,y_train,img_cols,img_rows,batch_size)


# In[ ]:


# In[ ]:
#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#create model
model = create_model7(img_rows,img_cols)
#compile model
model.compile('adam','mse')
#generate
history = model.fit_generator(data_generator,
                    samples_per_epoch=20000, nb_epoch=nb_epoch,
                    verbose=1)


#save
model_json = model.to_json()
with open("model.json", "w") as json_file:
     json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
