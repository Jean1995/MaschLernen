
# coding: utf-8

# In[1]:


import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

from skimage import color
from skimage import io

from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image


# In[2]:


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


# In[4]:


def import_image_and_crop(datapath,crop_size=1000):
    ''' Read the image, convert it to grayscale, and use only the (crop_size x crop_size) center
    If the crop_size is larger than the image_size, use padding'''
    image = color.rgb2gray(mpimg.imread(datapath))
    if(np.shape(image)[0]<crop_size or np.shape(image)[1]<crop_size):
        # If the crop size is too large for the image, copy the border to enlarge the image again
        dif_height = int(np.ceil( (crop_size - np.shape(image)[0]) /2))
        dif_width = int(np.ceil( (crop_size - np.shape(image)[1]) /2))
        if(dif_height<0):
            dif_height = 0
        if(dif_width<0):
            dif_width = 0
        image = cv2.copyMakeBorder(image,dif_height,dif_height,dif_width,dif_width,cv2.BORDER_REPLICATE)
    image = crop_center(image, crop_size, crop_size)
    return image



# In[6]:


# import data

from os import listdir
from os.path import isfile, join
import random

crop_size = 250

# Here, you can choose the folder data_prepared for all data, data_prepared_X for the language X only

onlyfiles = [f for f in listdir('data_prepared_english/') if isfile(join('data_prepared_english/', f))] # get list of datanames in folder
#onlyfiles = [f for f in listdir('data_prepared_arabic/') if isfile(join('data_prepared_arabic/', f))] # get list of datanames in folder
#onlyfiles = [f for f in listdir('data_prepared/') if isfile(join('data_prepared/', f))] # get list of datanames in folder

number_data = len(onlyfiles) - 1 # How many datafiles? One less because there is a list for the rotations
datanames = np.linspace(0,number_data-1,number_data, dtype=np.int32)

X_train_pre = np.empty([number_data,crop_size,crop_size]) # create array to be filled, remember crop_size

for counter in datanames:
    print(counter, end="\r")
    datapath = 'data_prepared_english/' + str(int(counter))+'.png' # i know this is ugly
    #datapath = 'data_prepared_arabic/' + str(int(number_of_file))+'.png' # i know this is ugly
    #datapath = 'data_prepared/' + str(int(number_of_file))+'.png' # i know this is ugly
    X_train_pre[counter,:,:] = import_image_and_crop(datapath, crop_size)
    
    


# In[7]:


plt.imshow(X_train_pre[5,:,:], cmap='gray') # show image in grayscale, now squared and in the middle


# In[8]:


# load angle list
Y_train_pre2 = np.loadtxt('data_prepared_english/angle_list.txt', delimiter=',', unpack=True)
#Y_train_pre2 = np.loadtxt('data_prepared_arabic/angle_list.txt', delimiter=',', unpack=True)
#Y_train_pre2 = np.loadtxt('data_prepared/angle_list.txt', delimiter=',', unpack=True)

Y_train_pre2.reshape((number_data,1))


# In[9]:


X_train_pre = X_train_pre.astype('float32')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_pre = scaler.fit_transform(X_train_pre.reshape(X_train_pre.shape[0],crop_size*crop_size))
Y_train_pre = scaler_y.fit_transform(Y_train_pre2.reshape(-1,1))


# In[10]:


# own periodic loss function
# 0.5 <-> 180 depending on using the scaler or not
from keras import backend as K
def mse_periodic(y_true, y_pred):
    per_value = 0.5 #0.5 for scaled to (0,1), 180 for scaled to (0,360) (normal)
    #y_true = y_true % (per_value * 2)
    #y_pred = y_pred % (per_value * 2)
    diff = per_value - abs(abs(y_true - y_pred) - per_value)
    return K.mean(K.square(diff), axis=-1)


# In[47]:


# weird thingy 
shape_ord = (crop_size, crop_size, 1)

X_train_pre2 = X_train_pre


X_train_pre2 = X_train_pre2.reshape((X_train_pre2.shape[0],) + shape_ord)
X_train_pre2 = X_train_pre2.astype('float32')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.regularizers import l2



from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train_pre2, Y_train_pre, test_size=0, random_state=20)



# In[71]:


## Hyperparameter optimalisation

def make_model(dense_layer_sizes, dense_activation, filters, 
               kernel_size, pool_size, padding_type, stride_size, dropout_rate, optimizer, reg_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes. This list has one number for each layer
    dense_activation: activation funciton in dense layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    padding_type: type of padding: same or valid
    stride_size: symmetric stride size
    dropout_rate: dropout rate
    optimizer: optimizer used for mimizing
    '''

    model = Sequential()
    
    model.add(Conv2D(filters, (kernel_size, kernel_size), padding=padding_type, 
                     strides=(stride_size, stride_size), activation='relu', input_shape=shape_ord))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size, activation=dense_activation, kernel_regularizer=l2(reg_size)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    #model.add(Activation('softmax'))

    model.compile(loss=mse_periodic, optimizer=optimizer)

    return model


# In[72]:


from sklearn.model_selection import GridSearchCV

optimizer = ['Adagrad', 'adam', 'adadelta']
dense_size_candidates = [[32], [64], [32, 32], [64, 64], [32,64], [64,32],[16],[16,16],[16,32],[32,16],[16,64],[64,16]]

param_grid={'dense_layer_sizes': dense_size_candidates,
            'dense_activation' : activation,
             'filters': [8,16,32,64],
             'kernel_size': [2,3,4,5,8],
             'pool_size': [2,3,4,5],
             'padding_type' : ['valid'],
             'stride_size'  : [1],
             'dropout_rate' : [0.25,0.5,0.75],
             'optimizer' : optimizer,
             # epochs and batch_size are avail for tuning even when not
             # an argument to model building function
             'epochs': [20],
             'batch_size': [256],
             'reg_size': [0.01,0.001,0.0001]
            }


# In[73]:


from keras.callbacks import ModelCheckpoint

filepath = "models/best_cnn.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# In[74]:


from keras.wrappers.scikit_learn import KerasRegressor

my_cnn = KerasRegressor(make_model)

grid = GridSearchCV(my_cnn, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
grid_result = grid.fit(X_train, Y_train, callbacks=[checkpoint])


# In[65]:


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


