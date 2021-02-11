#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt


# In[2]:


#Input: Two real values (a sample)
#Output: Classification (True or False, 0.0 - 1.0) 

def define_discriminator(n_inputs= 2):
    model = keras.models.Sequential()  #Groups a linear stack of layers to the model https://www.tensorflow.org/api_docs/python/tf/keras/Model
    
    model.add(keras.layers.Dense(units= 50, #Regular densely-connected NN layer https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
                                 activation= 'relu', #Activation function - It takes in the output signal and converts it into some form that can be taken as the next input
                                 input_dim= n_inputs,
                                 kernel_initializer= 'he_uniform')) #Recomended for RELU
    model.add(keras.layers.Dense(units= 1,  #Regular densely-connected NN layer 
                                 activation= 'sigmoid')) #Activation function
    model.compile(loss= 'binary_crossentropy',  #Configure model for trainning  - returns a weighted loss float tensor
                  optimizer= 'adam', #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                  metrics= ['accuracy']) #Calculates how often predictions match binary labels - returns prediction accurary
    
    return model


# In[3]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[5]:


import tensorflow as tf


# In[6]:


from tensorflow.python.keras.layers import Input, Dense


# In[7]:


from tensorflow.keras.layers import Input, Dense


# In[ ]:




