# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:46:05 2024

@author: shiva
"""

import numpy as np
import streamlit as sl
import pickle

loaded_model = pickle.load(open('C:/Users/shiva/OneDrive/Desktop/webModel/trained_model.sav','rb'))


input_data = (1,85,66,29,0,26.6,0.351,31)

# changing the input_data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# we need to standardize the data same as we did it while training the data

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]):
  print("THE PERSON IS NOT DIABETIC")
else:
  print("THE PERSON IS DIABETIC")