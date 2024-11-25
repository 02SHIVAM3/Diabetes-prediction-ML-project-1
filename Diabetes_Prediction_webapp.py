# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:03:04 2024

@author: Lenovo
"""


import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:\Users\shiva\OneDrive\Desktop\MLmodelAPI\diabetes_model.sav','rb'))
#creating a function for prediction

def diabetes_prediction(input_data):
    
    # Convert input data to float and handle empty inputs
    input_data = [float(x) if x else 0 for x in input_data]
    

    #change the input data to a nump array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return "The person is not diabetic"

    else:
        return "The person is diabetic"
    
    
    
    
def main():
    
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
   
    
    preg = st.text_input('Pregnancies')
    glu = st.text_input('Glucose')
    bp = st.text_input('Blood Pressure')
    skin = st.text_input('Skin Thickness')
    ins = st.text_input('Insulin')
    dpf = st.text_input('Diabetes prediction function')
    age = st.text_input('Age')
    
    
    
    
    #code for prediction
    diagnosis = ""
    #creating a button for prediction
    if st.button("Heart Defect Test Result"):
        diagnosis = diabetes_prediction([preg, glu, bp, skin, ins, dpf, age])
        
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
    
    
    