import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.linear_model import LinearRegression
from pickle import dump
from pickle import load
import pickle
from sklearn.preprocessing import StandardScaler




data=pd.read_csv("insurance.csv")
array = data.values
X = array[:, 0:-1]

loaded_model = load(open('model.pkl', 'rb'))



def INSURANCE_prediction(input_data):
    

    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return round(prediction[0], 2)
    



def main():
    
    
    # giving a title
    st.title('Model Deployment: MULTIPLE LINEAR REGRESSION Model')
    
    
    # getting the input data from the user
    
    
    AGE = st.number_input('Insert AGE', min_value=0, max_value=85, value=0)
    SEX = st.number_input('Insert SEX', min_value=0, max_value=1)
    BMI = st.number_input('Insert BMI', min_value=0.0, max_value=100.0, value=0.0, format="%f")
    CHILDERN = st.number_input('Insert CHILDERN', min_value=0, max_value=5, value=0, step=1)
    SMOKER = st.number_input('Insert SMOKER', min_value=0, max_value=1)
    REGION = st.number_input('Insert REGION ', min_value=0, max_value=4,value=0, step=1)
    Predicted_Insurance_Premium = ''
    
    # creating a button for Prediction
    
    if st.button('Predicted Insurance Premium'):
        Predicted_Insurance_Premium = INSURANCE_prediction([AGE, SEX, BMI, CHILDERN, SMOKER, REGION])
        
        
    st.success(Predicted_Insurance_Premium)
    
    
    
    
    
if __name__ == '__main__':
    main()