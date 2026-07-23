# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:22:14 2026

@author: USER
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#loading the saved models
diabetes_model = pickle.load(open(os.path.join(BASE_DIR, 'Disease sav', 'Diabetes.sav'), 'rb'))
heart_disease_model = pickle.load(open(os.path.join(BASE_DIR, 'Disease sav', 'Heart Disease.sav'), 'rb'))
parkinson_model = pickle.load(open(os.path.join(BASE_DIR, 'Disease sav', 'Parkinson Disease.sav'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
heart_scaler = pickle.load(open(os.path.join(BASE_DIR, 'heart_scaler.pkl'), 'rb'))
parkinson_scaler = pickle.load(open(os.path.join(BASE_DIR, 'parkinson_scaler.pkl'), 'rb'))

#sidebar for navigate
with st.sidebar: 
    selected = option_menu('Multiple Disease Prediction System using ML',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons = ['activity', 'heart', 'person'],
                           default_index = 0)
    
#Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level') 
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Function Value')
    with col2:
        Age = st.text_input('Age of the Person')
        
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            # 1. Convert inputs to floats and put them into a DataFrame
            # The column names MUST match your training data exactly
            input_df = pd.DataFrame([{
                'Pregnancies': float(Pregnancies), 
                'Glucose': float(Glucose), 
                'BloodPressure': float(BloodPressure), 
                'SkinThickness': float(SkinThickness), 
                'Insulin': float(Insulin), 
                'BMI': float(BMI), 
                'DiabetesPedigreeFunction': float(DiabetesPedigreeFunction), 
                'Age': float(Age)
            }])
            
            # 2. Scale the input using the SAME scaler used during training
            input_scaled = scaler.transform(input_df)
            
            # 3. Predict using the scaled data
            diab_prediction = diabetes_model.predict(input_scaled)
            
            if diab_prediction[0] == 1:
                st.error('The Person is Diabetic')
            else:
                st.success('The Person is not Diabetic')
                st.balloons()
                
        except ValueError:
            st.warning('Please fill in all fields with valid numbers before testing.')
       
    
if (selected == 'Heart Disease Prediction'):
    #page title
    st.title('Heart Disease Prediction using ML')
    
    #Getting input data from user
    col1, col2, col3 = st.columns(3)
    
    with col1:
         age = st.text_input('Age')
    with col2:
         sex = st.text_input('Sex') 
    with col3:
         cp = st.text_input('Chest Pain Types')
    with col1:
         trestbps = st.text_input('Resting Blood Pressure')
    with col2:
         chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
         restecg = st.text_input('Resting Electrocardiographic Results')
    with col2:
         thalach = st.text_input('maximum heart rate achieved')
    with col3:
         exang = st.text_input('exercise induced angina')
    with col1:
         oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    with col2:
         slope = st.text_input('The slope of the peak exercise ST segment')
    with col3:
         ca = st.text_input('number of major vessels (0-3) colored by flourosopy')
    with col1:
         thal = st.text_input('0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        try:
            # 1. Build a DataFrame — column names must match training data exactly
            input_df = pd.DataFrame([{
                'age': float(age),
                'sex': float(sex),
                'cp': float(cp),
                'trestbps': float(trestbps),
                'chol': float(chol),
                'fbs': float(fbs),
                'restecg': float(restecg),
                'thalach': float(thalach),
                'exang': float(exang),
                'oldpeak': float(oldpeak),
                'slope': float(slope),
                'ca': float(ca),
                'thal': float(thal)
            }])
            
            # 2. Scale using the scaler fitted on the HEART DISEASE training data
            input_scaled = heart_scaler.transform(input_df)
            
            # 3. Predict using the scaled data
            heart_disease_prediction = heart_disease_model.predict(input_scaled)
            
            if heart_disease_prediction[0] == 1:
                heart_diagnosis = 'The Person has Heart Disease'
            else:
                heart_diagnosis = 'The Person does not have Heart Disease'
                st.balloons()
            
            st.success(heart_diagnosis)
            
        except ValueError:
            st.warning('Please fill in all fields with valid numbers before testing.')
    
if (selected == 'Parkinsons Prediction'):
    #page title
    st.title('Parkinsons Prediction using ML')
    #Getting input data from user
    col1, col2, col3 = st.columns(3)
    
    with col1:
         MDVPFo = st.text_input('Average vocal fundamental frequency')
    with col2:
         MDVPFhi = st.text_input('Maximum vocal fundamental frequency') 
    with col3:
         MDVPFlo = st.text_input('Minimum vocal fundamental frequency')
    with col1:
         MDVPJitter = st.text_input('MDVP:Jitter(%)')
    with col2:
         MDVPJitterAbs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
         MDVPRAP = st.text_input('MDVP:RAP')
    with col1:
         MDVPPPQ = st.text_input('MDVP:PPQ')
    with col2:
         JitterDDP = st.text_input('Jitter:DDP')
    with col3:
           MDVPShimmer = st.text_input('MDVP:Shimmer')
    with col1:
           MDVPShimmerDB = st.text_input('MDVP:Shimmer(dB)')
    with col2:
          ShimmerAPQ3 = st.text_input('Shimmer:APQ3')
    with col3:
           ShimmerAPQ5 = st.text_input('Shimmer:APQ5')
    with col1:
           MDVPAPQ = st.text_input('MDVP:APQ')
    with col2:
         ShimmerDDA = st.text_input('Shimmer:DDA')
    with col3:
         NHR = st.text_input('Ratio of Noise - NHR')
    with col1:
         HNR = st.text_input('Ratio of Noise -HNR')
    with col2:
           RPDE = st.text_input('RPDE')
    with col3:
           DFA = st.text_input('Signal fractal scaling exponent')
    with col1:
          spread1 = st.text_input('spread1')
    with col2:
           spread2 = st.text_input('spread2')
    with col3:
           D2 = st.text_input('D2')    
    with col1:
         PPE = st.text_input('PPE')    
    
    parkinson_diagnosis = ''
    
    if st.button('Parkinson Test Result'):
        try:
            # 1. Build a DataFrame — column names must match training data exactly
            input_df = pd.DataFrame([{
                'MDVP:Fo(Hz)': float(MDVPFo),
                'MDVP:Fhi(Hz)': float(MDVPFhi),
                'MDVP:Flo(Hz)': float(MDVPFlo),
                'MDVP:Jitter(%)': float(MDVPJitter),
                'MDVP:Jitter(Abs)': float(MDVPJitterAbs),
                'MDVP:RAP': float(MDVPRAP),
                'MDVP:PPQ': float(MDVPPPQ),
                'Jitter:DDP': float(JitterDDP),
                'MDVP:Shimmer': float(MDVPShimmer),
                'MDVP:Shimmer(dB)': float(MDVPShimmerDB),
                'Shimmer:APQ3': float(ShimmerAPQ3),
                'Shimmer:APQ5': float(ShimmerAPQ5),
                'MDVP:APQ': float(MDVPAPQ),
                'Shimmer:DDA': float(ShimmerDDA),
                'NHR': float(NHR),
                'HNR': float(HNR),
                'RPDE': float(RPDE),
                'DFA': float(DFA),
                'spread1': float(spread1),
                'spread2': float(spread2),
                'D2': float(D2),
                'PPE': float(PPE)
            }])
            
            # 2. Scale using the scaler fitted on the PARKINSON'S training data
            input_scaled = parkinson_scaler.transform(input_df)
            
            # 3. Predict using the scaled data
            parkinson_prediction = parkinson_model.predict(input_scaled)
            
            if parkinson_prediction[0] == 1:
                parkinson_diagnosis = 'The Person has Parkinson Disease'
            else:
                parkinson_diagnosis = 'The Person does not have Parkinson Disease'
                st.balloons()
            
            st.success(parkinson_diagnosis)
            
        except ValueError:
            st.warning('Please fill in all fields with valid numbers before testing.')
    



















