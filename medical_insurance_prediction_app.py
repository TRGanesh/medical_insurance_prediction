# IMPORTING REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie
from PIL import Image

# LOADING THE SAVED MODEL,ONE-HOT-ENCODER,SCALER
loaded_model = pickle.load(open('medical_insurance_model.sav','rb'))
loaded_encoder = pickle.load(open('One_Hot_Encoder.sav','rb'))
loaded_scaler = pickle.load(open('Scaler.sav','rb'))

# CREATING A FUNCTION THAT PREDICTS USING LOADED_MODEL
def insurance_prediction(data):
    input_data =np.hstack(([[data[0],data[2],data[3]]],loaded_encoder.transform([[data[1],data[4],data[5]]])))
    scaled_input_data = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_input_data)
    return str(prediction)

# LOADING LOTTIE FILE
def load_lottier(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
     
    # SETTING PAGE CONFIGURATION 
    st.set_page_config(page_title='Medical Insurance Prediction',layout='wide')
        
    # LOADING ASSETS
    # ANIMATION FILES BY "LOTTIE FILES"
    lottie_coding = load_lottier("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")
    
    #USING LOCAL CSS
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    local_css('style.css')        
    
    # TITLE
    st.title('Medical Insurance Prediction')
    
    with st.container():
        st.write('- - -')
        left_column,right_column = st.columns((2,1))
        with left_column:
            st.header('Input Values')
            # GETTING INPUT DATA FROM USER
            Age = st.text_input('Age',placeholder='Enter age')
            Sex = st.selectbox('Sex',['male','female'],key='sex')
            BMI = st.text_input('Body Mass Index',placeholder='Enter BMI Value')
            Children = st.text_input('Children',placeholder='Enter number of children you have')
            Smoker = st.selectbox('Smoker(Yes/No)',['yes','no'],key='smoker')
            Region = st.selectbox('Select Region',['northeast','northwest','southeast','southwest'],key='region')
        with right_column:
            st.write('- - -')
            st.write('##')
            st.write('##')
            st_lottie(lottie_coding,height=300,key='lottie_coding')
    
        st.header('Model Prediction')
            
        # INITIALIZING ANSWER
        answer = ' '
            
        if st.button('Result'):
            answer = insurance_prediction([Age,Sex,BMI,Children,Smoker,Region])
            st.write('Medical Insurance provided is')  
            answer = answer.replace('[','')
            answer = answer.replace(']','')
            st.success(f'${answer}',icon="✅")
      

# By below code main function gets called by only terminal or command prompt    
if __name__ == '__main__':
    main()     
