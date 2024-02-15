import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import warnings 
warnings.filterwarnings('ignore')

#import data 
data = pd.read_csv('USA_Housing.csv')

#import model 
model = joblib.load('housepredictor.pkl')


st.markdown("<h1 style = 'color: #CD1818; text-align: center; font-family: helvetica '>HOUSE PRICE PREDICTION </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #12372A; text-align: center; font-family: cursive '>Built By GomyCode MTW</h4>", unsafe_allow_html = True)

st.image('pngwing.com (3).png',width= 20, use_column_width = True)
st.markdown("<br>", unsafe_allow_html = True)

st.markdown("<p style = 'text-align: justify'>The predictive house price modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting the market value of residential properties. By analyzing historical data, identifying key features influencing house prices, and employing advanced regression algorithms, the project seeks to provide valuable insights for homebuyers, sellers, and real estate professionals. The primary objective of this project is to create a reliable machine learning model that accurately predicts house prices based on relevant features such as location, size, number of bedrooms, amenities, and other influencing factors. The model should be versatile enough to adapt to different real estate markets, providing meaningful predictions for a wide range of properties.", unsafe_allow_html= True)

st.sidebar.image('pngwing.com (1).png',caption = 'Welcome Dear User')

st.dataframe(data, use_container_width= True)

input_choice = st.sidebar.radio('choose your input type',['slider input','number input'])

if input_choice == 'slider input':
    area_income = st.sidebar.slider('Average area income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.slider('Average area house age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_number = st.sidebar.slider('Average number of Rooms', data['Avg. Area Number of Rooms'].min(),data['Avg. Area Number of Rooms'].max())
    bedroom_number = st.sidebar.slider('Average number of bed rooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population = st.sidebar.slider('Area population', data['Area Population'].min(),data['Area Population'].max() )
else: 
    st.markdown('<br>', unsafe_allow_html = True)
    area_income = st.sidebar.number_input('Average area income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.number_input('Average area house age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_number = st.sidebar.number_input('Average number of Rooms', data['Avg. Area Number of Rooms'].min(),data['Avg. Area Number of Rooms'].max())
    bedroom_number = st.sidebar.number_input('Average number of bed rooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population = st.sidebar.number_input('Area population', data['Area Population'].min(),data['Area Population'].max() )

input_vars =pd.DataFrame({'Avg. Area Income':[area_income],
                         'Avg. Area House Age':[house_age],
                         'Avg. Area Number of Rooms':[room_number],
                         'Avg. Area Number of Bedrooms':[bedroom_number],
                         'Area Population':[area_population]})

st.markdown("<hr class = 'colorful-divider'>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html= True)

st.markdown("<h5 style = 'margin:-30px ;color: olive; font-family: helvetica'> USER INPUT VARAIBLES</h5>",unsafe_allow_html= True)
st.dataframe(input_vars)
predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')
