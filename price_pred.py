from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

#load the model

model = joblib.load('rf_model.pkl')

# name the app 
st.title('Price prediction for the Amazon sales.')

# define the fields and features
rating = st.number_input('Rating', min_value=1.0, max_value=5.0, value=5.0, step=0.1)
rating_count = st.number_input('Rating Count', min_value=0, max_value=1500, value=500, step=1)
discounted_price = st.number_input('Discounted Price', min_value=0, max_value=500000, value=200, step=50)
discount_percentage = st.number_input('Discount Percentage', min_value=0.00, max_value=1.00, value=0.50, step=0.01)

# create a button for making predictions
import pandas as pd
if st.button('Predict'):
    # process the field values
    input_data = pd.DataFrame(
        {
            'rating': [rating],
            'rating_count': [rating_count],
            'discounted_price': [discounted_price],
            'discount_percentage': [discount_percentage]
        }
    )
    # scale the data using the same scale used during model training
    scaler = StandardScaler()
    # transform the data
    input_data_scaled = scaler.fit_transform(input_data)
    
    predictions = model.predict(input_data_scaled)
   
    message = f'Your price is: {predictions[0]:.2f}'
    st.success(message)
