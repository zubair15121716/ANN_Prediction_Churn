import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

### --------------- Loading Models and Preprocessors -----------------

# Loading the trained model
model = tf.keras.models.load_model('model.keras')  

# Load the encoder and scaler pkl files
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## ------------------- Streamlit app ---------------------------------

# App title 
st.title("Customer Churn Prediction")
st.info("""
This app predicts whether a customer is likely to churn based on different attributes.
""")

# User inputs with Defaults sets
geography = st.selectbox('Select Customer Geography:', onehot_encoder_geo.categories_[0], help="Select the country of the customer.")
gender = st.selectbox('Select Gender:', label_encoder_gender.classes_, help="Select the gender of the customer.")
age = st.slider('Select Age:', 18, 100, 40, help="Select the age of the customer.")
balance = st.number_input('Enter Balance:', min_value=0, value=50000, step=1000, help="Enter the balance of the customer.")
credit_score = st.number_input('Enter Credit Score:', min_value=300, max_value=850, value=450, step=10, help="Enter the credit score of the customer.")
estimated_salary = st.number_input('Enter Estimated Salary:', min_value=0, value=50000, step=1000, help="Enter the estimated salary of the customer.")
tenure = st.slider('Select Tenure (Years):', 0, 10, 2, help="How long has the customer been with the company?")
num_of_products = st.slider('Select Number of Products:', 1, 4, 2, help="Select how many products the customer has.")
has_cr_card = st.selectbox('Has Credit Card:', [0, 1], help="Does the customer have a credit card? (0 = No, 1 = Yes)")
is_active_member = st.selectbox('Is Active Member:', [0, 1], help="Is the customer an active member? (0 = No, 1 = Yes)")


st.markdown("---")

# Collecting all input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Fitting One hot encode "Geography"
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combining one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scaling the input data
input_data_scaled = scaler.transform(input_data)

# Making Predictions
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]


st.markdown("---")

with st.expander("View Probability and Percentage"):
    st.write(f'**Churn Probability:** {prediction_probability:.2f}')
    st.write(f'**Churn Percentage:** {(prediction_probability*100):.2f}%')

if prediction_probability  > 0.5:
    st.error("🚨 **The customer is likely to churn!** 🚨")
else:
    st.success("🎉 **The customer is not likely to churn!** 🎉")





