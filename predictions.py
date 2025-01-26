import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np


model = load_model('model.keras')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

input_data = {
    'CreditScore': float(input("Enter Credit Score: ")),
    'Geography': input("Enter Geography (e.g., France, Spain, Germany): "),
    'Gender': input("Enter Gender (Male/Female): "),
    'Age': int(input("Enter Age: ")),
    'Tenure': int(input("Enter Tenure: ")),
    'Balance': float(input("Enter Balance: ")),
    'NumOfProducts': int(input("Enter Number of Products: ")),
    'HasCrCard': int(input("Has Credit Card (1/0): ")),
    'IsActiveMember': int(input("Is Active Member (1/0): ")),
    'EstimatedSalary': float(input("Enter Estimated Salary: "))
}


# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Prepare the input dataframe
input_df = pd.DataFrame([input_data])

# Gender (Label Encoding)
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

# Concatenating the one-hot encoded Geography
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Scaling the input data
input_scaled = scaler.transform(input_df)

# Predicting the churn possibility
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]


if prediction_probability > 0.5:
    print('The customer is likely to churn.')
    print(f'Probability: {prediction_probability:.2f}')
else:
    print('The customer is not likely to churn.')
    print(f'Probability: {prediction_probability:.2f}')




