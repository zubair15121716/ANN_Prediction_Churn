import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import datetime

# Load the dataset through pandas
data = pd.read_csv("Churn_Modelling.csv")
# Checking the top 5 rows and column
print(data.head())

## Preprocessing data

## Drop irrelevant columns like Row-Number , "Customer-Id" , " Surname of Customer "
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print(data.head())

## Encode categorical variables

# Gender encoding (Male and Female )
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
print(data.head())

# Geography encoding ( Germany France Spain )
from sklearn.preprocessing import OneHotEncoder
## Using Hot Encoding 
onehot_encoder_geo = OneHotEncoder()

# Creating dummy variables for Geography (OneHot Encoding)
geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
print(geo_encoder)

# Saving the encoders for later use in streamlit and predictions
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo, file)

# Getting the feature names of one-hot encoded Geography columns
geo_encoded_df = pd.DataFrame(geo_encoder, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df.head())

# Combining one-hot encoded columns with original data
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)
print(data.head())



## Spliting the dataset into independent(x) and dependent features(y)
X = data.drop('Exited', axis=1)
Y = data['Exited']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

## Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Preprocessing and splitting completed.")

# --- ANN Model Implementation ---

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dropout


# Initializing the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') ])

# Summary of model
model.summary()

## Tunning the attributes for Adam Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Setting Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=150,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# Save the trained model
model.save('model.keras')


print("Model training completed and saved.")
