# /////////////////////////////Renderforest///////////////////////////////

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Split into input and output
input_data = heart_data.drop(columns='target', axis=1)
output_data = heart_data['target']

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.2, stratify=output_data, random_state=2
)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Random Forest model
model = RandomForestClassifier(random_state=2)
model.fit(x_train, y_train)

# Predict on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

# Predict on testing data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

# Show accuracies
print("Training Data Accuracy: {:.2f}%".format(training_data_accuracy * 100))
print("Testing Data Accuracy: {:.2f}%".format(test_data_accuracy * 100))

# Column names (same as the dataset)
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Demo input
demo_data = (21,1,0,140,277,0,1,160,1,0,2,0,2)
demo_df = pd.DataFrame([demo_data], columns=feature_names)

# Scale the data
demo_df_scaled = scaler.transform(demo_df)

# Predict
demo_prediction = model.predict(demo_df_scaled)

# Show result
if demo_prediction[0] == 1:
    print('The person has heart disease')
else:
    print('The person does not have heart disease')



