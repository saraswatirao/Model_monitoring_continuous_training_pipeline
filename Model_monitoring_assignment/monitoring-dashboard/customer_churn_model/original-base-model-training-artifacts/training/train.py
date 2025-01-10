import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import classification_report
import os
import boto3

aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

train_df = pd.read_csv('train_raw.csv')
test_df = pd.read_csv('test_raw.csv')

X_train = train_df.drop(['Churn'], axis=1)
y_train = train_df['Churn']

X_test = test_df.drop(['Churn'], axis=1)
y_test = test_df['Churn']

# Print dataset shapes
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Vectorize the categorical columns
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_categorical = encoder.fit_transform(X_train[categorical_cols])
X_test_categorical = encoder.transform(X_test[categorical_cols])

# Print shapes of vectorized data
print("Training categorical data shape:", X_train_categorical.shape)
print("Testing categorical data shape:", X_test_categorical.shape)

# Impute missing values in numerical columns
imputer = SimpleImputer(strategy='mean')  # You can choose another strategy like median or most_frequent
X_train_numerical = imputer.fit_transform(X_train[numerical_cols])
X_test_numerical = imputer.transform(X_test[numerical_cols])

# Combine the categorical and numerical columns
X_train = np.concatenate([X_train_categorical, X_train_numerical], axis=1)
X_test = np.concatenate([X_test_categorical, X_test_numerical], axis=1)

# Print shapes of combined data
print("Training data shape after combining numerical columns:", X_train.shape)
print("Testing data shape after combining numerical columns:", X_test.shape)

# Create a logistic regression classifier
clf = LogisticRegression()

# Train the classifier on the training set
clf.fit(X_train, y_train)
print("Model training completed.")

# Make predictions on the training set
pred_train = clf.predict(X_train)

# Make predictions on the testing set
pred_test = clf.predict(X_test)

# Evaluate the model performance on the testing set
print("Testing Set Performance:")
print(classification_report(y_test, pred_test))

model_path = '../churn-classifier-model/model.pkl'
encoder_path = '../churn-classifier-model/encoder.pkl'
imputer_path = '../churn-classifier-model/imputer.pkl'
bucket_name = 'churn-classifier-data'

joblib.dump(clf, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(imputer, imputer_path)

# Upload the model and encoder files to S3
s3.upload_file(model_path, bucket_name, 'models/model_latest.pkl')
s3.upload_file(encoder_path, bucket_name, 'models/encoder_latest.pkl')
s3.upload_file(imputer_path, bucket_name, 'models/imputer_latest.pkl')

os.remove(model_path)
os.remove(encoder_path)
os.remove(imputer_path)

print("Model Saved to S3")

train_df['Predicted_Churn'] = pred_train
train_df.to_csv('training_data_with_predictions.csv', index=False)
