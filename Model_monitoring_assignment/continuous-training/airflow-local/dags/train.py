import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import boto3
import os

aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']


def train_and_evaluate_model(original_train_data, new_train_data, bucket_name):

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # Read the original and the new data file from S3
    original_train_df = pd.read_csv(s3.get_object(Bucket=bucket_name, Key=original_train_data)['Body'])
    new_data_train_df = pd.read_csv(s3.get_object(Bucket=bucket_name, Key=new_train_data)['Body'])

    train_df = pd.concat([original_train_df, new_data_train_df], axis=0, ignore_index=True)

    X_train = train_df.drop(['Churn'], axis=1)
    y_train = train_df['Churn']

    # Print dataset shapes
    print("Training set shape:", X_train.shape)

    # Vectorize the categorical columns
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train_categorical = encoder.fit_transform(X_train[categorical_cols])

    # Print shapes of vectorized data
    print("Training categorical data shape:", X_train_categorical.shape)

    # Impute missing values in numerical columns
    imputer = SimpleImputer(strategy='mean')  # You can choose another strategy like median or most_frequent
    X_train_numerical = imputer.fit_transform(X_train[numerical_cols])

    # Combine the categorical and numerical columns
    X_train = np.concatenate([X_train_categorical, X_train_numerical], axis=1)

    # Print shapes of combined data
    print("Training data shape after combining numerical columns:", X_train.shape)

    # Create a logistic regression classifier
    clf = LogisticRegression()

    # Train the classifier on the training set
    clf.fit(X_train, y_train)
    print("Model training completed.")

    joblib.dump(clf, 'model_latest.pkl')
    joblib.dump(encoder, 'encoder_latest.pkl')
    joblib.dump(imputer, 'imputer_latest.pkl')

    # Upload the model and encoder files to S3
    s3.upload_file('model_latest.pkl', bucket_name, 'models/model_latest.pkl')
    s3.upload_file('encoder_latest.pkl', bucket_name, 'models/encoder_latest.pkl')
    s3.upload_file('imputer_latest.pkl', bucket_name, 'models/imputer_latest.pkl')

    os.remove('model_latest.pkl')
    os.remove('encoder_latest.pkl')

    print("New Model Saved to S3")
