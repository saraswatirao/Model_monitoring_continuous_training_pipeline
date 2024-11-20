import boto3
import io
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def least_confidence_sampling():
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    s3_bucket_name = 'sentiment-classifier-data'
    s3_location = 'new-data-unlabeled.txt'
    s3_output_location = 'new-data-to-be-labeled.txt'  # New file for data with predicted class probabilities < 0.5

    # Initialize a connection to the S3 bucket
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    # Download the test data file from S3
    s3_response = s3.get_object(Bucket=s3_bucket_name, Key=s3_location)
    s3_data = s3_response['Body'].read().decode('utf-8')

    # Parse the S3 data and store it in a DataFrame
    data = [line.split(';') for line in s3_data.split('\n') if line.strip()]
    df = pd.DataFrame(data, columns=['Text', 'Label'])

    # Load the pre-trained model, CountVectorizer, and LabelEncoder
    classifier = joblib.load('/opt/airflow/models/sentiment_classifier_model_latest.pkl')
    vectorizer = joblib.load('/opt/airflow/models/sentiment_classifier_vectorizer_latest.pkl')
    original_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    label_encoder = LabelEncoder()
    label_encoder.fit(original_labels)

    # Preprocess the test data using the same vectorizer
    X_test_vec = vectorizer.transform(df['Text'])

    # Make predictions and save data points with predicted class probabilities < 0.5 to a new file on S3
    predictions = classifier.predict(X_test_vec)
    probabilities = classifier.predict_proba(X_test_vec)

    # Create a list to store data points with predicted class probabilities and their confidence scores
    data_confidence = []

    for i, row in df.iterrows():
        text = row['Text']
        label = row['Label']
        predicted_label = label_encoder.inverse_transform([predictions[i]])[0]

        # Calculate confidence score as the maximum predicted class probability
        confidence_score = max(probabilities[i])

        data_confidence.append((text, label, confidence_score))

    # Sort the data by confidence score in ascending order
    data_confidence.sort(key=lambda x: x[2])

    # Select the least confident 150 data points
    selected_data = data_confidence[:150]

    # Create a list to store data points to be labeled
    new_data_to_label = [f"{text}" for text, label, _ in selected_data]

    # Save data points with predicted class probabilities to a new file on S3
    if new_data_to_label:
        new_data_content = '\n'.join(new_data_to_label)
        s3.put_object(Bucket=s3_bucket_name, Key=s3_output_location, Body=new_data_content)

    # Print the count of data points and their confidence scores
    print(f"Least confident 150 data points saved to S3.")
    for i, (_, _, confidence_score) in enumerate(selected_data):
        print(f"Data Point {i + 1}: Confidence Score = {confidence_score}")
