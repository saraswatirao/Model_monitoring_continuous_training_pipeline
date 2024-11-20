import boto3
import io
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os
import time

def evaluate_model(classifier, vectorizer, label_encoder, test_data):
    X_test = test_data['Text']
    y_test = test_data['Label']
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, label_encoder.inverse_transform(y_pred))
    return accuracy

def train_and_evaluate_model():
    # Initialize a connection to the S3 bucket
    s3 = boto3.client('s3')

    # Replace with your S3 bucket name and file keys
    s3_bucket_name = 'sentiment-classifier-data'
    s3_original_training_data_location = 'original-train-data.txt'
    s3_new_labeled_data_location = 'new-data-labeled.txt'
    s3_test_data_location = 'original-test-data.txt'

    # Fetch AWS credentials from environment variables
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

    # Initialize a connection to the S3 bucket using environment variables
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    # Download original training data from S3
    s3_response = s3.get_object(Bucket=s3_bucket_name, Key=s3_original_training_data_location)
    s3_original_training_data = s3_response['Body'].read().decode('utf-8')

    # Download new labeled data from S3
    s3_response = s3.get_object(Bucket=s3_bucket_name, Key=s3_new_labeled_data_location)
    s3_new_labeled_data = s3_response['Body'].read().decode('utf-8')

    # Download test data from S3
    s3_response = s3.get_object(Bucket=s3_bucket_name, Key=s3_test_data_location)
    s3_test_data = s3_response['Body'].read().decode('utf-8')

    # Parse the training data and store it in a DataFrame
    original_training_data = [line.split(';') for line in s3_original_training_data.split('\n') if line.strip()]
    new_labeled_data = [line.split(';') for line in s3_new_labeled_data.split('\n') if line.strip()]
    test_data = [line.split(';') for line in s3_test_data.split('\n') if line.strip()]

    original_training_df = pd.DataFrame(original_training_data, columns=['Text', 'Label'])
    new_labeled_df = pd.DataFrame(new_labeled_data, columns=['Text', 'Label'])
    test_df = pd.DataFrame(test_data, columns=['Text', 'Label'])

    # Combine original and new labeled data for training
    combined_training_data = pd.concat([original_training_df, new_labeled_df], ignore_index=True)

    # Split the combined training data into features (X_train) and labels (y_train)
    X_train = combined_training_data['Text']
    y_train = combined_training_data['Label']

    # Convert training labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Create a CountVectorizer to convert text data into numerical features
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train a logistic regression classifier
    print("Training the model...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_vec, y_train_encoded)

    # Define the model directory
    model_dir = '/opt/airflow/models'

    # Evaluate the model on the test data
    print("Evaluating the model on the test data...")
    accuracy_new_model = evaluate_model(classifier, vectorizer, label_encoder, test_df)

    # Check if the new model's performance is better than the previous one
    previous_model_filename = os.path.join(model_dir, 'sentiment_classifier_model_latest.pkl')
    previous_vectorizer_filename = os.path.join(model_dir, 'sentiment_classifier_vectorizer_latest.pkl')

    if os.path.exists(previous_model_filename):
        previous_model = joblib.load(previous_model_filename)
        previous_vectorizer = joblib.load(previous_vectorizer_filename)

        # Assuming that the data doesn't have any new labels
        accuracy_old_model = evaluate_model(previous_model, previous_vectorizer, label_encoder, test_df)
        print(f"Accuracy of the previous model: {accuracy_old_model:.2%}")
        print(f"Accuracy of the new model: {accuracy_new_model:.2%}")

        if accuracy_new_model > accuracy_old_model:
            # Rename the previous model with a backup suffix
            timestamp = int(time.time())
            backup_model_filename = os.path.join(model_dir, f'sentiment_classifier_model_bkp_{timestamp}.pkl')
            backup_vectorizer_filename = os.path.join(model_dir, f'sentiment_classifier_vectorizer_bkp_{timestamp}.pkl')
            os.rename(previous_model_filename, backup_model_filename)
            os.rename(previous_vectorizer_filename, backup_vectorizer_filename)
            print(f"Previous model renamed to: {backup_model_filename}")

            # Save the new model
            model_filename = os.path.join(model_dir, 'sentiment_classifier_model_latest.pkl')
            vectorizer_filename = os.path.join(model_dir, 'sentiment_classifier_vectorizer_latest.pkl')
            joblib.dump(classifier, model_filename)
            print(f"New model saved as: {model_filename}")

            joblib.dump(vectorizer, vectorizer_filename)
            print(f"New vectorizer saved as: {vectorizer_filename}")
        else:
            print("New model performance is not better. Discarding the current model.")
    else:
        # Save the new model if there's no previous model
        model_filename = os.path.join(model_dir, 'sentiment_classifier_model_latest.pkl')
        joblib.dump(classifier, model_filename)
        print(f"New model saved as: {model_filename}")

if __name__ == "__main__":
    train_and_evaluate_model()