# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load your training data from the file
train_data = []
with open('train-data.txt', 'r') as file:
    for line in file:
        text, label = line.strip().split(';')
        train_data.append((text, label))

# Split the training data into features (X_train) and labels (y_train)
X_train = [sample[0] for sample in train_data]
y_train = [sample[1] for sample in train_data]

# Convert training labels to numerical values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Create a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train a logistic regression classifier
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_vec, y_train_encoded)

# Save the trained model to a file
model_filename = '../sentiment-classifier-model/sentiment_classifier_model.pkl'
joblib.dump(classifier, model_filename)

# Save the vectorizer to a file
vectorizer_filename = '../sentiment-classifier-model/sentiment_classifier_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_filename)

# Load your test data from the file
test_data = []
with open('test-data.txt', 'r') as file:
    for line in file:
        text, label = line.strip().split(';')
        test_data.append((text, label))

# Split the test data into features (X_test) and labels (y_test)
X_test = [sample[0] for sample in test_data]
y_test = [sample[1] for sample in test_data]

# Convert test labels to numerical values using the same label encoder
y_test_encoded = label_encoder.transform(y_test)

# Transform test data using the same vectorizer
X_test_vec = vectorizer.transform(X_test)

# Make predictions on the test data
y_pred = classifier.predict(X_test_vec)

# Convert numerical predictions back to original labels
y_pred_original = label_encoder.inverse_transform(y_pred)

# Evaluate the classifier on the test data
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
