from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
import evidently_config as evcfg
from datetime import datetime
from typing import Optional


app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = "RinInori/bert-base-uncased_finetuned_sentiments"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def write_prediction_to_db(text, prediction, ground_truth=None):

    conn = sqlite3.connect(evcfg.SQLITE_DB)

    cursor = conn.cursor()

    data_to_insert = (text, prediction, ground_truth, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0)

    insert_query = "INSERT INTO predictions (input_text, predicted_sentiment, ground_truth, timestamp, reported) VALUES (?, ?, ?, ?, ?)"
    cursor.execute(insert_query, data_to_insert)

    conn.commit()
    conn.close()



class TextData(BaseModel):
    text: str
    ground_truth: Optional[int]

@app.post("/predict/")
async def predict_sentiment(data: TextData):
    # Extract the text from the JSON payload
    text = data.text

    ground_truth = data.ground_truth

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get the sentiment prediction
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    write_prediction_to_db(text, predicted_class, ground_truth)

    return {"sentiment_class": predicted_class}
