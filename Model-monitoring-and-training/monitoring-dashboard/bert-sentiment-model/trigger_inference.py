import requests
import evidently_config as evcfg
from tqdm import tqdm
import random as rnd

# rnd.seed(10)

# API endpoint URL where you want to make the POST requests
api_url = evcfg.MODEL_API_URL 


test_data_stats_limit = 50


test_data_char_limit = None

# Read the data from the text file
with open(evcfg.INFERENCE_DATA_PATH, 'r') as file:
    lines = file.readlines()

gt_mapping = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}

# Iterate through the lines, split by semicolon, and make POST requests
for line in tqdm(lines[:test_data_stats_limit]):
    parts = line.strip().split(';')
    if len(parts) == 2:
        text_to_post = parts[0]

        if test_data_char_limit is not None:
            text_to_post = text_to_post[:rnd.randint(1, test_data_char_limit)]

        ground_truth = gt_mapping[parts[1]]
        emotion = parts[1]

        # Create a dictionary with the 'text' parameter

        # We also randomly decide whether to provide a label to this datapoint.
        # This is to simulate a production scenario where labels for all datapoints may not be available

        if not rnd.choice([True, False]):
            ground_truth = None

        data = {'text': text_to_post, 'ground_truth': ground_truth}

        # Make the POST request to the API
        response = requests.post(api_url, json=data)

        # Check the response status and print it
        if response.status_code == 200:
            print(f"POST request successful for text: '{text_to_post}' with emotion: '{emotion}'")
        else:
            print(f"POST request failed for text: '{text_to_post}' with emotion: '{emotion}'")
