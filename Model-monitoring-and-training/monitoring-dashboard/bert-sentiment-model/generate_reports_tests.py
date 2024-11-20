import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetSummaryMetric, ColumnDistributionMetric, ColumnDriftMetric, ColumnMissingValuesMetric
from evidently.ui.workspace import Workspace
import evidently_config as evcfg
import sqlite3
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestAccuracyScore, TestPrecisionByClass

import warnings
import random as rnd

warnings.simplefilter(action='ignore')

train_data_stats_limit = 1000

# Column mappings are mandatory to avoid errors, where evidently tries to self interpret columns in your dataset
# ground_truth is the target column (label), predicted_sentiment has predictions made by the model in the last batch, 
# and text_length is the numerical feature
column_mapping = ColumnMapping(
    target="ground_truth",
    prediction="predicted_sentiment",
    text_features=["input_text"],
    numerical_features=["text_length"]
)

# Dictionary to map class to an integer. Required to interpret the class integers from training data
gt_mapping = {"anger": 0, "fear": 1, "joy": 2, "love": 3, "sadness": 4, "surprise": 5}

def get_data_from_database():

    """
    Connect to the sqlite database and retreive predictions made in the last batch
    """

    conn = sqlite3.connect(evcfg.SQLITE_DB)

    # Fetch datapoints which haven't been reported to the dashboard before. 
    query = "SELECT * FROM predictions WHERE reported = 0"

    df = pd.read_sql_query(query, conn)

    # Given the input text, create a new column specifying the length of the text
    df['text_length'] = df['input_text'].apply(lambda x: len(x))

    return df

def set_predictions_as_reported(df):

    """
    After the report and tests have been generated, sets the last batch of data as reported
    """
    conn = sqlite3.connect(evcfg.SQLITE_DB)

    for _, row in df.iterrows():
        prediction_id = row['prediction_id']
        update_query = "UPDATE predictions SET reported = 1 WHERE prediction_id = ?"
        conn.execute(update_query, (prediction_id,))

    conn.commit()
    conn.close()
    
    print("All new predictions set to reported")

def get_reference_data():

    """
    Returns the reference data to find drift against. In this case, we are using the training data as reference.
    """

    text_to_post_list = []

    with open(evcfg.TRAINING_DATA_PATH, 'r') as file:
        lines = file.readlines()
    
    # Used limited lines in training data to save time, and draw comparable visualizations
    for line in lines[:train_data_stats_limit]:
        parts = line.strip().split(';')
        if len(parts) == 2:
            text_to_post = parts[0]

            text_to_post = text_to_post

            ground_truth = gt_mapping[parts[1]]

            # For convinience setting predicted sentiment as ground truth. The model isn't scoring on training data, and we are
            # not interested in those predictions. However, evidently requires current and reference data to have the same schema
            text_to_post_list.append({'input_text': text_to_post, 'ground_truth': ground_truth, 'predicted_sentiment': ground_truth})

    df = pd.DataFrame(text_to_post_list)

    # Given the input text, create a new column specifying the length of the text
    df['text_length'] = df['input_text'].apply(lambda x: len(x))

    return df       


def create_report(reference_data, predictions):

    """
    Creates a report with DatasetSummaryMetric, ColumnDistributionMetric, ColumnDriftMetric, ClassificationPreset, ColumnMissingValuesMetric
    """

    # Exit if no new predictions were made since the last reported batch
    total_new_records = predictions.shape[0]

    if total_new_records == 0:
        print("No new predictions made by the model. Exiting...")
        exit()

    print(f"{total_new_records} new predictions done by the model. Reporting the same.")

    selected_columns = ['ground_truth', 'predicted_sentiment', 'text_length']

    data_drift_report = Report(
        metrics=[
            # Generates a table of Summary of the Dataset (similar to how pandas generates one)
            DatasetSummaryMetric(),

            # Generates how many predictions were made for each class
            ColumnDistributionMetric(column_name="predicted_sentiment"),

            # Provides a drift report for the column. Since text_length is a discrete variable
            # we are using PSI as the statistical test.
            # The inference data already is different in text length from the training
            # data. Using a much higher threshold for psi to evade that drift alarm.
            # This threshold however isn't large enough to evade the drift where we artificially
            # limit the text length in inference dataset i.e. set test_data_stats_limit in generate_reports.py to 20.
            ColumnDriftMetric(column_name='text_length', stattest='psi', stattest_threshold=0.7),

            # Generates a range of classification metrics such as Precision, Recall etc.
            ClassificationPreset(),

            # Finds out how many labelled examples were available in the last batch
            ColumnMissingValuesMetric(column_name='ground_truth')
        ]
    )

    data_drift_report.run(reference_data=reference_data[selected_columns], current_data=predictions[selected_columns], column_mapping=column_mapping)

    print("Report generation complete.... ")

    return data_drift_report

def create_test_suite(reference_data, predictions):

    """
    Creates a test suite with TestAccuracyScore, and TestPrecisionByClass for class '4' (sadness)
    """

    data_test_suite = TestSuite(
        # For both the tests, the threshold is 0.5 i.e. the tests will fail if the score is below 0.5
        tests = [TestAccuracyScore(gte=0.5),
                 TestPrecisionByClass(gte=0.5, label=4)
                 ],
        tags = ["bert-classifier-tests"]
    )

    # Not all columns from the dataframe are reported. Some such as prediction_id are not needed for the report.
    selected_columns = ['ground_truth', 'predicted_sentiment', 'text_length']

    data_test_suite.run(reference_data=None, current_data=predictions[selected_columns], column_mapping=column_mapping)

    return data_test_suite

if __name__ == '__main__':

    # Returns the original created workspace if it exists already
    ws = Workspace.create(evcfg.WORKSACE_PATH)

    existing_projects = ws.search_project(project_name=evcfg.PROJECT_NAME)
    project = existing_projects[0]

    predictions = get_data_from_database()

    reference_data = get_reference_data()

    report = create_report(reference_data, predictions)
    ws.add_report(project.id, report)

    test_suite = create_test_suite(reference_data, predictions)

    ws.add_test_suite(project.id, test_suite)

    set_predictions_as_reported(predictions)