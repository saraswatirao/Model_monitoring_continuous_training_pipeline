from evidently.report import Report
from evidently.metrics import DatasetSummaryMetric, ColumnDistributionMetric
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestAccuracyScore, TestPrecisionByClass, TestColumnDrift

import warnings

warnings.simplefilter(action='ignore')

column_mapping = ColumnMapping(
    target="Churn",
    prediction="Predicted_Churn",
    numerical_features=["tenure", "MonthlyCharges", "TotalCharges"],
    categorical_features=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']
)

def get_report(reference_data, predictions):

    """
    Creates a report with DatasetSummaryMetric, ColumnDistributionMetric, ColumnDriftMetric, ClassificationPreset, ColumnMissingValuesMetric
    """

    # Exit if no new predictions were made since the last reported batch
    total_new_records = predictions.shape[0]

    if total_new_records == 0:
        print("No new predictions made by the model. Exiting...")
        exit()

    print(f"{total_new_records} new predictions done by the model. Reporting the same.")

    data_drift_report = Report(
        metrics=[
            # Generates a table of Summary of the Dataset (similar to how pandas generates one)
            DatasetSummaryMetric(),

            # Generates how many predictions were made for each class
            ColumnDistributionMetric(column_name="Predicted_Churn"),

            # Generates a range of classification metrics such as Precision, Recall etc.
            ClassificationPreset(),
        ]
    )

    data_drift_report.run(reference_data=reference_data, current_data=predictions, column_mapping=column_mapping)

    print("Report generation complete.... ")

    return data_drift_report

def get_test_suite(reference_data, predictions):

    """
    Creates a test suite with TestAccuracyScore, TestPrecisionByClass for class '1' (churn), TestColumnDrift
    """

    data_test_suite = TestSuite(
        tests = [
                 TestAccuracyScore(gte=0.5),
                 TestPrecisionByClass(gte=0.5, label=1),
                 TestColumnDrift(column_name='tenure', stattest='psi', stattest_threshold=0.1),
                 TestColumnDrift(column_name='MonthlyCharges', stattest='ks', stattest_threshold=0.05),
                 TestColumnDrift(column_name='TotalCharges', stattest='ks', stattest_threshold=0.05)
                ],
        tags = ["churn-classifier-tests"]
    )

    data_test_suite.run(reference_data=reference_data, current_data=predictions, column_mapping=column_mapping)

    print("Test Suite generation complete.... ")

    return data_test_suite