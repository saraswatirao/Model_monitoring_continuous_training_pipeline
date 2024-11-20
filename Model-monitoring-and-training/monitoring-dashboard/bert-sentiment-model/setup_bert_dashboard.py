import os
from evidently.metrics import DatasetSummaryMetric, ColumnMissingValuesMetric, ColumnDriftMetric
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.workspace import Workspace
from evidently.renderers.html_widgets import WidgetSize
import evidently_config as evcfg
import shutil
import sqlite3

# Define a function to create a project within a workspace
def create_project(ws):
    project = ws.create_project(evcfg.PROJECT_NAME)

    # Add a counter panel to the project's dashboard
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="BERT Sentiment Classifier Performance Dashboard",
            size=WidgetSize.FULL,
        )
    )

    # Add counter panels for various metrics to the dashboard
    # such as Model Calls, Unlabelled Inference Datapoints, Model Precision, Model Accuracy, and Drift Score
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path=DatasetSummaryMetric.fields.current.number_of_rows,
                legend="count",
            ),
            agg=CounterAgg.SUM,
            size=WidgetSize.FULL,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Unlabelled Inference Datapoints",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ColumnMissingValuesMetric",
                metric_args={"column_name": "ground_truth"},
                field_path=ColumnMissingValuesMetric.fields.current.number_of_missing_values,
                legend="count",
            ),
            agg=CounterAgg.SUM,
            size=WidgetSize.FULL,
        )
    )

    # Commented out panel for Predicted Class Distributions

    # project.dashboard.add_panel(
    #     DashboardPanelPlot(
    #         title="Predicted Class Distributions",
    #         filter=ReportFilter(metadata_values={}, tag_values=[]),
    #         values=[
    #             PanelValue(
    #             metric_id="ColumnDistributionMetric",
    #             field_path=ColumnDistributionMetric.fields.current.y,
    #             legend="Predicted Class Distribution",
    #         )],
    #         plot_type=PlotType.BAR,
    #         size=1,
    #     )
    # )

    # Add panels for Model Precision and Model Accuracy
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Model Precision",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ClassificationQualityMetric",
                    field_path="current.precision",
                    legend="precision",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.HALF,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Model Accuracy",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ClassificationQualityMetric",
                    field_path="current.accuracy",
                    legend="accuracy",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.HALF,
        )
    )

    # Add a panel for Drift Score over time
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Drift Score of Text Length",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.FULL,
        )
    )

    # Save the project
    project.save()
    return project

# Define a function to create a SQLite database
def create_sqlite_db():
    # Check if the database file already exists, and if so, delete it
    if os.path.exists(evcfg.SQLITE_DB):
        print("Database already exists.. Deleting...")
        os.remove(evcfg.SQLITE_DB)
    
    print("Creating Database...")

    # Create a connection to the SQLite database
    conn = sqlite3.connect(evcfg.SQLITE_DB)

    cursor = conn.cursor()

    # Define a SQL query to create a table for predictions
    create_table_query = """
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            predicted_sentiment INTEGER,
            ground_truth INTEGER,
            timestamp DATETIME,
            reported INTEGER
        );
    """
    cursor.execute(create_table_query)

    conn.commit()
    conn.close()

# Define a function to set up the workspace
def setup_workspace():
    # Check if the workspace directory already exists, and if so, delete it
    if os.path.exists(evcfg.WORKSACE_PATH):
        print("Workspace already exists... Deleting existing workspace")
        shutil.rmtree(evcfg.WORKSACE_PATH)
    
    print("Creating Workspace.... ")

    # Create a workspace and project
    ws = Workspace.create(evcfg.WORKSACE_PATH)
    print("Creating Project....")
    project = create_project(ws)

if __name__ == '__main__':
    # Call the setup_workspace function to set up the workspace and project
    setup_workspace()

    # Call the create_sqlite_db function to create the SQLite database
    create_sqlite_db()
