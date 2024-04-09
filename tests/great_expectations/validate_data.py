import os
from datetime import datetime

import great_expectations as ge
# The DataContext is a central object in Great Expectations that represents
# the entire context in which data expectations are defined and validated.
context = ge.data_context.DataContext()

datasource_name = "fake_news_data"
#  It specifies the path to the data file, the data source name, and the name
# of the expectation suite to apply to this batch.
train_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_train_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")
val_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_val_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")
test_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_test_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")

results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[train_batch, val_batch, test_batch],
    run_id=str(datetime.now()))

print(results)
if results["success"]:
    print("Test suite passed!")
    exit(0)
else:
    print("Test suite failed!")
    exit(1)