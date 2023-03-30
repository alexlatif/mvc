import pandas as pd
import tensorflow as tf

from main import ModelVersionController

data = {'TIMESTAMP': ['2022-01-01', '2022-01-02', '2022-01-03'],
        'value': [10.0, 1000.0, 30.0]}
df = pd.DataFrame(data)
df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)

service_name = "lstm_options"
service_name = "workout_decisions"
dataset_name = "cookies"
model_file_name = "pos123"

test = ModelVersionController()

# DATASETS
test.list_datasets(service_name=service_name)
test.create_dataset(df=df, dataset_name=dataset_name, service_name=service_name)
df = test.get_dataset(service_name=service_name, dataset_name=dataset_name)
test.create_vertex_dataset(service_name=service_name, dataset_name=dataset_name, version="1")

# MODELS
import random_nn as nn
model = nn.train_random()
holdout_x = nn.holdout_random()
test.save_model(service_name=service_name, model_file_name=model_file_name, model_object=model, garbage_collect=False)
model_out = test.load_model(service_name=service_name, model_file_name=model_file_name)
assert type(model_out) == tf.keras.Sequential, "model not loaded correctly"

res = test.predict_endpoint(service_name=service_name, model_name="model_name", x_instance=holdout_x)
assert len(res.predictions) == 67, "prediction not working"
