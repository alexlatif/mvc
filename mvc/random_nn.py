# import argparse
import tensorflow_cloud as tfc
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
from google.cloud import bigquery

PROJECT_ID = "wtz-ml"


# import pickle
# tfc.remote() is True
# import keras]

def random_data():
    bq_source = "bigquery-public-data.ml_datasets.penguins"

    # Download a table
    bq_client = bigquery.Client(project=os.getenv("PROJECT_ID"))
    table = bq_client.get_table(bq_source)
    df = bq_client.list_rows(table).to_dataframe()

    # Drop unusable rows
    NA_VALUES = ["NA", "."]
    df = df.replace(to_replace=NA_VALUES, value=np.NaN).dropna()

    # Convert categorical columns to numeric
    df["island"], _ = pd.factorize(df["island"])
    df["species"], _ = pd.factorize(df["species"])
    df["sex"], _ = pd.factorize(df["sex"])

    # Split into a training and holdout dataset
    df_train = df.sample(frac=0.8, random_state=100)
    df_holdout = df[~df.index.isin(df_train.index)]

    return df_train, df_holdout

# download_table(training_data_uri)
# df_validation = download_table(validation_data_uri)
# df_test = download_table(test_data_uri)

def convert_dataframe_to_dataset(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
):
    LABEL_COLUMN = "species"
    df_train_x, df_train_y = df_train, df_train.pop(LABEL_COLUMN)
    df_validation_x, df_validation_y = df_validation, df_validation.pop(LABEL_COLUMN)

    y_train = np.asarray(df_train_y).astype("float32")
    y_validation = np.asarray(df_validation_y).astype("float32")

    # Convert to numpy representation
    x_train = np.asarray(df_train_x) 
    x_test = np.asarray(df_validation_x)

    # Convert to one-hot representation
    num_species = len(df_train_y.unique())
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_species)
    y_validation = tf.keras.utils.to_categorical(y_validation, num_classes=num_species)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_validation = tf.data.Dataset.from_tensor_slices((x_test, y_validation))
    return (dataset_train, dataset_validation)

def create_model(num_features):
    Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
        [
            Dense(
                100,
                activation=tf.nn.relu,
                kernel_initializer="uniform",
                input_dim=num_features,
            ),
            Dense(75, activation=tf.nn.relu),
            Dense(50, activation=tf.nn.relu),            
            Dense(25, activation=tf.nn.relu),
            Dense(3, activation=tf.nn.softmax),
        ]
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
        
    return model

def train_model(model: tf.keras.Model, dataset_train: pd.DataFrame, dataset_validation: pd.DataFrame):
    BATCH_SIZE = 10
    dataset_train = dataset_train.batch(BATCH_SIZE)
    dataset_validation = dataset_validation.batch(BATCH_SIZE)

    SAVE_EPOCH = 5
    SERVICE_NAME = "model_name"
    MODEL_VERSION = "version_x"
    MODEL_FILE_NAME = "model1"
    checkpoint_path = f"gs://{SERVICE_NAME}.{MODEL_VERSION}-{MODEL_FILE_NAME}-save_at_{SAVE_EPOCH}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # prod
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--label_column', required=True, type=str)
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--batch_size', default=10, type=int)
    # args = parser.parse_args(args=['--label_column', '--epochs', '--batch_size'])

    # # Read environmental variables
    # training_data_uri = os.getenv("AIP_TRAINING_DATA_URI")
    # validation_data_uri = os.getenv("AIP_VALIDATION_DATA_URI")
    # test_data_uri = os.getenv("AIP_TEST_DATA_URI")

    tensorboard_path = os.path.join(
        f"gs://{SERVICE_NAME}.{MODEL_VERSION}-{MODEL_FILE_NAME}-save_at_{SAVE_EPOCH}"
        f"gs://{SERVICE_NAME}.{MODEL_VERSION}-{MODEL_FILE_NAME}-LOG-{timestamp}"
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]

    if tfc.remote():
        epochs = 20
    else:
        epochs = 10
        callbacks = None
    
    model.fit(dataset_train, epochs=epochs, validation_data=dataset_validation)

    return model


def train_random():
    df_train, _ = random_data()
    train=df_train.sample(frac=0.8,random_state=200)
    validation=df_train.drop(train.index)
    dataset_train, dataset_validation = convert_dataframe_to_dataset(train, validation)
    dataset_train = dataset_train.shuffle(len(df_train))

    model: tf.keras.Model = create_model(num_features=dataset_train._flat_shapes[0].dims[0].value)

    return train_model(model, dataset_train, dataset_validation)



def test_random(model: tf.keras.Model):
    LABEL_COLUMN = "species"
    _, df_holdout = random_data()
    df_holdout_y = df_holdout.pop(LABEL_COLUMN)
    df_holdout_x = df_holdout

    # Convert to list representation
    holdout_x = np.array(df_holdout_x).tolist()
    holdout_y = np.array(df_holdout_y).astype("float32").tolist()

    predictions = model.predict(holdout_x)
    y_predicted = np.argmax(predictions, axis=1)
    (holdout_y - y_predicted).sum() / len(holdout_y)


def holdout_random():
    LABEL_COLUMN = "species"
    _, df_holdout = random_data()
    df_holdout.pop(LABEL_COLUMN)
    df_holdout_x = df_holdout

    # Convert to list representation
    return np.array(df_holdout_x).tolist()
