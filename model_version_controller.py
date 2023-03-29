import shutil
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from google.cloud import bigquery, storage
from google.cloud import aiplatform
import tensorflow as tf

MODEL_PREDICT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"

# what models are available and what version are we using
MODELS_CONFIGED = {
    "lstm_options": "version_1",
}

# TODO online training
# TODO version control data
# TODO batch predictions
# TODO experiments
# TODO model monitoring
# TODO models config as enum?

def bq_driver(func):
    def wrapped(*args, **kwargs):
        try:
            args[0].bq_client = bigquery.Client(project=args[0].project_id)
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        return result
    return wrapped

def storage_driver(func):
    def wrapped(*args, **kwargs):
        try:
            args[0].storage_client = storage.Client(project=args[0].project_id)
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        return result
    return wrapped

class MVCModelLogModel(BaseModel):
    model_filename: str
    prediction_runs: list[dict]
    training_runs: list[dict]
    metric_summary: list[dict]
    tensorboards: list[dict]

class MVCDataServiceModel(BaseModel):
    model_name: str
    version: str
    table_names: list[str]
    table_data: dict[str, pd.DataFrame] = {} # table_name: dataframe
    models: list[str] = [] # strings for model file names
    endpoints: list[str] = [] # strings for model file namess
    # TODO logs: list[MVCLogModel] = []

    class Config:
        arbitrary_types_allowed = True


class MVCModelConfigEnum(Enum):
    model_name = "version_x"

class ModelVersionController():
    '''
    ## End-to-End ML ops pipeline controller and collaboration interface.
    
    ### Parameters:
    model_name (str): name of model bucket stored in Google Storage use by Vertex AI
    refresh_data (bool): chose to use cached version or refresh
    
    ### Returns:
    reverse (pd.DataFrame): dataframe of table
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose: bool = False
        self.project_id: str = "ml-wtz"
        self.region: str = "us-east1"
        self.storage_client: storage.Client = None
        self.bq_client: bigquery.Client = None
        self.services: dict[str, MVCDataServiceModel] = {}
        self.model_config = MODELS_CONFIGED
        # TODO self.model_config = MVCModelConfigEnum

        # prefetch dataservices available from datasets
        model_names = self.list_datasets()
        for i, model_name in enumerate(model_names):
            table_names: list[str] = self.show_dataset(model_name=model_name, init=True)
            self.services[model_name] = (
                MVCDataServiceModel(
                    model_name=model_name,
                    version=self.model_config[model_name],
                    table_names=table_names,
                )
            )

        # autocreate buckets configed but not created
        models_buckets_not_created = [model_name for model_name in self.model_config.keys() if model_name not in self.services.keys()]
        if len(models_buckets_not_created) > 0:
            print(f"creating model buckets not created: {models_buckets_not_created}")
            for model_name in models_buckets_not_created:
                self.create_model_bucket(model_name)
        if self.verbose:
            print(f"{len(model_names)} models prefetched")

        # prefetch models available
        models = aiplatform.Model.list()
        for model_file in models:
            self.services[model_name].models.append(model_file.display_name)
            model_version = self.services[model_name].version
            if self.verbose:
                print(f"found model file: {model_file.display_name}, in service: {model_name}, version: {model_version}")

        # prefetch endpoints available
        endpoints = {}
        for service in ["model_name"]:
            for end in list(aiplatform.Endpoint.list()):
                if service in end.display_name:
                    endpoints[service] = end
                    if self.verbose:
                        print(f"found endpoint: {end.display_name}, in service: {service}")
        self.endpoints = endpoints

    @storage_driver
    def list_buckets(self):
        buckets = self.storage_client.list_buckets()
        return [bucket.name for bucket in buckets if bucket.name in self.services.keys()]

    @storage_driver
    def list_blobs(self, model_name: str):
        model_version = self.services[model_name].version
        blobs = self.storage_client.list_blobs(model_name)
        return [blob.name for blob in blobs if model_version in blob.name]

    @storage_driver
    def create_model_bucket(self, model_bucket_name: str, storage_class='STANDARD'): 

        # if model_bucket_name not in self.services:
        bucket = self.storage_client.bucket(model_bucket_name)
        bucket.storage_class = storage_class
    
        if not bucket.exists():
            bucket = self.storage_client.create_bucket(bucket, location=self.region) 

        self.services[model_bucket_name] = MVCDataServiceModel(model_name=model_bucket_name, version="version_0", table_names=[])
        return f'Service bucket with model_name="{bucket.name}" successfully created. 1) add model_name to MVC config 2) create create_dataset_table'

    @storage_driver
    def delete_model_bucket(self, model_bucket_name: str, confirm: bool = False):
        # assert model_bucket_name in self.services.keys(), f"model_name {model_bucket_name} not found in services"
        assert confirm, "are you sure you want to delete a model service? -> all versions of data and models will be destroyed"
        bucket = self.storage_client.get_bucket(model_bucket_name)
        bucket.delete()
        if model_bucket_name in self.services.keys():
            del self.services[model_bucket_name]
        if self.verbose:
            print(f"Model bucket {bucket.name} deleted")

    @bq_driver
    def list_datasets(self, configed_only=True):
        dataset_req = self.bq_client.list_datasets()  # Make an API request.
        models = [dataset.dataset_id for dataset in dataset_req]
        if configed_only:
            models = [model for model in models if model in self.model_config.keys()]
        if len(models) > 0:
            if self.verbose:
                print(f"Datasets in project {self.project_id}:")
            for dataset in models:
                if self.verbose:
                    print(f"\t{dataset}")
        else:
            if self.verbose:
                print(f"{self.project} project does not contain any datasets.")

        return models

    @bq_driver
    def show_dataset(self, model_name: str, init=False):
        if not init:
            assert model_name in self.services, f"{model_name} not found in services - maybe MVC.create_model_bucket() first?"
        dataset = self.bq_client.get_dataset(model_name)  # Make an API request.

        full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
        if self.verbose:
            print(f"Dataset ID: {full_dataset_id}")
            print("Labels:")
        labels = dataset.labels
        if labels:
            for label, value in labels.items():
                print("\t{}: {}".format(label, value))
        else:
            if self.verbose:
                print("\tDataset has no labels defined.")

        if self.verbose:
            print("Tables:")
        tables_req = self.bq_client.list_tables(dataset)  # Make an API request(s).
        tables = [table.table_id for table in tables_req]
        if tables:
            for table in tables:
                if self.verbose:
                    print("\t{}".format(table))
        else:
            if self.verbose:
                print("\tThis dataset does not contain any tables.")
        
        return tables

    # VERTEX DATASETS
    @bq_driver
    def create_dataset_table(self, df: pd.DataFrame, model_name: str, table_name: str, version: str | None = None, create_new_version: bool = False):
        '''
        ## Create new table as Vertex Dataset
        
        ### Parameters:
        model_name (str): name of model bucket stored in Google Storage use by Vertex AI
        refresh_data (bool): chose to use cached version or refresh
        
        ### Returns:
        reverse (pd.DataFrame): dataframe of table
        '''
        assert model_name in self.services, f"{model_name} not found in services - maybe MVC.create_model_bucket() first?"
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=model_name,
            # custom google.auth.credentials.Credentials
            # credentials=my_credentials,

            # encryption_spec_key_name=my_encryption_key_name,
            # experiment='my-experiment',
            # experiment_description='my experiment decsription'
        )
        # create table name
        current_version = self.services[model_name].version
        if version is None and create_new_version is True:
            old_num = current_version.split("_")[-1]
            new_num = int(old_num) + 1
            current_version = f"version_{new_num}"
        assert current_version == self.services[model_name].version, f"model {model_name} version {version} not found in available services - maybe MVC.create_model() first?"
        bq_table_name = f"{current_version}-{table_name}"

        # TODO currently delete the re-write table if exists, can only update tables once schema validations set
        if bq_table_name in self.services[model_name].table_names:
            self.download_table(model_name=model_name, version=current_version, table_name=table_name, delete=True)
            if self.verbose:
                print(f"overwriting table {bq_table_name}")
        
        # create table uri
        bq_dataset_id = f"{self.project_id}.{model_name}"
        training_data_uri = f"bq://{bq_dataset_id}.{bq_table_name}"
        bq_dataset = bigquery.Dataset(bq_dataset_id)
        self.bq_client.create_dataset(bq_dataset, exists_ok=True)

        # create table
        ds = aiplatform.TabularDataset.create_from_dataframe(
            df_source=df,
            staging_path=training_data_uri,
            display_name=table_name,
        )
        ds.wait()
        # ensure data is in sync
        self.services[model_name].table_names.append(bq_table_name)
        created_table = self.download_table(model_name=model_name, version=current_version, table_name=table_name)
        self.services[model_name].table_data[table_name] = created_table
        return self.services[model_name].table_data[table_name]

    @bq_driver
    def download_table(self, model_name: str, table_name: str, version: str | None = None, delete: bool = False) -> pd.DataFrame:
        '''
        Download table from BigQuery
            - only uses a singular configed version specified in MVC.model_config
            - if version is specified, it must match the configed version, otherwise use configed version
        '''
        assert model_name in self.services, f"{model_name} not found in services - maybe MVC.create_model_bucket() first?"
        if version is None:
            version = self.services[model_name].version
        else:
            assert version == self.services[model_name].version, f"version {version} not found in available services - maybe change MVC.model_config?"
        if not delete:
            assert f"{version}-{table_name}" in self.services[model_name].table_names, f"table {table_name} not found in available services - maybe call MVC.create_dataset_table?"
        
        try:
            bq_dataset_id = f"{self.project_id}.{model_name}"
            bq_table_uri = f"bq://{bq_dataset_id}.{version}-{table_name}"
            prefix = "bq://"
            if bq_table_uri.startswith(prefix):
                bq_table_uri = bq_table_uri[len(prefix) :]
            table = self.bq_client.get_table(bq_table_uri)
            df = self.bq_client.list_rows(table).to_dataframe()

            if delete:
                self.bq_client.delete_table(table)
                self.services[model_name].table_names.remove(f"{version}-{table_name}")
                self.services[model_name].table_data.pop(table_name)
                if self.verbose:
                    print(f"table {table_name} deleted")
            return df

        except Exception as e:
            if self.verbose:
                print(f"error downloading table {table_name} from {bq_table_uri}")
            if delete:
                pass
            else:
                raise ValueError(f"error downloading table {table_name} from {bq_table_uri}")

    def delete_dataset(self, model_name: str, version: str,  table_name: str | None = None):
        if table_name is None:
            if self.verbose:
                print(f"deleting all tables for {model_name}-{version}")
            for table_name in self.services[model_name].table_names:
                if self.verbose:
                    print(f"\t - deleting table {table_name}")
                self.download_table(model_name=model_name, version=version, table_name=table_name, delete=True)
        else:
            self.download_table(model_name=model_name, version=version, table_name=table_name, delete=True)

    def get_table(self, model_name: str, version: str, table_name: str, refresh_data: bool = True) -> pd.DataFrame:
        '''
        ## Get table from local cache or download from BigQuery Vertex AI Datasets

        ### Parameters:
        model_name (str): name of model bucket stored in Google Storage use by Vertex AI
        refresh_data (bool): chose to use cached version or refresh

        ### Returns:
        reverse (pd.DataFrame): dataframe of table 
        '''
        if table_name in self.services[model_name].table_data.keys() and refresh_data is False:
            print(f"table {table} found locally")
            return self.services[model_name].table_data[table_name]
        
        table = self.download_table(model_name=model_name, version=version, table_name=table_name)
        self.services[model_name].table_data[table_name] = table
        if self.verbose:
            print(f"table {table} downloaded")
        return self.services[model_name].table_data[table_name]

    def get_tables(self, model_name: str, version: str) -> list[pd.DataFrame]:
        tables: list[pd.DataFrame] = []
        for table in self.services[model_name].table_names:
            tables.append(self.get_table(model_name=model_name, version=version, table_name=table))
        return tables

    def _model_storage_path(self, model_name: str, version: str, file_name: str) -> str:
        return f"{model_name}/{version}/{file_name}"

    def _model_storage_name(self, model_name: str, version: str, file_name: str) -> str:
        return f"{model_name}-{version}-{file_name}"

    # VERTEX MODELS
    @storage_driver
    def save_model(self, model_object: tf.keras.models.Model, model_name: str, model_file_name: str, garbage_collect: bool = True): 
        assert model_name in self.services.keys(), f"model {model_name} not found in available services - maybe call MVC.create_model?"
        model_version = self.services[model_name].version
        registry_uri = self._model_storage_path(model_name=model_name, version=model_version, file_name=model_file_name)
        
        model_object.save(registry_uri)

        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{model_name}",
        )

        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))

        if len(models) == 0:
            model_uploaded = aiplatform.Model.upload(
                display_name=registry_uri,
                artifact_uri=registry_uri,
                # container_predict_route="/predict",
                # container_health_route="/health",
                serving_container_image_uri=MODEL_PREDICT_CONTAINER_URI,
                is_default_version=True,
                # version_description="This is the first version of the model",
            )

        else:
            parent_model = models[0].resource_name
            
            model_uploaded = aiplatform.Model.upload(
                display_name=registry_uri,
                artifact_uri=registry_uri,
                # container_predict_route="/predict",
                # container_health_route="/health",
                serving_container_image_uri=MODEL_PREDICT_CONTAINER_URI,
                is_default_version=False,
                parent_model=parent_model,
                # version_description="This is the first version of the model",
            )

        model_uploaded.wait()

        if garbage_collect:
            shutil.rmtree(registry_uri)

        if model_file_name not in self.services[model_name].models:
            self.services[model_name].models.append(model_file_name)

        else:
            if self.verbose:
                print(f"model file [{model_file_name}] with new version [{model_uploaded.version_id}] in service [{model_name}/{model_version}]")

        if self.verbose:
            print(f"model file [{model_file_name}] saved in service [{model_name}/{model_version}]")

        return True

    @storage_driver
    def load_model(self, model_name: str, model_file_name: str, delete: bool = False, version: str = None) -> tf.keras.models.Model | bool:
        assert model_name in self.services.keys(), f"model {model_name} not found in available services - maybe call MVC.create_model?"
        model_version = self.services[model_name].version

        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{model_name}",
        )

        registry_uri = self._model_storage_path(model_name=model_name, version=model_version, file_name=model_file_name)

        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))
        
        if len(models) > 0:
            model = models[0]
        else:
            raise Exception(f"model {model_name} not found in available services - maybe call MVC.create_model?")

        if delete:
            assert version is not None, "Must provide version to delete"
            try:
                print(version)
                model.versioning_registry.delete_version(version=version)
                if self.verbose:
                    print(f"service {model_name}, model {model_file_name}, version {version} deleted")
            except Exception as e:
                print(f"model file version {version} not found or deleted {e}")

                return False

            return True
        
        return tf.keras.models.load_model(model.uri)

    def predict_endpoint(self, model_name, x_instance: pd.DataFrame, x_batch: list[pd.DataFrame] = None):
        assert x_instance is not None or x_batch is not None, "Must provide either x_instance or x_batch"

        if model_name in self.endpoints.keys():
            try:
                if x_instance is not None:
                    predictions = self.endpoints[model_name].predict(instances=x_instance)
                else:
                    predictions = self.endpoints[model_name].batch_predict(
                        instances=x_batch, parameters={"confidence_threshold": 0.5}
                    )
                return predictions
            except Exception as e:
                print(f"PREDICTION ERROR: {e}")
                print(f"model service name {model_name}")
                return None
        else:
            print(f"model service name {model_name} not found in endpoints")
            return None
