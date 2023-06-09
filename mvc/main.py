import os
import shutil
import pickle
import typing
import pandas as pd
from pydantic import BaseModel
import torch
import tensorflow as tf
from google.cloud import storage, bigquery, aiplatform


class MvcModelVersion(BaseModel):
    version_id: int
    params: dict = {}
    training_runs: list[dict] = []
    prediction_runs: list[dict] = []
    metric_summary: list[dict] = []
    tensorboards: list[dict] = []

class McvModelModel(BaseModel):
    model_name: str
    model_type: str
    latest_version: int
    default_version: int
    versions: list[MvcModelVersion]

class MvcServiceModel(BaseModel):
    datasets: list[str]
    models: dict[str, McvModelModel] # model_name: McvModelModel
    endpoints: dict[str, aiplatform.models.Endpoint] # model_name: Enpoint

    class Config:
        arbitrary_types_allowed = True


def storage_driver(func):
    def wrapped(*args, **kwargs):
        try:
            args[0].storage_client = storage.Client(project=args[0].project_id)
            result = func(*args, **kwargs)
        except Exception as e:
            raise e
        return result
    return wrapped


class ModelVersionController():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_id: str = os.environ["PROJECT_ID"]
        self.region: str = os.environ["REGION"]
        self.storage_client: storage.Client = None
        # self.bq_client: bigquery.Client = None
        self.services: dict[str, MvcServiceModel] = {} # service name
        self.service_config = os.environ["SERVICES_CONFIGED"].split(",")

        for service_name in self.service_config:
            # prefetch datasets
            datasets = self.list_datasets(service_name=service_name, init=True)
            self.services[service_name] = MvcServiceModel(
                datasets=datasets,
                models={},
                endpoints={},
            )
            # prefetch models
            aiplatform.init(
                project=self.project_id,
                location=self.region,
                staging_bucket=f"gs://{service_name}",
            )
            for model in aiplatform.Model.list():
                if service_name in model.display_name:
                    file_name = model.display_name.split("/")[-1]
                    self.download_model_meta(service_name=service_name, model_name=file_name)
            # prefetch endpoints
            for end in list(aiplatform.Endpoint.list()):
                end_model = end.list_models()
                if len(end_model) > 0:
                    # TODO - multi-model endpoint unavailable
                    model_name = end_model[0].display_name
                    if service_name in model_name:
                        self.services[service_name].endpoints[model_name] = end

    def gen_file_path(self, dataset_name: str, version: str, file_format: str):
        return f"{dataset_name}_{version}.{file_format}"
    
    def gen_dataset_storage_path(self, service_name: str, dataset_name: str, version: str, file_format: str):
        file_path = self.gen_file_path(dataset_name=dataset_name, version=version, file_format=file_format)
        return f"gs://{service_name}/{file_path}"
    
    def gen_gcs_file_path(self, service_name: str, dataset_name: str, version: typing.Union[str, None] = None, file_format: str = "csv"):
        datasets = [d for d in self.services[service_name].datasets if dataset_name in d]

        if not version:
            curr_version = str(max([int(d.split(".")[0].split("_")[-1]) for d in datasets]))
            file_path = self.gen_dataset_storage_path(service_name=service_name, dataset_name=dataset_name, version=curr_version, file_format=file_format)
        else:
            file_path = self.gen_dataset_storage_path(service_name=service_name, dataset_name=dataset_name, version=version, file_format=file_format)

        return file_path

    @storage_driver
    def create_dataset(self, df: pd.DataFrame, service_name: str, dataset_name: str, new_version: bool = False, file_format: str = "csv"):
        datasets = self.list_datasets(service_name)
        d_avail = [d for d in datasets if dataset_name in d]

        overwrite = False
        if len(d_avail) < 1:
            version = "1"
        else:
            versions = [int(d.split(".")[0].split("_")[-1]) for d in d_avail]
            latest_version = max(versions)
            if new_version:
                version = str(latest_version + 1)
            else:
                overwrite = True
                version = str(latest_version)

        if overwrite:
            self._delete_datasets(service_name=service_name, dataset_name=dataset_name, version=version, file_format=file_format)

        storage_path = self.gen_dataset_storage_path(service_name=service_name, dataset_name=dataset_name, version=version, file_format=file_format)     
        df.to_csv(storage_path, index=False)

        return self.list_datasets(service_name)

    @storage_driver
    def list_datasets(self, service_name: str, init=False):
        if not init:
            assert service_name in self.services, "service name not in CONFIG"

        buckets_avail = [b.name for b in self.storage_client.list_buckets()]
        if service_name not in buckets_avail:
            print(f"creating new bucket with service name {service_name}")
            try:
                self.storage_client.create_bucket(service_name)
            except Exception as e:
                print(f"bucket {service_name} already exists - error {e}")
        bucket = self.storage_client.bucket(service_name)
        blobs = bucket.list_blobs()
        datasets = [b.name for b in blobs if "vertex_ai_auto_staging" not in b.name]

        if not init:
            self.services[service_name].datasets = datasets

        return datasets

    def get_dataset(self, service_name: str, dataset_name: str, version: typing.Union[str, None] = None, file_format: str = "csv"):
        datasets = [d for d in self.services[service_name].datasets if dataset_name in d]
        file_path = self.gen_gcs_file_path(service_name=service_name, dataset_name=dataset_name, version=version, file_format=file_format)
        for d in datasets:
            if d in file_path:
                df = pd.read_csv(file_path)
                if "TIMESTAMP" in df.columns:
                    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
                return df
        return pd.DataFrame()

    @storage_driver
    def _delete_datasets(self, service_name: str, dataset_name: str, version: typing.Union[str, None] = None, file_format: str = "csv"):
        bucket = self.storage_client.bucket(service_name)
        blobs = bucket.list_blobs()
        for blob in blobs:
            if version is not None:
                file_name = self.gen_file_path(dataset_name=dataset_name, version=version, file_format=file_format)
                if file_name == blob.name:
                    blob.delete()
            else:
                if dataset_name in blob.name:
                    blob.delete()

    def create_vertex_dataset(self, service_name: str, dataset_name: str, version: typing.Union[str, None] = None):
        file_path = self.gen_gcs_file_path(service_name=service_name, dataset_name=dataset_name, version=version)

        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=service_name,
        )

        for d in aiplatform.TimeSeriesDataset.list():
            if d.display_name == dataset_name:
                d.delete()

        aiplatform.TimeSeriesDataset.create(
            display_name=dataset_name,
            gcs_source=[file_path],
        )

    def _model_storage_path(self, service_name: str, file_name: str) -> str:
        return f"{service_name}/{file_name}"

    def unpickle_model_meta(self, blob):
        model_bytes = blob.download_as_bytes()
        return pickle.loads(model_bytes)
    
    def _gen_model_meta_blob_name(self, service_name: str, model_name: str):
        return f"{service_name}_{model_name}.pkl"

    def upload_model_meta(self, service_name: str, model_metas: McvModelModel):
        bucket_name = 'model_metadata'
        blob_name = self._gen_model_meta_blob_name(service_name=service_name, model_name=model_metas.model_name)
        model_bytes = pickle.dumps(model_metas)
        bucket = self.storage_client.bucket(bucket_name)

        if not bucket.exists():
            bucket.create()

        blob = bucket.blob(blob_name)

        model_bytes = pickle.dumps(model_metas)
        blob.upload_from_string(model_bytes, content_type='application/pickle')
        uploaded_model_meta = self.unpickle_model_meta(blob)
        self.services[service_name].models[model_metas.model_name] = uploaded_model_meta

        return self.unpickle_model_meta(blob)

    def download_model_meta(self, service_name: str, model_name: str):
        bucket_name = 'model_metadata'
        bucket = self.storage_client.bucket(bucket_name)
        model_meta_name = self._gen_model_meta_blob_name(service_name=service_name, model_name=model_name)
        blob = bucket.blob(model_meta_name)

        if not blob.exists():
            return None
        
        model_meta = self.unpickle_model_meta(blob)
        self.services[service_name].models[model_name] = model_meta
        
        return self.services[service_name].models[model_name]

    def model_type(self, model_object) -> typing.Union[str, None]:
        if isinstance(model_object, tf.keras.Model):
            return "tensorflow"
        if isinstance(model_object, torch.nn.Module):
            return "pytorch"
        return None
    
    def load_torch(self, model_uri: str):
        bucket_name = model_uri.split("//")[1].split("/")[0]

        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix="vertex_ai_auto_staging")

        for blob in blobs:
            if "saved_model.pb" in blob.name:
                tmp_path = "/tmp/saved_model.pb"
                blob.download_to_filename(tmp_path)
                print("downloading model")
                model_dict = torch.load(tmp_path)
                print("model downloaded")

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                return model_dict
        return None
        
    # VERTEX MODELS
    @storage_driver
    def save_model(self, model_object: typing.Any, service_name: str, model_file_name: str, model_params: dict = None, garbage_collect: bool = True) -> bool: 
        '''
        NOTE: model_params must be passed down for pytorch models to create the best version controlling available given pytorch load procedure
        # this can only be improved further by saving the actual model architecture class (along with its module) - practically infeasible
        # in tensorflow life is simpler (mostly)
        '''
        registry_uri = self._model_storage_path(service_name=service_name, file_name=model_file_name)

        model_type = self.model_type(model_object)

        if model_type is None:
            print("model type not supported")
            return False
        
        if model_type == "pytorch":
            assert model_params != None, "must save model_params for load procedure with pytorch models"

        model_type_check = self.services[service_name].models.get(model_file_name)
        if model_type_check is not None:
            assert model_type_check.model_type == model_type, f"ERROR SAVING: trying to save a {model_type} model type to a {model_type_check.model_type} model"
            
        if model_type == "tensorflow":
            model_object.save(registry_uri)
        else:
            os.makedirs(registry_uri, exist_ok=True)
            torch.save(model_object.state_dict(), os.path.join(registry_uri, 'saved_model.pb'))
        
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{service_name}",
        )

        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))

        if len(models) == 0:
            model_uploaded = aiplatform.Model.upload(
                display_name=registry_uri,
                artifact_uri=registry_uri,
                # container_predict_route="/predict",
                # container_health_route="/health",
                serving_container_image_uri=os.environ["MODEL_PREDICT_CONTAINER_URI"],
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
                serving_container_image_uri=os.environ["MODEL_PREDICT_CONTAINER_URI"],
                is_default_version=True,
                parent_model=parent_model,
                # version_description="This is the first version of the model",
            )

        model_uploaded.wait()

        if garbage_collect:
            shutil.rmtree(registry_uri)


        if model_params is not None:
            model_version_meta = MvcModelVersion(version_id=model_uploaded.version_id, params=model_params)
        else:
            model_version_meta = MvcModelVersion(version_id=model_uploaded.version_id)

        if model_file_name not in self.services[service_name].models:
            try:
                default_version = int([m.version_id for m in model_uploaded.versioning_registry.list_versions() if "default" in m.version_aliases][0])
            except:
                default_version = 0

            model_metas = McvModelModel(
                model_name=model_file_name,
                latest_version=model_uploaded.version_id,
                default_version=default_version,
                model_type=model_type,
                versions=[model_version_meta]
            )
            self.upload_model_meta(service_name=service_name, model_metas=model_metas)
        else:
            # updating version
            self.services[service_name].models[model_file_name].latest_version = model_uploaded.version_id
            self.services[service_name].models[model_file_name].versions.append(model_version_meta)
            updated_metas = self.services[service_name].models[model_file_name]
            self.upload_model_meta(service_name=service_name, model_metas=updated_metas)

        return True

    def get_latest_model_version(self, service_name: str, model_file_name: str, model: aiplatform.Model) -> aiplatform.Model:
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{service_name}",
        )
        registry_uri = self._model_storage_path(service_name=service_name, file_name=model_file_name)
        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))
        model = models[0]
        for m in models[0].versioning_registry.list_versions():
            if int(model.version_id) < int(m.version_id):
                model = m
        return model

    @storage_driver
    def load_model(self, service_name: str, model_file_name: str, latest_dev_version: bool = True):
        '''
        NOTE: when using this function with PYTORCH you will need to load_dict_state from the original model architecture
        - model = mvc.load_model(...)
        -> ModelArchitecture.load_state_dict(model)
        '''
        assert service_name in self.services, f"service {service_name} not found"
        assert model_file_name in self.services[service_name].models, f"model {model_file_name} not found in service {service_name}"
        aiplatform.init(
            project=self.project_id,
            location=self.region,
            staging_bucket=f"gs://{service_name}",
        )

        registry_uri = self._model_storage_path(service_name=service_name, file_name=model_file_name)

        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))

        if len(models) > 0:
            if latest_dev_version:
                model = self.get_latest_model_version(service_name=service_name, model_file_name=model_file_name, model=models[0])
            else:
                model = models[0]

            model_metas = self.services[service_name].models[model_file_name]
            model_type = model_metas.model_type
            if model_type == "tensorflow":
                return tf.keras.models.load_model(model.uri)
            else:
                model_state =  self.load_torch(model.uri)
                return model_state
                    
        return None
    

    def load_torch_model(self, service_name: str, model_file_name: str, model_architecture: dict = None, latest_dev_version: bool = True):
        '''
        Pytorch integration with Vertex only allows for state_dict to be saved and loaded
        Python requires module access so classes cannot be saved in isolation
        Hence this function is required for pytorch models where the architecture code is passed down
        NOTE: model is loaded with correct parameters as they reference the parameters saved for the specific model version
        '''
        model_state = self.load_model(service_name=service_name, model_file_name=model_file_name, latest_dev_version=latest_dev_version)

        registry_uri = self._model_storage_path(service_name=service_name, file_name=model_file_name)

        models = aiplatform.Model.list(filter=(f"display_name={registry_uri}"))

        if len(models) < 1:
            return None
            # if latest_dev_version:
            #     model = self.get_latest_model_version(service_name=service_name, model_file_name=model_file_name, model=models[0])
            # else:
            #     model = models[0]
        model = models[0]
        versions = self.services[service_name].models[model_file_name].versions
        version_found = [v for v in versions if v.version_id == int(model.version_id)]
        if len(version_found) > 0:
            params = version_found[0].params
        else:
            version_ids = [v.version_id for v in versions]
            print(f"model version metadata {model.version_id} not found in {version_ids}")
            return None

        print("params", params)
        print("model_state", version_found[0])
        model = model_architecture(**params)
        model.load_state_dict(model_state)
        model.eval()
        return model

    def predict_endpoint(self, service_name: str, model_name: str, x_instance: typing.Union[pd.DataFrame, None] = None, x_batch: typing.Union[list[pd.DataFrame], None] = None):
        assert x_instance is not None or x_batch is not None, "Must provide either x_instance or x_batch"
        end_name = f"{service_name}/{model_name}"
        if end_name in self.services[service_name].endpoints:
            try:
                if x_instance is not None:
                    predictions = self.services[service_name].endpoints[end_name].predict(instances=x_instance)
                else:
                    predictions = self.services[service_name].endpoints[end_name].batch_predict(
                        instances=x_batch, parameters={"confidence_threshold": 0.5}
                    )
                return predictions
            except Exception as e:
                print(f"PREDICTION ERROR: {e}")
                print(f"model service name {model_name}")
                return None
        else:
            print(f"endpoint for model {model_name} for service {service_name} not found in endpoints")
            return None