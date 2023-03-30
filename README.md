
## End-to-End ML ops pipeline controller and collaboration interface built on Google Cloud.

Uses Google Cloud Storage to save datasets where data version controlling is handled by MVC.

Uses Vertex AI model registry to save model, versions and create training runs.

Uses Vertex AI to host endpoints and make predictions.


## Usage

### install package
```bash
pip install git+https://github.com/alexlatif/mvc.git
```

### config enviroment variables
1. Set enviroment variables for credentials to GCP instance. 
2. Configure SERVICES_CONFIGED to specify which ML services MVC should be aware of.
```python
os.environ["PROJECT_ID"] = "proj_name_on_gcp"
os.environ["REGION"] = "us-east1"
os.environ["DEPLOY_COMPUTE"] = "n1-standard-2"
# pre-configured model container
os.environ["MODEL_PREDICT_CONTAINER_URI"]  = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"
# services you want the model controller to be aware of
# NOTE: requires .join to convert to string as mvc will parse the list
os.environ["SERVICES_CONFIGED"] = ",".join(["service_1", "service_2"])
```

### initialize model version controller
```python
import mvc as model_version_controller
mvc = model_version_controller.ModelVersionController()
```

### version control datasets
The datasets are stored in Google Cloud Storage and are version controlled by MVC. 

```python
# list datasets available for a service
mvc.list_datasets(service_name=service_name)
# create a new dataset passing a pandas dataframe
mvc.create_dataset(df=df, dataset_name=dataset_name, service_name=service_name)
# get a dataset as a pandas dataframe
df = mvc.get_dataset(service_name=service_name, dataset_name=dataset_name)
```

### version control models
```python
# save tensorflow model to service
mvc.save_model(service_name=service_name, model_file_name=model_file_name, model_object=model)
# load tensorflow model from service
model_out = mvc.load_model(service_name=service_name, model_file_name=model_file_name)
```

### enpoint predictions
Requires using the Vertex AI GUI to set the default model version for each service as well as chosing a model to deploy to an endpoint. Safer and easier this way.
```python
res = mvc.predict_endpoint(service_name=service_name, model_name=model_file_name, x_instance=holdout_x)
```

#### TODO - train online with custom image containers
#### TODO - update model version after training
#### TODO - predict batches
#### TODO - log and save training and prediction runs to GCS
#### TODO - implement tensorboards saved to GCS
#### TODO - create model summary for monitoring 
#### ---
#### TODO - create generic shareable interface
#### TODO - doc + share