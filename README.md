
# MVC (Model Version Controller)
## An open-source end-to-End ML ops pipeline controller and collaboration interface built on Google Cloud.

![version control](https://github.com/alexlatif/mvc/blob/main/img.jpg)

When it comes to version controlling datasets and machine learning models Amazon and Google don't fully support this, tools like DVC are complex, tools like Neptune are great, but f#%$ paying for ML infrastructure beyond compute. This is open-source, simple and will cover everything you need to deploy robust and monitored machine learning models on the latest Google hardware.

### Features:
1. Save, version control and share datasets on Google Cloud Storage.
2. Save, version control and share models on Vertex AI.
3. Create training runs on Vertex AI.
4. Make predictions on endpoints on Vertex AI.
5. Monitor model performance in training and prediction runs.
6. Share notebooks on Colab and Workbench easily.

### Values:
1. Open-source
2. Simple
3. Pythonic

### To collaborate:
1. Request features in the discussion sections
2. To extend email me at ale@watz.coach (please avoid forking where possible as it would be great to collaborate to improve MVC for everyone)

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



## TODO's
- [ ] train online with custom image containers
- [ ] update model version after training
- [ ] predict batches
- [ ] log and save training and prediction runs to GCS
- [ ] implement tensorboards saved to GCS
- [ ] create model summary for monitoring 
- [ ] tests. lol.