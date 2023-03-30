
## End-to-End ML ops pipeline controller and collaboration interface built on Google Cloud.

Uses Google Cloud Storage to save datasets where data version controlling is handled by MVC.

Uses Vertex AI model registry to save model, versions and create training runs.

Uses Vertex AI to host endpoints and make predictions.

- Configure SERVICES_CONFIGED to specify which ML services MVC should be aware of.
- Set enviroment variables for credentials to GCP instance.
- Use Vertex AI GUI to set default model version for each service.
- Use Vertex AI GUI to deploy model to endpoint.


#### TODO - train online with custom image containers
#### TODO - update model version after training
#### TODO - predict batches
#### TODO - log and save training and prediction runs to GCS
#### TODO - implement tensorboards saved to GCS
#### ---
#### TODO - create generic shareable interface
#### TODO - doc + catchy medium article