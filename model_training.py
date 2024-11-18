from google.cloud import aiplatform

def deploy_model(model, project_id, model_display_name, endpoint_display_name):
    aiplatform.init(project=project_id)
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri='gs://your-bucket/model-artifacts',  # Adjust as needed
        serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest'  # Example for TensorFlow
    )
    endpoint = model.deploy(
        endpoint_display_name=endpoint_display_name,
        machine_type='n1-standard-4'
    )
    return endpoint
