from google.cloud import aiplatform

def set_up_monitoring(endpoint_name):
    endpoint = aiplatform.Endpoint(endpoint_name)
    endpoint.set_monitoring(
        enabled=True,
        model_monitoring_config={
            "prediction": {
                "disable_analysis": False
            }
        }
    )
