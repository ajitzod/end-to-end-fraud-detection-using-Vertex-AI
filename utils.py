import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def log_metrics(metrics):
    logging.info(f"Model metrics: {metrics}")
