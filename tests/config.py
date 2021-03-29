config = {
    "runtime": 
        {
            "image": "hydrosphere/serving-runtime-python-3.7",
            "tag": "2.4.0"
        },
    "http_cluster":
        {  
            "url": "http://localhost",
        },
    "grpc_cluster":
        {
            "url": "localhost:9090",
            "ssl": False
        },
    "lock_timeout": 120,
    "default_application_name": "infer",
    "default_model_name": "infer"
}

HTTP_CLUSTER_ENDPOINT = config["http_cluster"]["url"]
GRPC_CLUSTER_ENDPOINT = config["grpc_cluster"]["url"]
GRPC_CLUSTER_ENDPOINT_SSL = config["grpc_cluster"]["ssl"]

DEFAULT_APP_NAME = config["default_application_name"]
DEFAULT_MODEL_NAME = config["default_model_name"]
DEFAULT_RUNTIME_IMAGE = config["runtime"]["image"]
DEFAULT_RUNTIME_TAG = config["runtime"]["tag"]
DEFAULT_RUNTIME_REFERENCE = f"{DEFAULT_RUNTIME_IMAGE}:{DEFAULT_RUNTIME_TAG}"
LOCK_TIMEOUT = config["lock_timeout"]
