config = {
    "runtime": 
        {
            "image": "hydrosphere/serving-runtime-python-3.6",
            "tag": "2.4.0"
        },
    "http_cluster":
        {  
            "url": "http://localhost:80"
        },
    "grpc_cluster":
        {
            "url": "localhost:9090",
            "ssl": False
        },
    "test_model":
        {
            "path": "tests/resources/identity_model",
            "serving": "serving.yaml"
        },
    "api_endpoint": "api/v2",
    "endpoint":
        {"metric_spec": {
            "list": "/monitoring/metricspec",
            "create": "/monitoring/metricspec",
            "get": "/monitoring/metricspec/{specId}",
            "delete": "/monitoring/metricspec/{specId}",
            "get_for_model_version": "/monitoring/metricspec/modelversion/{versionId}"
        },
            "application": {
                "delete": "/application/",
                "generate_input": "/application/generateInputs/",
                "get": "/application/{appName}",
                "applications": "/application",
                "add": "/application",
                "update": "/application"
            },
            "host_selector": {
                "list": "/hostSelector",
                "create": "/hostSelector",
                "get": "/hostSelector/{envName}",
                "delete": "/hostSelector/{envName}"
            },

            "external_model": {
                "register": "/extrenalmodel"
            },

            "model": {
                "list": "/model",
                "all_version": "/model/version",
                "get_version": "/model/version/{versionName}/{version}",
                "get": "/model/{modelId}",
                "delete": "/model/{modelId}",
                "upload": "/model/upload"
            },

            "servable":
                {
                    "servables": "/servable",
                    "deploy": "/servable",
                    "get": "/servable/{name}",
                    "stop": "/servable/{name}",
                    "get_logs": "/servable/{name}/logs"
                }

        },
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

# TODO: add error messages to all tests, hard to debug otherwise
