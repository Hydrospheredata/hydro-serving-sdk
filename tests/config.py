config = {
    "http_cluster":
        {
            "ip": "http://localhost",
            "port": "80"
        },
    "grpc_cluster":
        {
            "ip": "localhost",
            "port": "9090"
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
            "get_for_model_version": "/monitoring/metricspec/model_version/{versionId}"
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

HTTP_CLUSTER_ENDPOINT = config["http_cluster"]["ip"] + ":" + config["http_cluster"]["port"]
GRPC_CLUSTER_ENDPOINT = config["grpc_cluster"]["ip"] + ":" + config["grpc_cluster"]["port"]
PATH_TO_SERVING = config["test_model"]["path"] + "/" + config["test_model"]["serving"]
DEFAULT_APP_NAME = config["default_application_name"]
DEFAULT_MODEL_NAME = config["default_model_name"]


# TODO: add error messages to all tests, hard to debug otherwise