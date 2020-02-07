config = {
    "cluster":
        {
            "ip": "http://localhost",
            "port": "80"
        },
    "test_model":
        {
            "path": "tests/resources/model_1",
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

        }
}

CLUSTER_ENDPOINT = config["cluster"]["ip"] + ":" + config["cluster"]["port"]
PATH_TO_SERVING = config["test_model"]["path"] + "/" + config["test_model"]["serving"]
