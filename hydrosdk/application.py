from collections import namedtuple


ApplicationDef = namedtuple('ApplicationDef', ('name', 'executionGraph', 'kafkaStreaming'))


def streaming_params(in_topic, out_topic):
    return {
        'sourceTopic': in_topic,
        'destinationTopic': out_topic
    }


class Application:
    def __init__(self, name, execution_graph, kafka_streaming, metadata):
        self.name = name
        self.execution_graph = execution_graph
        self.kafka_streaming = kafka_streaming
        self.metadata = metadata

    @staticmethod
    def app_json_to_app_obj(application_json):
        app_name = application_json.get("name")
        app_execution_graph = application_json.get("executionGraph")
        app_kafka_streaming = application_json.get("kafkaStreaming")
        app_metadata = application_json.get("metadata")
        app = Application(name=app_name, execution_graph=app_execution_graph,
                          kafka_streaming=app_kafka_streaming, metadata=app_metadata)
        return app

    @staticmethod
    def list_all(cluster):
        """
        Returns:
            list of app_objs:
        """
        resp = cluster.request("GET", "/api/v2/application")
        if resp.ok:
            resp_json = resp.json()
            applications = [Application.app_json_to_app_obj(app_json) for app_json in resp_json]
            return applications

        raise Exception(
            f"Failed to list all models. {resp.status_code} {resp.text}")

    @staticmethod
    def find_by_name(cluster, app_name):
        """
        Args:
            cluster (Cluster)
            app_name (str):
        """
        resp = cluster.request("GET", "/api/v2/application/{}".format(app_name))
        if resp.ok:
            resp_json = resp.json()
            app = Application.app_json_to_app_obj(resp_json)
            return app

        raise Exception(
            f"Failed to find by name. Name = {app_name}. {resp.status_code} {resp.text}")

    @staticmethod
    def delete(cluster, app_name):
        resp = cluster.request("DELETE", "/api/v2/application/{}".format(app_name))
        if resp.ok:
            return resp.json()

        raise Exception(
            f"Failed to delete application. Name = {app_name}. {resp.status_code} {resp.text}")

    @staticmethod
    def create(cluster, application: dict):
        resp = cluster.request(method="POST", url="/api/v2/application", json=application)
        if resp.ok:
            resp_json = resp.json()
            app = Application.app_json_to_app_obj(resp_json)
            return app

        raise Exception(
            f"Failed to create application. Application = {application}. {resp.status_code} {resp.text}")

    @staticmethod
    def parse_streaming_params(in_list):
        """
        Args:
            in_list (list of dict):
        Returns:
            StreamingParams:
        """
        params = []
        for item in in_list:
            params.append(streaming_params(item["in-topic"], item["out-topic"]))
        return params

    @staticmethod
    def parse_singular_app(in_dict):
        return {
            "stages": [
                {
                    "modelVariants": [Application.parse_singular(in_dict)]
                }
            ]
        }

    @staticmethod
    def parse_singular(in_dict):
        return {
            'modelVersionId': in_dict['model'],
            'weight': 100
        }

    @staticmethod
    def parse_model_variant_list(in_list):
        services = [
            Application.parse_model_variant(x)
            for x in in_list
        ]
        return services

    @staticmethod
    def parse_model_variant(in_dict):
        return {
            'modelVersion': in_dict['model'],
            'weight': in_dict['weight']
        }

    @staticmethod
    def parse_pipeline_stage(stage_dict):
        if len(stage_dict) == 1:
            parsed_variants = [Application.parse_singular(stage_dict[0])]
        else:
            parsed_variants = Application.parse_model_variant_list(stage_dict)
        return {"modelVariants": parsed_variants}

    @staticmethod
    def parse_pipeline(in_list):
        pipeline_stages = []
        for i, stage in enumerate(in_list):
            pipeline_stages.append(Application.parse_pipeline_stage(stage))
        return {'stages': pipeline_stages}

    @staticmethod
    def parse_application(in_dict):
        singular_def = in_dict.get("singular")
        pipeline_def = in_dict.get("pipeline")

        streaming_def = in_dict.get('streaming')
        if streaming_def:
            streaming_def = Application.parse_streaming_params(streaming_def)

        if singular_def and pipeline_def:
            raise ValueError("Both singular and pipeline definitions are provided")

        if singular_def:
            executionGraph = Application.parse_singular_app(singular_def)
        elif pipeline_def:
            executionGraph = Application.parse_pipeline(pipeline_def)
        else:
            raise ValueError("Neither model nor graph are defined")

        return ApplicationDef(
            name=in_dict['name'],
            executionGraph=executionGraph,
            kafkaStreaming=streaming_def
        )
