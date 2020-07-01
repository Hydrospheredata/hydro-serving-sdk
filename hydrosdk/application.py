from collections import namedtuple
from enum import Enum
from typing import List, Dict

from hydro_serving_grpc.contract import ModelSignature

from hydrosdk.cluster import Cluster
from hydrosdk.contract import _signature_dict_to_ModelSignature
from hydrosdk.data.types import PredictorDT
from hydrosdk.predictor import PredictServiceClient, MonitorableImplementation

ApplicationDef = namedtuple('ApplicationDef', ('name', 'executionGraph', 'kafkaStreaming'))


def streaming_params(in_topic, out_topic):
    """
    Deserializes topics into StreamingParams

    :param in_topic: input topic
    :param out_topic: output topic
    :return: StreamingParams
    """
    return {
        'sourceTopic': in_topic,
        'destinationTopic': out_topic
    }


class ApplicationStatus(Enum):
    """
    Possible statuses of an Application
    """
    FAILED = 0
    ASSEMBLING = 1
    READY = 2


class Application:
    """
    An application is a publicly available endpoint to reach your models. Learn more about them - https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications
    """

    def __init__(self, name, execution_graph, kafka_streaming, metadata, status, signature: ModelSignature,
                 cluster: Cluster):
        self.name = name
        self.execution_graph = execution_graph
        self.kafka_streaming = kafka_streaming
        self.metadata = metadata
        self.status = status
        self.signature = signature
        self.cluster = cluster

    def predictor(self, return_type: PredictorDT = PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """
        Returns a predictor object which is used to pass data into the Application.
         Data passed to the Application will be automatically shadowed into monitoring services.

        :param return_type: Specifies into which data format should predictor return Servable outputs
        :return: PredictServiceClient
        """

        return PredictServiceClient(impl=MonitorableImplementation(channel=self.cluster.channel, target=self.name),
                                    signature=self.signature,
                                    return_type=return_type)

    def update_status(self) -> None:
        """
        Setter method that updates application status
        :return: None
        """
        application = Application.find_by_name(cluster=self.cluster, application_name=self.name)
        self.status = application.status

    @staticmethod
    def app_json_to_app_obj(cluster: Cluster, application_json: dict) -> 'Application':
        """
        Deserializes Application JSON representation into an Application
        :param cluster: Hydrosphere cluster
        :param application_json: JSON representation of an Application
        :return Application: Application
        """
        app_name = application_json.get("name")
        app_execution_graph = application_json.get("executionGraph")
        app_kafka_streaming = application_json.get("kafkaStreaming")
        app_metadata = application_json.get("metadata")
        app_signature = _signature_dict_to_ModelSignature(data=application_json.get("signature"))
        app_status = ApplicationStatus[application_json.get("status").upper()]

        app = Application(name=app_name, execution_graph=app_execution_graph, kafka_streaming=app_kafka_streaming,
                          metadata=app_metadata, status=app_status, cluster=cluster, signature=app_signature)
        return app

    @staticmethod
    def list_all(cluster) -> List['Application']:
        """
        Lists all available applications in the cluster

        :param cluster: Hydrosphere cluster
        :raises Exception: If response from server is not 200
        :return: List of all Applications
        """
        resp = cluster.request("GET", "/api/v2/application")
        if resp.ok:
            resp_json = resp.json()
            applications = [Application.app_json_to_app_obj(cluster=cluster, application_json=app_json) for app_json in
                            resp_json]
            return applications

        raise Exception(
            f"Failed to list all models. {resp.status_code} {resp.text}")

    @staticmethod
    def find_by_name(cluster, application_name) -> 'Application':
        """
        Finds an Application in a cluster by its name

        :param cluster: Hydrosphere cluster
        :raises Exception: If response from server is not 200
        :return: Application
        """
        resp = cluster.request("GET", "/api/v2/application/{}".format(application_name))
        if resp.ok:
            resp_json = resp.json()
            app = Application.app_json_to_app_obj(cluster=cluster, application_json=resp_json)
            return app

        raise Exception(
            f"Failed to find by name. Name = {application_name}. {resp.status_code} {resp.text}")

    @staticmethod
    def delete(cluster, application_name):
        """
        Permanently deletes Application from the cluster

        :param cluster: active cluster
        :raises Exception: If response from server is not 200
        :return: response from the server
        """
        resp = cluster.request("DELETE", "/api/v2/application/{}".format(application_name))
        if resp.ok:
            return resp.json()

        raise Exception(
            f"Failed to delete application. Name = {application_name}. {resp.status_code} {resp.text}")

    @staticmethod
    def create(cluster, application: dict) -> 'Application':
        """
        Creates an Application in the cluster from existing ModelVersions.

        :param cluster: Hydrosphere cluster
        :param application: JSON representation of Application created
        :raises Exception: If response from server is not 200
        :return: Application
        """
        resp = cluster.request(method="POST", url="/api/v2/application", json=application)
        if resp.ok:
            resp_json = resp.json()
            app = Application.app_json_to_app_obj(cluster=cluster, application_json=resp_json)
            return app

        raise Exception(
            f"Failed to create application. Application = {application}. {resp.status_code} {resp.text}")

    @staticmethod
    def parse_streaming_params(in_list: List[Dict]) -> List[Dict]:
        """
        Deserializes list of StreamingParams

        :param in_list: input list of Dict
        :return: list of dicts in a form of {'sourceTopic': in_topic, 'destinationTopic': out_topic}
        """
        params = []
        for item in in_list:
            params.append(streaming_params(item["in-topic"], item["out-topic"]))
        return params

    @staticmethod
    def parse_singular_app(in_dict):
        """
        Part of parse_application method, parses singular

        :param in_dict: singular def
        :return: stages with model variants
        """
        return {
            "stages": [
                {
                    "modelVariants": [Application.parse_singular(in_dict)]
                }
            ]
        }

    @staticmethod
    def parse_singular(in_dict):
        """
        Part of parse_application method, parses singular pipeline stage

        :param in_dict: pipieline stage
        :return: model version id and weight
        """
        return {
            'modelVersionId': in_dict['model'],
            'weight': 100
        }

    @staticmethod
    def parse_model_variant_list(in_list) -> list:
        """
        Part of parse_application method, parses list of model variants

        :param in_list: dict with list model variants
        :return: list of services
        """
        services = [
            Application.parse_model_variant(x)
            for x in in_list
        ]
        return services

    @staticmethod
    def parse_model_variant(in_dict):
        """
        Part of parse_application method, parses model variant

        :param in_dict: dict with model variant
        :return: dict with model version and weight
        """
        return {
            'modelVersion': in_dict['model'],
            'weight': in_dict['weight']
        }

    @staticmethod
    def parse_pipeline_stage(stage_dict):
        """
        Part of parse_application method, parses pipeline stages

        :param stage_dict: dict with list of pipeline stages
        :return: dict with list of model variants
        """
        if len(stage_dict) == 1:
            parsed_variants = [Application.parse_singular(stage_dict[0])]
        else:
            parsed_variants = Application.parse_model_variant_list(stage_dict)
        return {"modelVariants": parsed_variants}

    @staticmethod
    def parse_pipeline(in_list):
        """
        Part of parse_application method, parses pipeline

        :param in_list: input list with info about pipeline
        :return: dict with list of pipeline stages
        """
        pipeline_stages = []
        for i, stage in enumerate(in_list):
            pipeline_stages.append(Application.parse_pipeline_stage(stage))
        return {'stages': pipeline_stages}

    @staticmethod
    def parse_application(in_dict):
        """
        Deserializes received from yaml file dict into Application Definition

        :param in_dict: received from yaml file dict
        :raises ValueError: If wrong definitions are provided
        :return: Application Definition
        """
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
