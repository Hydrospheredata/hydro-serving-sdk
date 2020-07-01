from collections import namedtuple
from enum import Enum
from typing import List, Optional

from hydro_serving_grpc.contract import ModelSignature

from hydrosdk.cluster import Cluster
from hydrosdk.contract import _signature_dict_to_ModelSignature
from hydrosdk.data.types import PredictorDT
from hydrosdk.predictor import PredictServiceClient, MonitorableImplementation
from hydrosdk.exceptions import RequestsErrorHandler

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
    FAILED = 0
    ASSEMBLING = 1
    READY = 2


class Application(RequestsErrorHandler):
    """
    An application is a publicly available endpoint to reach your models (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications)
    """

    @classmethod
    def list_all(cls, cluster: Cluster) -> List['Application']:
        """
        Lists all available applications from server

        :param cluster: active cluster
        :raises Exception: If response from server is not 200
        :return: deserialized list of application objects
        """
        resp = cluster.request("GET", "/api/v2/application")
        cls.handle_request_error(
            resp, f"Failed to list all applications. {resp.status_code} {resp.text}")
        applications = [Application.app_json_to_app_obj(cluster, app_json) 
                        for app_json in resp.json()]
        return applications

    @classmethod
    def find_by_name(cls, cluster: Cluster, app_name: str) -> 'Application':
        """
        By the *app_name* searches for the Application

        :param cluster: active cluster
        :raises Exception: If response from server is not 200
        :return: deserialized Application object
        """
        resp = cluster.request("GET", "/api/v2/application/{}".format(app_name))
        cls.handle_request_error(
            resp, f"Failed to find application by name={app_name}. {resp.status_code} {resp.text}")
        return Application.app_json_to_app_obj(cluster, resp.json())

    @classmethod
    def delete(cls, cluster: Cluster, app_name: str) -> dict:
        """
        By the *app_name* deletes Application

        :param cluster: active cluster
        :raises Exception: If response from server is not 200
        :return: response from the server
        """
        resp = cluster.request("DELETE", "/api/v2/application/{}".format(app_name))
        cls.handle_request_error(
            resp, f"Failed to delete application for name={app_name}. {resp.status_code} {res.text}")
        return resp.json()

    @classmethod
    def create(cls, cluster: Cluster, application: dict):
        """
        By the *app_name* searches for the Application

        :param cluster: active cluster
        :param application: dict with necessary to create application fields
        :raises Exception: If response from server is not 200
        :return: deserialized Application object
        """
        resp = cluster.request(method="POST", url="/api/v2/application", json=application)
        cls.handle_request_error(
            resp, f"Failed to create an application. {resp.status_code} {res.text}")
        return Application.app_json_to_app_obj(cluster=cluster, application_json=resp.json())

    @staticmethod
    def app_json_to_app_obj(cluster: Cluster, application_json: dict) -> 'Application':
        """
        Deserializes json into Application
        :param cluster: active cluster
        :param application_json: input json with application object fields
        :return Application : application object
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
    def parse_streaming_params(in_list: List[dict]) -> list:
        """
        Deserializes from input list StreamingParams

        :param in_list: input list of dicts
        :return: list ofr StreamingParams
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
        Deserializes received file dict into Application Definition

        :param in_dict: received file dict
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

    def predictor(self, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        self.impl = MonitorableImplementation(channel=self.cluster.channel, target=self.name)
        self.predictor_return_type = return_type

        return PredictServiceClient(impl=self.impl, signature=self.signature,
                                    return_type=self.predictor_return_type)

    def update_status(self) -> None:
        """
        Setter method that updates application status
        :return: None
        """
        self.status = self.find_by_name(cluster=self.cluster, app_name=self.name).status

    def __init__(self, cluster: Cluster, name: str, execution_graph: dict, status: int, signature: ModelSignature, 
                 kafka_streaming: Optional[dict] = None, metadata: Optional[dict] = None):
        self.name = name
        self.execution_graph = execution_graph
        self.kafka_streaming = kafka_streaming
        self.metadata = metadata
        self.status = status
        self.signature = signature
        self.cluster = cluster

    class BadRequest(Exception):
        """
        Used for cases, when cluster returns 4xx response on user request.
        """
        pass
