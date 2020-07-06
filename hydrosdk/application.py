from collections import namedtuple
from enum import Enum
from typing import List, Dict

from hydro_serving_grpc.contract import ModelSignature

from exceptions import ApplicationNotFoundError, ApplicationCreationError, ApplicationDeletionError, BadResponse
from hydrosdk.cluster import Cluster
from hydrosdk.contract import _signature_dict_to_ModelSignature
from hydrosdk.data.types import PredictorDT
from hydrosdk.modelversion import ModelVersion
from hydrosdk.predictor import PredictServiceClient, MonitorableImplementation

StreamingParams = namedtuple('StreamingParams', ['sourceTopic', 'destinationTopic'])
ModelVariant = namedtuple("ModelVariant", ["modelVersion", "weight"])


class ApplicationStatus(Enum):
    FAILED = 0
    ASSEMBLING = 1
    READY = 2


class Application:
    _BASE_URL = "/api/v2/application"

    @staticmethod
    def list_all(cluster) -> List['Application']:
        resp = cluster.request("GET", Application._BASE_URL)
        if resp.ok:
            applications = [Application.from_json(cluster=cluster, application_json=app_json) for app_json in resp.json()]
            return applications

        raise BadResponse(f"Failed to list all models. {resp.status_code} {resp.text}")

    @staticmethod
    def find(cluster, name) -> 'Application':
        resp = cluster.request("GET", f"{Application._BASE_URL}/{name}")
        if resp.ok:
            resp_json = resp.json()
            app = Application.from_json(cluster=cluster, application_json=resp_json)
            return app

        raise ApplicationNotFoundError(f"Failed to find by name. Name = {name}. {resp.status_code} {resp.text}")

    @staticmethod
    def delete(cluster, name):
        resp = cluster.request("DELETE", f"{Application._BASE_URL}/{name}")
        if resp.ok:
            return True
        raise ApplicationDeletionError(f"Failed to delete application. Name = {name}. {resp.status_code} {resp.text}")

    def __init__(self,
                 id: int,
                 cluster: Cluster,
                 name: str,
                 signature: ModelSignature,
                 execution_graph: 'ExecutionGraph',
                 status: ApplicationStatus,
                 kafka_streaming: List[StreamingParams],
                 metadata: Dict[str, str]):
        """
        Applications are used to combine your ModelVersions
        into a linear graph and deploy it into production, exposing
        HTTP and gRPC interfaces for consumers.

        Use ApplicationBuilder class to create a new Application in your Hydrosphere cluster
         or an Application.find method to get the existing Application.

        :param id: Unique Application ID
        :param cluster: Hydrosphere Cluster to which Application Belongs
        :param name: Application Name
        :param signature: Signature, specifying input and output fields names, dtypes and shapes
        :param execution_graph: Linear graph which specifies ExecutionStages which sequentially transform input
        :param status: Application Status - one of (Failed, Assembling, Ready)
        :param kafka_streaming: List of Kafka Parameters with input and output Kafka topics specified
        :param metadata: Metadata with string keys and string values.
        """

        self.cluster: Cluster = cluster
        self.id: int = id
        self.name: str = name
        self.status: ApplicationStatus = status
        self.execution_graph = execution_graph
        self.signature: ModelSignature = signature
        self.kafka_streaming = kafka_streaming
        self.metadata: Dict[str, str] = metadata

    def __str__(self):
        return f"Application {self.id} {self.name}"

    @staticmethod
    def from_json(cluster: Cluster, application_json: dict) -> 'Application':
        application_id = application_json.get("id")
        name = application_json["name"]
        execution_graph: ExecutionGraph = ExecutionGraph.from_json(cluster, application_json["executionGraph"])
        streaming_params: List[StreamingParams] = [StreamingParams(kafka_param["in-topic"], kafka_param["out-topic"]) for kafka_param in
                                                   application_json.get("kafkaStreaming")]
        app_metadata: Dict[str, str] = application_json["metadata"]
        app_signature: ModelSignature = _signature_dict_to_ModelSignature(data=application_json["signature"])
        app_status: ApplicationStatus = ApplicationStatus[application_json["status"].upper()]

        app = Application(id=application_id,
                          name=name,
                          execution_graph=execution_graph,
                          kafka_streaming=streaming_params,
                          metadata=app_metadata,
                          status=app_status,
                          cluster=cluster,
                          signature=app_signature)
        return app

    def predictor(self, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """
        Returns a predictor object which is used to transform your data
        into a proto message, pass it via gRPC to the cluster and decode
        the cluster output from proto to a dict with Python dtypes.
        :param return_type: Specifies into which format should predictor serialize model output.
         Numpy dtypes, Python dtypes or pd.DataFrame are supported.
        :return: PredictServiceClient with .predict() method which accepts your data
        """
        return PredictServiceClient(impl=MonitorableImplementation(channel=self.cluster.channel, target=self.name),
                                    signature=self.signature,
                                    return_type=return_type)

    def update_status(self) -> None:
        """Polls a cluster for a new Application status"""
        application = Application.find(cluster=self.cluster, name=self.name)
        self.status = application.status

    def delete(self, ):
        resp = self.cluster.request("DELETE", f"{Application._BASE_URL}/{self.name}")
        if resp.ok:
            return True
        raise ApplicationDeletionError(f"Failed to delete application. Name = {self.name}. {resp.status_code} {resp.text}")


class ApplicationBuilder:
    def __init__(self, cluster: Cluster, name: str):
        """
        ApplicationBuilder is used to create new Applications in your cluster
        :param cluster: Hydrosphere cluster where you want to create an Application
        :param name: Future Application name
        """
        self.cluster: Cluster = cluster
        self.name: str = name
        self.stages: List[ExecutionStage] = []
        self.metadata: Dict[str, str] = {}
        self.streaming_parameters: List[StreamingParams] = []

    def with_stage(self, stage: 'ExecutionStage') -> 'ApplicationBuilder':
        """
        Adds an ExecutionStage to your Application. See ExecutionStage for more information.
        :param stage:
        :return:
        """
        self.stages.append(stage)
        return self

    def with_metadata(self, key: str, value: str) -> 'ApplicationBuilder':
        """
        Adds a metadata value to your future Application
        :param key: String key under which `value` will be stored
        :param value: String value
        :return:
        """
        self.metadata[key] = value
        return self

    def with_kafka_params(self, source_topic: str, dest_topic: str) -> 'ApplicationBuilder':
        """
        Adds a kafka parameters to your Application
        :param source_topic: Source Kafka Topic
        :param dest_topic: Destination Kafka Topic
        :return:
        """
        self.streaming_parameters.append(StreamingParams(sourceTopic=source_topic, destinationTopic=dest_topic))
        return self

    def build(self) -> Application:
        """
        Creates an Application in your Hydrosphere cluster
        :return:
        """

        if not self.stages:
            raise ValueError("No execution stages were provided")

        execution_graph = ExecutionGraph(stages=self.stages)

        application_json = {"name": self.name,
                            "kafkaStreaming": [sp._asdict() for sp in self.streaming_parameters],
                            "executionGraph": execution_graph._asdict(),
                            "metadata": self.metadata}

        resp = self.cluster.request(method="POST", url=Application._BASE_URL, json=application_json)

        if resp.ok:
            resp_json = resp.json()
            app = Application.from_json(cluster=self.cluster, application_json=resp_json)
            return app

        raise ApplicationCreationError(f"Failed to create application {self.name}. {resp.status_code} {resp.text}")


class ExecutionGraph:
    def __init__(self, stages: List['ExecutionStage']):
        """
        ExecutionGraph is a representation of a linear graph which is used
        by Hydrosphere to create pipelines of ModelVersions. This linear graph
        consists of ExecutionStages following each other. Learn more about ExecutionStages.
        :param stages: List of ExecutionStages used to sequentially transform input.
        """
        self.stages = stages

    def _asdict(self):
        return {"stages": [s._asdict() for s in self.stages]}

    @staticmethod
    def from_json(cluster, ex_graph_dict: Dict):
        return ExecutionGraph([ExecutionStage.from_json(cluster, stage) for stage in ex_graph_dict['stages']])


class ExecutionStage:
    def __init__(self, model_variants: List[ModelVariant], signature: ModelSignature):
        """
        ExecutionStage is a single stage in a linear graph of ExecutionGraph. Each stage
        may contain from 1 to many different ModelVersions with the same signature. Every input
        requested routed to this stage will be automatically shadowed to all ModelVersions inside
        of it. Stage output response will be selected according to the relative weights associated with each version.
        :param model_variants: List of ModelVersions with corresponding weights
        :param signature: Signature specifying input and output field names, data types and shapes.
        """
        self.signature: ModelSignature = signature
        self.model_variants: List[ModelVariant] = model_variants

    def _asdict(self):
        return {"modelVariants": [{"modelVersionId": mv.modelVersion.id, "weight": mv.weight} for mv in self.model_variants]}

    @staticmethod
    def from_json(cluster, execution_stage_dict: Dict):
        execution_stage_signature = _signature_dict_to_ModelSignature(execution_stage_dict['signature'])

        model_variants = [ModelVariant(ModelVersion.from_json(cluster, mv['modelVersion']), mv['weight']) for mv in
                          execution_stage_dict['modelVariants']]

        return ExecutionStage(model_variants=model_variants,
                              signature=execution_stage_signature)


class ExecutionStageBuilder:
    def __init__(self):
        """
        Builder class to help building ExecutionStage
        """
        self.model_variants = []
        self.model_weights = []

    def with_model_variant(self, model_version: ModelVersion, weight) -> 'ExecutionStageBuilder':
        """
        Adds a ModelVersion with a weight to an ExecutionStage
        :param model_version: ModelVersion to which input requests will be shadowed
        :param weight: Weight which affects the chance of choosing a model output as an
         output of an ExecutionStage
        :return:
        """
        self.model_variants.append(ModelVariant(model_version, weight))
        return self

    def build(self) -> 'ExecutionStage':
        """
        Verifies that all ModelVersions inside an ExecutionStage have the same signature
         and finally creates an ExecutionStage
        :return:
        """
        common_signature = self.model_variants[0].modelVersion.contract.predict
        model_variant_signatures = [mv.modelVersion.contract.predict for mv in self.model_variants]

        if not all(common_signature == mv_signature for mv_signature in model_variant_signatures):
            raise ValueError("All model variants inside the same stage must have the same signature")

        return ExecutionStage(model_variants=self.model_variants, signature=common_signature)
