from enum import Enum
from typing import List, Optional, Dict
import json
import datetime

import sseclient
from google.protobuf.json_format import MessageToDict

from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature

from hydrosdk.cluster import Cluster
from hydrosdk.signature import signature_dict_to_ModelSignature
from hydrosdk.data.types import PredictorDT
from hydrosdk.deployment_configuration import DeploymentConfiguration
from hydrosdk.modelversion import ModelVersion
from hydrosdk.predictor import PredictServiceClient, MonitorableApplicationPredictionService
from hydrosdk.utils import handle_request_error
from hydrosdk.exceptions import TimeoutException
from pydantic import BaseModel

class StreamingParams(BaseModel):
    sourceTopic: str
    destinationTopic: str

class ModelVariant(BaseModel):
    modelVersionId: int
    weight: int
    deploymentConfigurationName: Optional[str]
    servableName: Optional[str]


class ApplicationStatus(Enum):
    FAILED = 0
    ASSEMBLING = 1
    READY = 2


class Application:
    """
    Applications are used to combine your ModelVersions into a linear graph and deploy 
    it into production, exposing HTTP and gRPC interfaces for consumers.

    Use ApplicationBuilder class to create a new Application in your Hydrosphere cluster
    or an Application.find method to get the existing Application.

    :Example:

    List all applications created on the cluster.

    >>> from hydrosdk.cluster import Cluster 
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> apps = Application.list(cluster)
    >>> for app in apps: 
    >>>     print(app)

    Find an application by name and perform a prediction from it.

    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint", "grpc-cluster-endpoint")  # important to use a gRPC endpoint 
    >>> app = Application.find(cluster, "my-application")
    >>> pred = app.predictor()
    >>> resp = pred.predict({"my-input": 1})
    """
    _BASE_URL = "/api/v2/application"

    @staticmethod
    def list(cluster: Cluster) -> List['Application']:
        """
        List all available applications from the cluster.

        :param cluster: active cluster
        :return: deserialized list of application objects
        """
        resp = cluster.request("GET", Application._BASE_URL)
        handle_request_error(
            resp, f"Failed to list all applications. {resp.status_code} {resp.text}")
        applications = [Application._from_json(cluster, app_json)
                        for app_json in resp.json()]
        return applications

    @staticmethod
    def find(cluster: Cluster, application_name: str) -> 'Application':
        """
        Search for an application by name. 

        :param cluster: active cluster
        :param application_name: application name
        :return: deserialized application object
        """
        resp = cluster.request("GET", f"{Application._BASE_URL}/{application_name}")
        handle_request_error(
            resp, f"Failed to find an application by name={application_name}. {resp.status_code} {resp.text}")
        return Application._from_json(cluster, resp.json())

    @staticmethod
    def delete(cluster: Cluster, application_name: str) -> dict:
        """
        Delete an application by name.

        :param cluster: active cluster
        :param application_name: application name
        :return: response from the cluster
        """
        resp = cluster.request("DELETE", f"{Application._BASE_URL}/{application_name}")
        handle_request_error(
            resp, f"Failed to delete application for name={application_name}. {resp.status_code} {resp.text}")
        return resp.json()

    @staticmethod
    def _from_json(cluster: Cluster, application_json: dict) -> 'Application':
        """
        Deserialize json into application object. 

        :param cluster: active cluster
        :param application_json: input json with application object fields
        :return: application object
        """
        id_ = application_json.get("id")
        name = application_json.get("name")
        execution_graph = ExecutionGraph._from_json(cluster, application_json.get("executionGraph"))
        kafka_streaming = [StreamingParams(sourceTopic=kafka_param["in-topic"], destinationTopic=kafka_param["out-topic"])
                               for kafka_param in application_json.get("kafkaStreaming")]
        metadata = application_json.get("metadata")
        message = application_json.get("message")
        signature = signature_dict_to_ModelSignature(data=application_json.get("signature"))
        status = ApplicationStatus[application_json.get("status").upper()]

        app = Application(cluster=cluster,
                          id=id_,
                          name=name,
                          execution_graph=execution_graph,
                          status=status,
                          signature=signature,
                          kafka_streaming=kafka_streaming,
                          metadata=metadata,
                          message=message)
        return app

    def lock_while_starting(self, timeout: int = 120) -> 'Application':
        """ Wait for an application to become ready. """
        events_stream = self.cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        self.status = self.find(self.cluster, self.name).status
        if self.status is ApplicationStatus.READY: 
            return self
        if self.status is ApplicationStatus.FAILED:
            raise ValueError(f'Application initialization failed {self.message}')
        try:
            deadline_at = datetime.datetime.now().timestamp() + timeout
            for event in events_client.events():
                if datetime.datetime.now().timestamp() > deadline_at:
                    raise TimeoutException('Time out waiting for an application to become available')
                if event.event == "ApplicationUpdate":
                    data = json.loads(event.data)
                    print(data)
                    if data.get("name") == self.name:
                        self.status = ApplicationStatus[data.get("status").upper()]
                        if self.status is ApplicationStatus.READY:
                            return self
                        elif self.status is ApplicationStatus.FAILED:
                            raise ValueError('Application initialization failed')
        finally:
            events_client.close()

    def predictor(self, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """
        Return a predictor object which is used to transform your data
        into a proto message, pass it via gRPC to the cluster and decode
        the cluster output from proto to a dict with Python dtypes.

        :param return_type: Specifies into which format should predictor serialize model output.
                            Numpy dtypes, Python dtypes or pd.DataFrame are supported.
        :return: PredictServiceClient with .predict() method which accepts your data
        """
        impl = MonitorableApplicationPredictionService(channel=self.cluster.channel, target=self.name)
        return PredictServiceClient(impl=impl, signature=self.signature, return_type=return_type)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "signature": MessageToDict(self.signature),
            "execution_graph": self.execution_graph.to_dict(),
            "metadata": self.metadata,
            "message": self.message,
            "status": self.status.name,
        }
    
    def __init__(self, cluster: Cluster, id: int, name: str, execution_graph: 'ExecutionGraph',
                 status: ApplicationStatus, signature: ModelSignature,
                 kafka_streaming: List[StreamingParams],
                 metadata: Optional[Dict[str, str]] = None,
                 message: Optional[str] = None) -> 'Application':
        """
        :param cluster: active cluster
        :param id: unique application ID
        :param name: application Name
        :param signature: signature, specifying input and output fields names, dtypes and shapes
        :param execution_graph: linear graph which specifies ExecutionStages which sequentially transform input
        :param status: Application Status
        :param kafka_streaming: list of Kafka parameters with input and output Kafka topics specified
        :param metadata: metadata with string keys and string values.
        :param message: possible error message from the cluster
        """
        self.id = id
        self.name = name
        self.execution_graph = execution_graph
        self.kafka_streaming = kafka_streaming
        self.metadata = metadata
        self.status = status
        self.signature = signature
        self.cluster = cluster
        self.message = message

    def __str__(self):
        return f"Application {self.id} {self.name}"


class ApplicationBuilder:
    """
    ApplicationBuilder is used to create new Applications in your cluster.

    :Example:
    
    Create an application from existing modelversions.

    >>> from hydrosdk import Cluster, ModelVersion
    >>> cluster = Cluster('http-cluster-endpoint')
    >>> mv1 = ModelVersion.find(cluster, "my-model", 1)
    >>> mv2 = ModelVersion.find(cluster, "my-model", 2)
    >>> stage = ExecutionStageBuilder() \
                    .with_model_variant(mv1, 50) \
                    .with_model_variant(mv2, 50) \ 
                    .build()
    >>> app = ApplicationBuilder("my-application-ab-test") \
                .with_stage(stage) \
                .build(cluster)
    """

    def __init__(self, name: str) -> 'ApplicationBuilder':
        """
        :param cluster: Hydrosphere cluster where you want to create an Application
        :param name: Future Application name
        """
        self.name = name
        self.stages = []
        self.metadata = {}
        self.streaming_parameters = []

    def with_stage(self, stage: 'ExecutionStage') -> 'ApplicationBuilder':
        """
        Add an ExecutionStage to your Application. See ExecutionStage for more information.

        :param stage:
        :return:
        """
        self.stages.append(stage)
        return self

    def with_metadata(self, key: str, value: str) -> 'ApplicationBuilder':
        """
        Add a metadata value to your future Application.

        :param key: string key under which `value` will be stored
        :param value: string value
        :return:
        """
        self.metadata[key] = value
        return self

    def with_metadatas(self, metadata: Dict[str, str]) -> 'ApplicationBuilder':
        """
        Add a metadata to your future Application.

        :param metadata: a dict containing metadata for your application
        :return:
        """
        self.metadata.update(metadata)
        return self

    def with_kafka_params(self, source_topic: str, dest_topic: str) -> 'ApplicationBuilder':
        """
        Add a kafka parameters to your Application.

        :param source_topic: source Kafka topic
        :param dest_topic: destination Kafka topic
        :return:
        """
        params = StreamingParams(sourceTopic=source_topic, destinationTopic=dest_topic)
        self.streaming_parameters.append(params)
        return self

    def build(self, cluster: Cluster) -> Application:
        """
        Create an Application in your Hydrosphere cluster.

        :return: Application object
        """
        if not self.stages:
            raise ValueError("No execution stages were provided")

        execution_graph = ExecutionGraph(stages=self.stages)
        application_json = {"name": self.name,
                            "kafkaStreaming": [sp.to_dict() for sp in self.streaming_parameters],
                            "executionGraph": execution_graph.to_dict(),
                            "metadata": self.metadata}

        resp = cluster.request("POST", Application._BASE_URL, json=application_json)
        handle_request_error(
            resp, f"Failed to create an application {self.name}. {resp.status_code} {resp.text}")
        return Application._from_json(cluster, resp.json())


class ExecutionGraph:
    def __init__(self, stages: List['ExecutionStage']) -> 'ExecutionGraph':
        """
        ExecutionGraph is a representation of a linear graph which is used
        by Hydrosphere to create pipelines of ModelVersions. This linear graph
        consists of ExecutionStages following each other. See ExecutionStage to learn more.

        :param stages: list of ExecutionStages used to sequentially transform input.
        """
        self.stages = stages

    def to_dict(self) -> Dict[str, List]:
        return {"stages": [s.to_dict() for s in self.stages]}

    @staticmethod
    def _from_json(cluster: Cluster, ex_graph_dict: Dict) -> 'ExecutionGraph':
        return ExecutionGraph([ExecutionStage._from_json(cluster, stage)
                               for stage in ex_graph_dict['stages']])


class ExecutionStage:
    def __init__(self, model_variants: List[ModelVariant], signature: Optional[ModelSignature]) -> 'ExecutionStage':
        """
        ExecutionStage is a single stage in a linear graph of ExecutionGraph. Each stage
        may contain from 1 to many different ModelVersions with the same signature. Every input
        requested routed to this stage will be automatically shadowed to all ModelVersions inside
        of it. Stage output response will be selected according to the relative weights associated 
        with each version.

        :param model_variants: list of ModelVersions with corresponding weights
        :param signature: signature specifying input and output field names, data types and shapes.
        """
        self.signature = signature
        self.model_variants = model_variants

    def to_dict(self) -> Dict[str, List[Dict[str, int]]]:
        model_variants = []
        for model_variant in self.model_variants:
            dict_repr = {
                'modelVersionId': model_variant.modelVersionId,
                'weight': model_variant.weight,
                'deploymentConfigName': model_variant.deploymentConfigurationName,
            }
            model_variants.append(dict_repr)
        return {"modelVariants": model_variants}

    @staticmethod
    def _from_json(cluster: Cluster, execution_stage_dict: Dict) -> 'ExecutionStage':
        execution_stage_signature = signature_dict_to_ModelSignature(execution_stage_dict['signature'])
        model_variants = [ModelVariant.parse_obj(mv) for mv in execution_stage_dict['modelVariants']]
        return ExecutionStage(model_variants=model_variants, signature=execution_stage_signature)


class ExecutionStageBuilder:
    def __init__(self) -> 'ExecutionStageBuilder':
        """
        Builder class to help building ExecutionStage.
        """
        self.model_variants = []

    def __validate(self):
        """
        Validate the stage for correctness.
        """
        if len(self.model_variants) == 0:
            raise ValueError("At least one model variant should be specified.")

        if sum(variant.weight for variant in self.model_variants) != 100:
            raise ValueError("All model variants' weights inside the same stage must sum up to 100")

    def with_model_variant(self, model_version: ModelVersion,
                           weight: int,
                           deployment_configuration: Optional[DeploymentConfiguration] = None) -> 'ExecutionStageBuilder':
        """
        Add a ModelVersion with a weight to an ExecutionStage.
        
        :param model_version: ModelVersion to which input requests will be shadowed
        :param weight: Weight which affects the chance of choosing a model output as an
         output of an ExecutionStage
        :param deployment_configuration: K8s Deployment Configuration of this Model Variant
        :return:
        """
        dc_name = None
        if deployment_configuration is not None:
            dc_name = deployment_configuration.name
        mv = ModelVariant(
            modelVersionId=model_version.id,
            weight=weight,
            deploymentConfigurationName=dc_name,
            servableName=None
        )
        self.model_variants.append(mv)
        return self

    def build(self) -> 'ExecutionStage':
        """
        Verify that all ModelVersions inside an ExecutionStage have the same signature
        and finally creates an ExecutionStage.

        :return:
        """
        self.__validate()
        return ExecutionStage(model_variants=self.model_variants, signature=None)
