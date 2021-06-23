import datetime
import json
import logging
import os
import tarfile
import time
import urllib.parse
from enum import Enum
from typing import Optional, Dict, List, Tuple, Iterator, Generator

import sseclient
import requests
from sseclient import Event
from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature
from hydro_serving_grpc.serving.manager.entities_pb2 import (
    ModelVersion as ModelVersionProto, DockerImage as DockerImageProto,
)
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.cluster import Cluster
from hydrosdk.signature import ModelSignature_to_signature_dict, validate_signature, signature_dict_to_ModelSignature
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, MetricModel, ThresholdCmpOp
from hydrosdk.exceptions import HydrosphereException, TimeoutException, BadResponseException, BadRequestException
from hydrosdk.utils import handle_request_error, read_in_chunks
from hydrosdk.builder import AbstractBuilder


def _upload_training_data(cluster: Cluster, modelversion_id: int, path: str) -> 'DataUploadResponse':
    """
    Upload training data to Hydrosphere

    :param cluster: Cluster instance
    :param modelversion_id: Id of the model version, for which to upload training data
    :param path: Path to the training data
    :raises BadResponseException: if request failed to process by Hydrosphere
    :return: DataUploadResponse obj
    """
    if path.startswith('s3://'):
        resp = _upload_s3_file(cluster, modelversion_id, path)
    else:
        resp = _upload_local_file(cluster, modelversion_id, path)
    if resp.ok:
        return DataUploadResponse(cluster, modelversion_id)
    raise BadResponseException(f'Failed to upload training data: {resp.text}')


def _upload_local_file(cluster: Cluster, modelversion_id: int, path: str, 
                       chunk_size=1024) -> requests.Response:
    """
    Internal method for uploading local training data to Hydrosphere.

    :param cluster: active cluster
    :param modelversion_id: modelversion_id for which to upload training data
    :param path: path to a local file
    :param chunk_size: chunk size to use for streaming
    """    
    gen = read_in_chunks(path, chunk_size)
    url = f'/monitoring/profiles/batch/{modelversion_id}'
    return cluster.request("POST", url, data=gen, stream=True)


def _upload_s3_file(cluster: Cluster, modelversion_id: int, path: str) -> requests.Response:
    """
    Internal method for submitting training data from S3 to Hydrosphere.

    :param cluster: Cluster instance
    :param url: url to which submit S3 path
    :param path: S3 path to training data
    """
    url = f'/monitoring/profiles/batch/{modelversion_id}/s3'
    return cluster.request("POST", url, json={"path": path})
    

def resolve_paths(path: str, payload: List[str], source_path: Optional[str] = None) -> Dict[str, str]:
    """
    Appends each element of payload to the path and makes {resolved_path: payload_element} dict

    :param path: absolute path
    :param payload: list of relative paths
    :return: dict with {resolved_path: payload_element}
    """
    source_path = source_path or "./"
    return {os.path.normpath(os.path.join(path, source_path, v)): v for v in payload}


class MonitoringConfiguration:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def to_dict(self):
        return {
            "batchSize": self.batch_size
        }
    

class ModelVersionBuilder(AbstractBuilder):
    """
    A local instance of the model yet to be uploaded to the cluster.
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

    :Example:

    Create a local instance of the model and upload it to the cluster.

    >>> from hydrosdk.cluster import Cluster
    >>> from hydrosdk.image import DockerImage
    >>> from hydrosdk.signature import SignatureBuilder, ProfilingType
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> builder = ModelVersionBuilder("my-model", "/path/to/my-model")
    >>> builder.with_runtime(DockerImage("hydrosphere/serving-runtime-python-3.7", tag="latest"))
    >>> builder.with_payload(["src/func_main.py", "requirements.txt"])
    >>> signature = SignatureBuilder("predict") \
            .with_input("x", int, "scalar", ProfilingType.NUMERICAL) \
            .with_output("y", float, "scalar", ProfilingType.NUMERICAL) \
            .build()
    >>> builder.with_signature(signature)
    >>> builder.with_install_command("pip install -r requirements.txt")
    >>> builder.with_training_data("training-data.csv")
    >>> model_version = builder.build(cluster)
    >>> model_version.lock_till_released(timeout=120)
    >>> data_upload_response = model_version.upload_training_data()
    """
    def __init__(self, name: str, path: str, source_path: Optional[str] = None) -> 'ModelVersionBuilder':
        """
        :param name: a name of the model
        :param path: a path to the root folder of the model
        """
        self.name = name
        self.path = path
        self.source_path = source_path or "./"
        self.runtime = None
        self.signature = None
        self.payload = {}
        self.metadata = {}
        self.install_command = None
        self.training_data = None
        self.monitoring_configuration = None

    def with_runtime(self, runtime: DockerImage) -> 'ModelVersionBuilder':
        """
        Specify runtime for the model version. 

        :param runtime: a docker image used to run model version's code.
        """
        if not isinstance(runtime, DockerImage):
            raise TypeError("runtime is not of type DockerImage")
        self.runtime = runtime
        return self
    
    def with_signature(self, signature: ModelSignature) -> 'ModelVersionBuilder':
        """
        Specify signature for the model version.

        :param signature: ModelSignature, which specifies the name of the inference function, 
                          its types and shapes of both inputs and outputs.
        """
        if not isinstance(signature, ModelSignature):
            raise TypeError("signature is not of type ModelSignature")
        validate_signature(signature)
        self.signature = signature
        return self
    
    def with_payload(self, payload: List[str]) -> 'ModelVersionBuilder':
        """
        Specify payload for the model version.

        :param payload: a list of paths to files (absolute or relative) with any 
                        additional resources that will be exported to the container.
        """
        self.payload = resolve_paths(path=self.path, source_path=self.source_path, payload=payload)
        return self
    
    def with_metadata(self, metadata: Dict[str, str]) -> 'ModelVersionBuilder':
        """
        Specify metadata for the model version.

        :param metadata: a metadata to describe the model version.
        """
        if not isinstance(metadata, dict):
            raise TypeError("metadata is not a dict")
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise TypeError(str(key) + " key from metadata is not a string")
            if not isinstance(value, str):
                raise TypeError(str(value) + " value from metadata is not a string")
        self.metadata = metadata
        return self
    
    def with_install_command(self, install_command: str) -> 'ModelVersionBuilder':
        """
        Specify install command for the model.

        :param install_command: a command to run within a runtime to prepare a model 
                                version environment.
        """
        if install_command and not isinstance(install_command, str):
            raise TypeError("install_command should be a string")
        self.install_command = install_command
        return self
    
    def with_training_data(self, training_data: str) -> 'ModelVersionBuilder':
        """
        Specify training data for the model version.
        
        :param training_data: path (absolute, relative or an S3 URI) to a csv file with 
                              the training data 
        """
        if training_data and not isinstance(training_data, str):
            raise TypeError("training_data should be a string")
        if not training_data.startswith("s3://"):
            if os.path.exists(training_data) and not os.path.isabs(training_data):
                training_data = os.path.join(self.source_path, training_data)
        self.training_data = training_data
        return self

    def with_monitoring_configuration(self, monitoring_configuration: MonitoringConfiguration) -> 'ModelVersionBuilder':
        """
        Specify monitoring configuration for the model version.

        :param monitoring_configuration: Specifies a configuration to be used when monitoring 
                                         the model version.
        """
        if monitoring_configuration and not isinstance(monitoring_configuration, MonitoringConfiguration):
            raise TypeError("monitoring_configuration should be of type MonitoringConfiguration")
        self.monitoring_configuration: MonitoringConfiguration = monitoring_configuration
        return self
    
    def __repr__(self):
        return f"ModelVersionBuilder(name={self.name})"

    def __create_tarball(self) -> str:
        logger = logging.getLogger("ModelDeploy")
        hs_folder = ".hs"
        os.makedirs(hs_folder, exist_ok=True)
        tarpath = os.path.join(hs_folder, f"{self.name}-{datetime.datetime.now()}")

        logger.debug("Creating payload tarball %s for %s model", tarpath, self.name)
        with tarfile.open(tarpath, "w:gz") as tar:
            for source, target in self.payload.items():
                logger.debug("Archiving %s as %s", source, target)
                tar.add(source, arcname=target)
        return tarpath
                
    def build(self, cluster: Cluster) -> 'ModelVersion':
        """
        Upload local model instance to the cluster.

        :param cluster: active cluster
        :return: ModelVersion object
        """
        if self.runtime is None:
            raise ValueError("runtime is not specified for the model")
        if self.signature is None:
            raise ValueError("signature is not specified for the model")

        meta = {
            "name": self.name,
            "runtime": self.runtime.dict(),
            "modelSignature": ModelSignature_to_signature_dict(self.signature),
            "installCommand": self.install_command,
            "metadata": self.metadata
        }
        
        if self.monitoring_configuration:
            meta.update({"monitoringConfiguration": self.monitoring_configuration.to_dict()})

        tarpath = self.__create_tarball()
        encoder = MultipartEncoder(
            fields={
                "payload": ("filename", open(tarpath, "rb")),
                "metadata": json.dumps(meta)
            }
        )
        
        resp = cluster.request("POST", "/api/v2/model/upload", 
            data=encoder, headers={'Content-Type': encoder.content_type})
        handle_request_error(
            resp, f"Failed to upload local model. {resp.status_code} {resp.text}")

        modelversion = ModelVersion._from_json(cluster, resp.json())
        modelversion.training_data = self.training_data
        return modelversion

    def build_external(self, cluster: Cluster) -> 'ModelVersion':
        """
        Create an external ModelVersion on the cluster. 

        :param cluster: active cluster
        :return: ModelVersion object
        """
        if self.signature is None:
            raise ValueError("signature is not specified for the model")
        model = {
            "name": self.name,
            "signature": ModelSignature_to_signature_dict(self.signature),
            "metadata": self.metadata
        }
        resp = cluster.request("POST", "/api/v2/externalmodel", json=model)
        handle_request_error(
            resp, f"Failed to create an external model. {resp.status_code} {resp.text}")
        return ModelVersion._from_json(cluster, resp.json())


class ModelVersionStatus(str, Enum):
    """
    Model building statuses.
    """
    Assembling = "Assembling"
    Released = "Released"
    Failed = "Failed"


class ModelVersion:
    """
    A Model, registered within the cluster. ModelVersion represents one of the Models' versions.
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

    :Example:

    List all modelversions, registered on the cluster.

    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> for model_version in ModelVersion.list(cluster):
            print(model_version)

    Upload training data for a model_version.

    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> model_version = ModelVersion.find(cluster, "my-model", 1)
    >>> model_version.lock_till_released()
    >>> model_version.training_data = "s3://my-bucket/path/to/training-data.csv"
    >>> data_upload_response = model_version.upload_traininf_data()

    Assign custom monitoring metrics for a model_version.

    >>> from hydrosdk.cluster import Cluster
    >>> from hydrosdk.monitoring import ThresholdCmpOp
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> model_version = ModelVersion.find(cluster, "my-model", 1)
    >>> modelversion_metric = ModelVersion.find(cluster, "my-model-metric", 1)
    >>> modelversion_metric = modmodelversion_metric.as_metric(1.4, ThresholdCmpOp.LESS_EQ)
    >>> model_version.assign_metrics([modelversion_metric])

    Check logs from a ModelVersion.

    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> model_version = ModelVersion.find(cluster, "my-model", 1)
    >>> for event in model_version.logs():
            print(event.data)
    """
    _BASE_URL = "/api/v2/model"

    @staticmethod
    def list(cluster: Cluster) -> List['ModelVersion']:
        """
        List all model versions on the cluster.

        :param cluster: active cluster
        :return: list of ModelVersions 
        """

        resp = cluster.request("GET", f"{ModelVersion._BASE_URL}/version")
        handle_request_error(
            resp, f"Failed to list model versions. {resp.status_code} {resp.text}")
        return [ModelVersion._from_json(cluster, modelversion_json)
                for modelversion_json in resp.json()]
    
    @staticmethod
    def find(cluster: Cluster, name: str, version: int) -> 'ModelVersion':
        """
        Find a ModelVersion on the cluster by model name and a version.

        :param cluster: active cluster
        :param name: name of the model
        :param version: version of the model
        :return: ModelVersion object
        """
        resp = cluster.request("GET", f"{ModelVersion._BASE_URL}/version/{name}/{version}")
        handle_request_error(
            resp, f"Failed to find model_version for name={name}, version={version}. {resp.status_code} {resp.text}")
        return ModelVersion._from_json(cluster, resp.json())

    @staticmethod
    def find_by_id(cluster: Cluster, id: int) -> 'ModelVersion':
        """
        Find a ModelVersion on the cluster by id.

        :param cluster: active cluster
        :param id: model version id
        :return: ModelVersion object
        """
        resp = cluster.request("GET", f"{ModelVersion._BASE_URL}/version")
        handle_request_error(
            resp, f"Failed to retrieve model_version by id={id}. {resp.status_code} {resp.text}")
        for modelversion_json in resp.json():
            if modelversion_json['id'] == id:
                return ModelVersion._from_json(cluster, modelversion_json)
        raise BadRequestException(f"Failed to retrieve model_version by id={id}.")
    
    @staticmethod
    def find_by_model_name(cluster: Cluster, model_name: str) -> list:
        """
        Find all model versions on the cluster filtered by model_name, sorted in ascending 
        order by version.

        :param cluster: active cluster
        :param model_name: a model name
        :return: list of ModelVersions with `model_name`
        """
        all_models = ModelVersion.list(cluster=cluster)
        modelversions_by_name = [model for model in all_models if model.name == model_name]
        sorted_by_version = sorted(modelversions_by_name, key=lambda model: model.version)
        return sorted_by_version

    @staticmethod
    def _from_json(cluster: Cluster, modelversion_json: dict) -> 'ModelVersion':
        """
        Internal method used for deserealization of a ModelVersion from a json object.

        :param cluster: active cluster
        :param modelversion_json: json response from the cluster
        :return: ModelVersion object
        """
        id = modelversion_json["id"]
        name = modelversion_json["model"]["name"]
        model_id = modelversion_json["model"]["id"]
        version = modelversion_json["modelVersion"]
        signature = signature_dict_to_ModelSignature(modelversion_json["modelSignature"])
        monitoring_configuration = MonitoringConfiguration(batch_size=modelversion_json["monitoringConfiguration"]["batchSize"])

        # external model deserialization handling
        is_external = modelversion_json.get('isExternal', False)
        if is_external:
            model_runtime = None
        else:
            model_runtime = DockerImage(**modelversion_json["runtime"])
        
        model_image = modelversion_json.get("image")
        if model_image is not None:
            model_image = DockerImage(**model_image)

        status = modelversion_json.get('status')
        if status:
            status = ModelVersionStatus[status]
        metadata = modelversion_json['metadata']

        return ModelVersion(
            id=id,
            model_id=model_id,
            is_external=is_external,
            name=name,
            version=version,
            signature=signature,
            runtime=model_runtime,
            image=model_image,
            cluster=cluster,
            status=status,
            metadata=metadata,
            monitoring_configuration=monitoring_configuration,
            created=modelversion_json.get("created"),
            finished=modelversion_json.get("finished"),
            applications=modelversion_json.get("applications"),
        )

    def lock_till_released(self, timeout: int = 120):
        """
        Lock till the model completes assembling.
        
        :param timeout: approximate value when to raise TimeoutException
        :raises ModelVersion.ReleaseFailed: if model failed to be released
        :raises TimeoutException: if timed out waiting for the model to be released
        """
        events_stream = self.cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        self.status = self.find_by_id(self.cluster, self.id).status
        if self.status is ModelVersionStatus.Released:
            return None
        if self.status is ModelVersionStatus.Failed:
            raise ModelVersion.ReleaseFailed("Model version build failed.")
        try:
            deadline_at = datetime.datetime.now().timestamp() + timeout
            for event in events_client.events():
                if datetime.datetime.now().timestamp() > deadline_at:
                    # last chance
                    self.status = self.find_by_id(self.cluster, self.id).status
                    if self.status is ModelVersionStatus.Released:
                        return None
                    raise TimeoutException("Timed out waiting for the model version to build.")
                if event.event == "ModelUpdate":
                    data = json.loads(event.data)
                    if data.get("id") == self.id:
                        self.status = ModelVersionStatus[data.get('status')]
                        if self.status is ModelVersionStatus.Released:
                            return None
                        raise ModelVersion.ReleaseFailed("Model version build failed.")
        finally:
            events_client.close()
    
    def build_logs(self) -> Iterator[Event]:
        """
        Sends request, saves and returns a build logs iterator.

        :return: Iterator over sseclient.Event
        """
        resp = self.cluster.request("GET", f"{self._BASE_URL}/version/{self.id}/logs", stream=True)
        return sseclient.SSEClient(resp).events()

    def upload_training_data(self) -> 'DataUploadResponse':
        """
        Uploads training data for a given model_version.

        :raises: ValueError if training data is not specified
        """
        if self.training_data is not None:
            return _upload_training_data(self.cluster, self.id, self.training_data)
        raise ValueError('Training data is not specified')

    def assign_metrics(self, metrics: List[MetricModel], wait: bool = True):
        """
        Adds metrics to the model.

        :param metrics: list of metrics
        :return: self
        """
        if wait:
            self.lock_till_released()
        
        for metric in metrics:
            modelversion = metric.modelversion
            if wait:
                modelversion.lock_till_released()

            config = MetricSpecConfig(
                modelversion_id=modelversion.id,
                threshold=metric.threshold,
                threshold_op=metric.comparator
            )
            MetricSpec.create(
                cluster=self.cluster,
                name=modelversion.name,
                modelversion_id=self.id,
                config=config
            )

    def to_proto(self) -> ModelVersionProto:
        """
        Convert ModelVersion object into a ModelVersion proto message.

        :return: ModelVersion proto message
        """
        return ModelVersionProto(
            _id=self.id,
            name=self.name,
            version=self.version,
            signature=self.signature,
            runtime=self.runtime,
            image=DockerImageProto(name=self.image.full, tag=self.image.tag),
            image_sha=self.image.sha256
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "modelVersion": self.version,
            "modelSignature": ModelSignature_to_signature_dict(self.signature),
            "runtime": self.runtime._asdict(),
            "image": self.image._asdict(),
            "status": self.status,
            "metadata": self.metadata,
            "monitoringConfiguration": self.monitoring_configuration.to_dict(),
            "isExternal": self.is_external,
            "created": self.created,
            "finished": self.finished,
            "applications": self.applications,
        }
    
    def as_metric(self, threshold: int, comparator: ThresholdCmpOp) -> MetricModel:
        """
        Converts model to MetricModel.

        :param threshold:
        :param comparator:
        :return: MetricModel
        """
        return MetricModel(modelversion=self, threshold=threshold, comparator=comparator)

    def __init__(self, cluster: Cluster, id: int, model_id: int, name: str, version: int, 
                 signature: ModelSignature, status: Optional[ModelVersionStatus], image: Optional[DockerImage], 
                 runtime: Optional[DockerImage], is_external: bool, 
                 metadata: Optional[Dict[str, str]] = None, install_command: Optional[str] = None, 
                 training_data: Optional[str] = None, monitoring_configuration: Optional[MonitoringConfiguration] = None, 
                 created: Optional[str] = None, finished: Optional[str] = None, applications: Optional[List[str]] = None):
        """
        :param cluster: active cluster
        :param id: id of the model_version assigned by the cluster
        :param model_id: id of the model assigned by the cluster
        :param name: a name of the model this ModelVersion belongs to
        :param version: a version of the model this ModelVersion belongs to
        :param signature: a ModelSignature which specifies the name of the inference function, 
                          as well as its types and shapes for both inputs and outputs
        :param status: a status of this ModelVersion, one of {Assembling, Released, Failed}
        :param image: DockerImage of a packed ModelVersion stored inside the cluster
        :param runtime: DockerImage of a runtime which was used to build an `image`
        :param is_external: indicates whether model is running outside the cluster
        :param metadata: metadata used for describing a ModelVersion
        :param install_command: a command which was run within a runtime to prepare a model_version
                                environment
        :param monitoring_configuration: specified monitoring configuration for the model
        :param created: specifies, when the model version was created
        :param finished: specified, when the model has finished building
        :param applications: specifies, which applications are using this model version
        """
        self.id = id
        self.model_id = model_id
        self.name = name
        self.runtime = runtime
        self.is_external = is_external
        self.signature = signature
        self.cluster = cluster
        self.version = version
        self.image = image
        self.status = status
        self.metadata = metadata
        self.install_command = install_command
        self.training_data = training_data
        self.monitoring_configuration = monitoring_configuration
        self.created = created
        self.finished = finished
        self.applications = applications or []

    def __repr__(self):
        return f"{self.name}:{self.version}"

    class ReleaseFailed(HydrosphereException):
        pass


class DataProfileStatus(Enum):
    Success = "Success"
    Failure = "Failure"
    Processing = "Processing"
    NotRegistered = "NotRegistered"
    Unknown = "Unknown"


# TODO: This should be rewritten to SSE, similar to lock_* methods from other entities, 
# like ModelVersion.lock_till_released or Application.lock_while_starting

class DataUploadResponse:
    """
    Handle processing status of the training data upload. 

    :Example:

    Wait till data processing gets finished.

    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> model_version = ModelVersion.find(cluster, "my-model", 1)
    >>> model_version.training_data = "s3://bucket/path/to/training-data.csv"
    >>> data_upload_response = model_version.upload_training_data()
    >>> try: 
            data_upload_response.wait(retry=3, sleep=30) 
        except TimeoutException as e:
            print(e)
        except DataUploadResponse.NotRegistered:
            id = data_upload_response.modelversion_id
            print(f"Unable to find training data for modelversion_id={id}")
        except DataUploadResponse.Failed:
            print(f"Failed to process training data")
    """
    def __init__(self, cluster: Cluster, modelversion_id: int) -> 'DataUploadResponse':
        self.cluster = cluster
        self.modelversion_id = modelversion_id

    @property
    def url(self) -> str:
        if not hasattr(self, '__url'):
            self.__url = f'/monitoring/profiles/batch/{self.modelversion_id}/status'
        return self.__url
    
    def get_status(self) -> DataProfileStatus:
        resp = self.cluster.request('GET', self.url)
        handle_request_error(
            resp, f"Failed to get status for modelversion_id={self.modelversion_id}. {resp.status_code} {resp.text}")
        return DataProfileStatus[resp.json().get('kind', "Unknown")]

    def wait(self, retry=12, sleep=30):
        """
        Wait till data processing gets finished. 
        
        :param retry: Number of retries before giving up waiting.
        :param sleep: Sleep interval between retries.
        """
        def tick(retry: int, sleep: int) -> Tuple[bool, int]:
            if retry == 0:
                return False, retry
            retry -= 1
            time.sleep(sleep)
            return True, retry
            
        while True:
            status = self.get_status()
            if status is DataProfileStatus.Success:
                break
            elif status is DataProfileStatus.Processing:
                can_be_polled, retry = tick(retry, sleep)
                if can_be_polled:
                    continue
                raise TimeoutException("Time out waiting for data processing to complete")
            elif status is DataProfileStatus.NotRegistered:
                raise DataUploadResponse.NotRegistered(f"Data profiling for modelversion_id={self.modelversion_id} has not been registered.")
            elif status is DataProfileStatus.Unknown:
                raise DataUploadResponse.Unknown(f"Data profiling for modelversion_id={self.modelversion_id} is unknown.")
            raise DataUploadResponse.Failed(f"Data profiling for modelversion_id={self.modelversion_id} failed.")

    class NotRegistered(HydrosphereException):
        pass

    class Failed(HydrosphereException):
        pass

    class Unknown(HydrosphereException):
        pass
