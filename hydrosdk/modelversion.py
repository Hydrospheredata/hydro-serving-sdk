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
import yaml
from sseclient import Event
from hydro_serving_grpc.contract import ModelContract
from hydro_serving_grpc.manager import ModelVersion as grpc_ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.cluster import Cluster
from hydrosdk.contract import ModelContract_to_contract_dict, contract_dict_to_ModelContract, \
    contract_yaml_to_ModelContract
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, MetricModel


def _upload_training_data(cluster: Cluster, model_version_id: int, path: str) -> 'DataUploadResponse':
    """
    Upload training data to Hydrosphere

    :param cluster: Cluster instance
    :param model_version_id: Id of the model version, for which to upload training data
    :param path: Path to the training data
    :raises Cluster.BadResponse: if request failed to process by Hydrosphere
    :return: DataUploadResponse obj
    """
    if path.startswith('s3://'):
        response = _upload_s3_file(cluster, model_version_id, path)
    else:
        response = _upload_local_file(cluster, model_version_id, path)
    if response.ok:
        return DataUploadResponse(cluster, model_version_id)
    raise Cluster.BadResponse('Failed to upload training data')


def _upload_local_file(cluster: Cluster, model_version_id: int, path: str, 
                       chunk_size=1024) -> requests.Response:
    """
    Internal method for uploading local training data to Hydrosphere.

    :param cluster: Cluster instance
    :param url: url to which upload the file
    :param path: path to a local file
    :param chunk_size: chunk size to use for streaming
    """
    def read_in_chunks(filename: str, chunk_size: int) -> Generator[bytes, None, None]:
        """ Generator to read a file peace by peace. """
        with open(filename, "rb") as file:
            while True:
                data = file.read(chunk_size)
                if not data:
                    break
                yield data
    
    gen = read_in_chunks(path, chunk_size)
    url = '/monitoring/profiles/batch/{}'.format(model_version_id)
    return cluster.request("POST", url, data=gen, stream=True)


def _upload_s3_file(cluster: Cluster, model_version_id: int, path: str) -> requests.Response:
    """
    Internal method for submitting training data from S3 to Hydrosphere.

    :param cluster: Cluster instance
    :param url: url to which submit S3 path
    :param path: S3 path to training data
    """
    url = '/monitoring/profiles/batch/{}/s3'.format(model_version_id)
    return cluster.request("POST", url, json={"path": path})
    

def resolve_paths(path: str, payload: List[str]) -> Dict[str, str]:
    """
    Appends each element of payload to the path and makes {resolved_path: payload_element} dict

    :param path: absolute path
    :param payload: list of relative paths
    :return: dict with {resolved_path: payload_element}
    """
    return {os.path.normpath(os.path.join(path, v)): v for v in payload}


class LocalModel:
    """
    Local Model
    A model is a machine learning model or a processing function that consumes provided inputs
    and produces predictions or transformations

    https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models
    """

    def __init__(self, name, runtime, payload, path, contract=None, metadata=None, install_command=None,
                 training_data=None):
        super().__init__()

        if not isinstance(name, str):
            raise TypeError("name is not a string")
        self.name = name
        if not isinstance(runtime, DockerImage):
            raise TypeError("runtime is not a DockerImage")
        self.runtime = runtime
        if contract:
            if not isinstance(contract, ModelContract):
                raise TypeError("contract is not a ModelContract")

            # TODO: move out contract validation
            # HYD-171
            if not contract.HasField("predict"):
                raise ValueError("Creating model without contract.predict is not allowed")
            if not contract.predict.signature_name:
                raise ValueError("Creating model without contract.predict.signature_name is not allowed")
            for model_field in contract.predict.inputs:
                if model_field.dtype == 0:
                    raise ValueError("Creating model with invalid dtype in contract-input is not allowed")
            for model_field in contract.predict.outputs:
                if model_field.dtype == 0:
                    raise ValueError("Creating model with invalid dtype in contract-output is not allowed")

        self.contract = contract

        if isinstance(payload, list):
            self.payload = resolve_paths(path=path, payload=payload)
            self.path = path
        if isinstance(payload, dict):
            self.payload = payload

        if metadata:
            if not isinstance(metadata, dict):
                raise TypeError("metadata is not a dict")

            for key, value in metadata.items():
                if not isinstance(key, str):
                    raise TypeError(str(key) + " key from metadata is not a string")
                if not isinstance(value, str):
                    raise TypeError(str(value) + " value from metadata is not a string")

        self.metadata = metadata

        if install_command and not isinstance(install_command, str):
            raise TypeError("install-command should be a string")
        self.install_command = install_command

        if training_data and not isinstance(training_data, str):
            raise TypeError("training-data should be a string")
        self.training_data = training_data

    def __repr__(self):
        return f"LocalModel {self.name}"

    def upload(self, cluster: Cluster) -> 'ModelVersion':
        """
        Direct implementation of uploading one model to the server. For internal usage

        :param cluster: active cluster
        :raises ValueError: If server returned not 200
        :return: ModelUploadResponse obj
        """
        logger = logging.getLogger("ModelDeploy")
        hs_folder = ".hs"
        os.makedirs(hs_folder, exist_ok=True)
        tarpath = os.path.join(hs_folder, f"{self.name}-{datetime.datetime.now()}")

        logger.debug("Creating payload tarball %s for %s model", tarpath, self.name)
        with tarfile.open(tarpath, "w:gz") as tar:
            for source, target in self.payload.items():
                logger.debug("Archiving %s as %s", source, target)
                tar.add(source, arcname=target)

        meta = {
            "name": self.name,
            "runtime": {
                "name": self.runtime.name,
                "tag": self.runtime.tag,
                "sha256": self.runtime.sha256
            },
            "contract": ModelContract_to_contract_dict(self.contract),
            "installCommand": self.install_command,
            "metadata": self.metadata
        }

        encoder = MultipartEncoder(
            fields={
                "payload": ("filename", open(tarpath, "rb")),
                "metadata": json.dumps(meta)
            }
        )
        resp = cluster.request("POST", "/api/v2/model/upload",
                               data=encoder,
                               headers={'Content-Type': encoder.content_type})
        if resp.ok:
            modelversion = ModelVersion.from_json(
                cluster=cluster, model_version=resp.json())
            modelversion.training_data = self.training_data
            return modelversion
        elif 400 <= resp.status_code < 500:
            raise LocalModel.BadRequest(
                f"Client error during model upload ({resp.status_code}): {resp.text}")
        elif 500 <= resp.status_code < 600:
            raise Cluster.BadResponse(
                f"Server error during model upload ({resp.status_code}): {resp.text}")
        else:
            raise Cluster.UnknownException(
                f"Unexpected status code during model upload ({resp.status_code}): {resp.text}")

    class BadRequest(Exception): 
        pass


class ModelVersionStatus(Enum):
    """
    Model building statuses
    """
    Assembling = "Assembling"
    Released = "Released"
    Failed = "Failed"

    @classmethod
    def is_assembling(cls, status: str) -> bool:
        if status == cls.Assembling.value:
            return True
        return False

    @classmethod
    def is_released(cls, status: str) -> bool:
        if status == cls.Released.value:
            return True
        return False


class ModelVersion:
    """
    Model (A model is a machine learning model or a processing function that consumes provided inputs
    and produces predictions or transformations
    https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)
    ModelVersion represents one of the Model's versions.
    """
    BASE_URL = "/api/v2/model"

    @staticmethod
    def find(cluster: Cluster, name: str, version: int) -> 'ModelVersion':
        """
        Finds a model on server by name and version (not ModelVersion!)

        :param cluster: active cluster
        :param name: model name
        :param version: version
        :raises Exception: if server returned not 200
        :return: ModelVersion obj
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version/{}/{}".format(name, version))

        if resp.ok:
            model_json = resp.json()
            return ModelVersion.from_json(cluster=cluster, model_version=model_json)

        else:
            raise ModelVersion.NotFound(
                f"Failed to find ModelVersion for name={name}, version={version}. {resp.status_code} {resp.text}")

    @staticmethod
    def find_by_id(cluster: Cluster, id_) -> 'ModelVersion':
        """
        Finds a modelversion on server by id.

        :param cluster: active cluster
        :param id_: model version id
        :raises ModelVersion.NotFound: if failed to return modelversion 
        :return: ModelVersion obj
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version")

        if resp.ok:
            for model_json in resp.json():
                if model_json['id'] == id_:
                    return ModelVersion.from_json(cluster=cluster, model_version=model_json)

        raise ModelVersion.NotFound(
            f"Failed to find_by_id ModelVersion for model_version_id={id_}. {resp.status_code} {resp.text}")

    @staticmethod
    def from_json(cluster: Cluster, model_version: dict) -> 'ModelVersion':
        """
        Internal method used for deserealization of a Model from json object.

        :param cluster: active cluster
        :param model_version: json response from the server
        :return: instance of ModelVersion
        """
        id_ = model_version["id"]
        name = model_version["model"]["name"]
        model_id = model_version["model"]["id"]
        version = model_version["modelVersion"]
        model_contract = contract_dict_to_ModelContract(model_version["modelContract"])

        # external model deserialization handling
        is_external = model_version.get('isExternal', False)
        if is_external:
            model_runtime = None
        else:
            model_runtime = DockerImage(model_version["runtime"]["name"], model_version["runtime"]["tag"],
                                        model_version["runtime"].get("sha256"))

        model_image = model_version.get("image")
        model_cluster = cluster

        status = model_version.get('status')
        if status:
            status = ModelVersionStatus[status]
        metadata = model_version['metadata']

        return ModelVersion(
            id=id_,
            model_id=model_id,
            is_external=is_external,
            name=name,
            version=version,
            contract=model_contract,
            runtime=model_runtime,
            image=model_image,
            cluster=model_cluster,
            status=status,
            metadata=metadata,
        )

    @staticmethod
    def create(cluster: Cluster, name: str, contract: ModelContract, 
               metadata: Optional[dict] = None, training_data: Optional[str] = None) -> 'ModelVersion':
        """
        Creates an external model version on the server. 

        :param cluster: active cluster
        :param name: name of model
        :param contract: contract of the model
        :param metadata: metadata for the model
        :raises ModelVersion.BadRequest: if a model registration request was invalid
        :raises Cluster.BadResponse: if the server failed to register an external model
        :raises Cluster.UnknownException: if received unknown exception from the server
        :return: instance of ModelVersion
        """
        model = {
            "name": name,
            "contract": ModelContract_to_contract_dict(contract),
            "metadata": metadata
        }

        resp = cluster.request(method="POST", url="/api/v2/externalmodel", json=model)
        if resp.ok:
            resp_json = resp.json()
            modelversion = ModelVersion.from_json(cluster=cluster, model_version=resp_json)
            modelversion.training_data = training_data
            return modelversion
        elif 400 <= resp.status_code < 500:
            raise ModelVersion.BadRequest(
                f"Failed to create_externalmodel. {resp.status_code} {resp.text}")
        elif 500 <= resp.status_code < 600:
            raise Cluster.BadResponse(
                f"Failed to create_externalmodel. {resp.status_code} {resp.text}")
        else:
            raise Cluster.UnknownException("Received unknown exception. {}".format(resp.text))

    @staticmethod
    def delete_by_model_id(cluster: Cluster, model_id: int) -> dict:
        """
        Deletes modelversion by model id.

        :param cluster: active cluster
        :param model_id: model version id
        :return: if 200, json. Otherwise None
        """
        res = cluster.request("DELETE", ModelVersion.BASE_URL + "/{}".format(model_id))
        if res.ok:
            return res.json()
        raise ModelVersion.BadRequest(
            f"Failed to delete_by_model_id for model_id={model_id}. {res.status_code} {res.text}")

    @staticmethod
    def list_model_versions(cluster) -> List['ModelVersion']:
        """
        List all model versions on server

        :param cluster: active cluster
        :return: list of modelversions
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version")

        if resp.ok:
            model_versions_json = resp.json()
            model_versions = [ModelVersion.from_json(cluster=cluster, model_version=model_version_json)
                              for model_version_json in model_versions_json]
            return model_versions
        raise Cluster.BadResponse(
            f"Failed to list_model_versions. {resp.status_code} {resp.text}")

    @staticmethod
    def list_modelversions_by_model_name(cluster: Cluster, model_name: str) -> list:
        """
        List all model versions on server filtered by model_name, sorted in ascending order by version

        :param cluster: active cluster
        :param model_name: model name
        :return: list of Models versions for provided model name
        """
        all_models = ModelVersion.list_model_versions(cluster=cluster)
        modelversions_by_name = [model for model in all_models if model.name == model_name]
        sorted_by_version = sorted(modelversions_by_name, key=lambda model: model.version)
        return sorted_by_version
    
    def _poll_status(self):
        """Updates status of the current modelversion."""
        self.status = self.find(self.cluster, self.name, self.version).status
    
    def lock_till_released(self) -> bool:
        """
        Waits till the model completes assembling.

        :return: True if model has been released successfully, False otherwise
        """
        events_steam = cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        self._poll_status()
        if not ModelVersionStatus.is_assembling(self.status) \
                and ModelVersionStatus.is_released(self.status): 
            return True
        try:
            for event in events_client.events():
                if event.event == "ModelUpdate":
                    data = json.loads(event.data)
                    if data.get("id") == self.id:
                        return ModelVersionStatus.is_released(data.get("status"))
        finally:
            events_client.close()
    
    def logs(self) -> Iterator[Event]:
        """
        Sends request, saves and returns a logs iterator.

        :return: Iterator over sseclient.Event
        """
        url = "/api/v2/model/version/{}/logs".format(self.modelversion.id)
        logs_response = self.cluster.request("GET", url, stream=True)
        return sseclient.SSEClient(logs_response).events()

    def upload_training_data(self) -> 'DataUploadResponse':
        """
        Uploads training data for a given modelversion.

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
        if wait and not self.lock_till_released():
            raise ModelVersion.BadRequest(
                f"Failed to assign_metrics for {self.name}:{self.version}. Monitored model failed to be released."
            )
        
        for metric in metrics:
            modelversion = metric.model.upload(self.cluster)
            if wait and not modelversion.lock_till_released():
                raise ModelVersion.BadRequest(
                    f"Failed to assign_metrics for {self.name}:{self.version}. "
                    f"Monitoring model {modelversion.name}:{modelversion.version} failed to be released."
                )

            msc = MetricSpecConfig(
                model_version_id=modelversion.id,
                threshold=metric.threshold,
                threshold_op=metric.comparator
            )
            ms = MetricSpec.create(
                cluster=self.cluster,
                name=modelversion.name,
                model_version_id=self.id,
                config=msc
            )

    def to_proto(self):
        """
        Converts to ModelVersion protobuf message.

        :return: model version obj
        """
        return grpc_ModelVersion(
            _id=self.id,
            name=self.name,
            version=self.version,
            contract=self.contract,
            runtime=self.runtime,
            image=DockerImageProto(name=self.image.name, tag=self.image.tag),
            image_sha=self.image.sha256
        )
    
    def as_metric(self, threshold: int, comparator: MetricSpec) -> MetricModel:
        """
        Converts model to MetricModel.

        :param threshold:
        :param comparator:
        :return: MetricModel
        """
        return MetricModel(model=self, threshold=threshold, comparator=comparator)

    def __init__(self, id: int, model_id: int, name: str, version: int, contract: ModelContract, cluster: Cluster,
                 status: Optional[ModelVersionStatus], image: Optional[dict],
                 runtime: Optional[DockerImage], is_external: bool, metadata: dict = None, install_command: str = None,
                 training_data: str = None):
        super().__init__()

        self.id = id
        self.model_id = model_id
        self.name = name
        self.runtime = runtime
        self.is_external = is_external
        self.contract = contract
        self.cluster = cluster
        self.version = version
        self.image = image
        self.status = status
        self.metadata = metadata
        self.install_command = install_command
        self.training_data = training_data

    def __repr__(self):
        return "ModelVersion {}:{}".format(self.name, self.version)

    class NotFound(Exception):
        pass

    class BadRequest(Exception):
        pass


class DataProfileStatus(Enum):
    Success = "Success"
    Failure = "Failure"
    Processing = "Processing"
    NotRegistered = "NotRegistered"


class DataUploadResponse:
    """
    Class that wraps processing status of the training data upload.

    Check the status of the processing using `get_status()`
    Wait for the processing using `wait()`
    """

    def __init__(self, cluster: Cluster, model_version_id: int) -> 'DataUploadResponse':
        self.cluster = cluster
        self.model_version_id = model_version_id

    @property
    def url(self) -> str:
        if not hasattr(self, '__url'):
            self.__url = '/monitoring/profiles/batch/{}/status'.format(self.model_version_id)
        return self.__url
    
    @staticmethod
    def __tick(retry: int, sleep: int) -> Tuple[bool, int]:
        if retry == 0:
            return False, retry
        retry -= 1
        time.sleep(sleep)
        return True, retry

    def get_status(self) -> DataProfileStatus:
        response = self.cluster.request('GET', self.url)
        if response.status_code != 200:
            raise Cluster.BadResponse('Could not fetch the status of the data processing task')
        return DataProfileStatus[response.json()['kind']]

    def wait(self, retry=12, sleep=30):
        """
        Wait till data processing gets finished. 
        
        :param retry: Number of retries before giving up waiting.
        :param sleep: Sleep interval between retries.
        """
        while True:
            status = self.get_status()
            if status == DataProfileStatus.Success:
                break
            elif status == DataProfileStatus.Processing:
                can_be_polled, retry = self.__tick(retry, sleep)
                if can_be_polled:
                    continue
                raise DataUploadResponse.DataProcessingNotFinished
            elif status == DataProfileStatus.NotRegistered:
                can_be_polled, retry = self.__tick(retry, sleep)
                if can_be_polled:
                    continue
                raise DataUploadResponse.DataProcessingNotRegistered
            raise DataUploadResponse.DataProcessingFailed

    class DataProcessingNotRegistered(Exception):
        pass

    class DataProcessingNotFinished(Exception):
        pass

    class DataProcessingFailed(Exception):
        pass
