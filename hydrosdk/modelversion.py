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
from hydro_serving_grpc.contract import ModelContract
from hydro_serving_grpc.manager import ModelVersion as grpc_ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.cluster import Cluster
from hydrosdk.contract import ModelContract_to_contract_dict, contract_dict_to_ModelContract, validate_contract
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, MetricModel, ThresholdCmpOp
from hydrosdk.exceptions import BadRequest, BadResponse
from hydrosdk.utils import handle_request_error, read_in_chunks


def _upload_training_data(cluster: Cluster, modelversion_id: int, path: str) -> 'DataUploadResponse':
    """
    Upload training data to Hydrosphere

    :param cluster: Cluster instance
    :param modelversion_id: Id of the model version, for which to upload training data
    :param path: Path to the training data
    :raises BadResponse: if request failed to process by Hydrosphere
    :return: DataUploadResponse obj
    """
    if path.startswith('s3://'):
        resp = _upload_s3_file(cluster, modelversion_id, path)
    else:
        resp = _upload_local_file(cluster, modelversion_id, path)
    if resp.ok:
        return DataUploadResponse(cluster, modelversion_id)
    raise BadResponse('Failed to upload training data')


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
    url = '/monitoring/profiles/batch/{}'.format(modelversion_id)
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
    A local instance of the model yet to be uploaded to the cluster.
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

    :Example:

    Create a local instance of the model and upload it to the cluster.
    >>> from hydrosdk.cluster import Cluster
    >>> from hydrosdk.image import DockerImage
    >>> from hydrosdk.contract import SignatureBuilder, ProfilingType, ModelContract
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> runtime = DockerImage("hydrosphere/serving-runtime-python-3.7", "latest", None)
    >>> payload = ["src/func_main.py", "requirements.txt"]
    >>> signature = SignatureBuilder("predict") \
            .with_input("x", int, "scalar", ProfilingType.NUMERICAL) \
            .with_output("y", float, "scalar", ProfilingType.NUMERICAL) \
            .build()
    >>> install_command = "pip install -r requirements.txt"
    >>> training_data = "training-data.csv"
    >>> contract = ModelContract(predict=signature)
    >>> localmodel = LocalModel(name="my-model", runtime=runtime, payload=payload, contract=contract
                                install_command=install_command, training_data=training_data)
    >>> modelversion = localmodel.upload(cluster)
    >>> modelversion.lock_till_released()
    >>> data_upload_response = modelversion.upload_training_data()
    """
    def __init__(self, name: str, runtime: DockerImage, payload: List[str], contract: ModelContract, 
                 metadata: Optional[dict] = None, install_command: Optional[str] = None,
                 training_data: Optional[str] = None) -> 'LocalModel':
        if not isinstance(name, str):
            raise TypeError("name is not a string")
        self.name = name
        if not isinstance(runtime, DockerImage):
            raise TypeError("runtime is not a DockerImage")
        self.runtime = runtime

        if not isinstance(contract, ModelContract):
            raise TypeError("contract is not a ModelContract")
        validate_contract(contract)
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
        Upload local model instance to the cluster.

        :param cluster: active cluster
        :return: ModelVersion object
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
        handle_request_error(
            resp, f"Failed to upload local model. {resp.status_code} {res.text}")

        modelversion = ModelVersion._from_json(
            cluster=cluster, model_version=resp.json())
        modelversion.training_data = self.training_data
        return modelversion


class ModelVersionStatus(Enum):
    """
    Model building statuses
    """
    ASSEMBLING = "Assembling"
    RELEASED = "Released"
    FAILED = "Failed"

    @classmethod
    def is_assembling(cls, status: str) -> bool:
        if status == cls.ASSEMBLING.value:
            return True
        return False

    @classmethod
    def is_released(cls, status: str) -> bool:
        if status == cls.RELEASED.value:
            return True
        return False


class ModelVersion:
    """
    A Model, registered within the cluster. ModelVersion represents one of the Models' versions.
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

    :Example:

    List all modelversions, registered on the cluster.
    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> for modelversion in ModelVersion.list_all(cluster):
            print(modelversion)

    Upload training data for a modelversion.
    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> modelversion = ModelVersion.find(cluster, "my-model", 1)
    >>> modelversion.lock_till_released()
    >>> modelversion.training_data = "s3://my-bucket/path/to/training-data.csv"
    >>> data_upload_response = modelversion.upload_traininf_data()

    Assign custom monitoring metrics for a modelversion.
    >>> from hydrosdk.cluster import Cluster
    >>> from hydrosdk.monitoring import ThresholdCmpOp
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> modelversion = ModelVersion.find(cluster, "my-model", 1)
    >>> modelversion_metric = ModelVersion.find(cluster, "my-model-metric", 1)
    >>> modelversion_metric = modmodelversion_metric.as_metric(1.4, ThresholdCmpOp.LESS_EQ)
    >>> modelversion.assign_metrics([modelversion_metric])

    Create an external model.
    >>> from hydrosdk.cluster import Cluster
    >>> from hydrosdk.contract import SignatureBuilder, ProfilingType, ModelContract
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> signature = SignatureBuilder("predict") \
            .with_input("x", int, "scalar", ProfilingType.NUMERICAL) \
            .with_output("y", float, "scalar", ProfilingType.NUMERICAL) \
            .build()
    >>> training_data = "training-data.csv"
    >>> contract = ModelContract(predict=signature)
    >>> modelversion = ModelVersion.create_externalmodel(
            cluster=cluster, name="my-external-model", contract=contract, training_data=training_data
        )
    >>> data_upload_response = modelversion.upload_training_data()

    Check logs from a ModelVersion.
    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> modelversion = ModelVersion.find(cluster, "my-model", 1)
    >>> for event in modelversion.logs():
            print(event.data)
    """
    BASE_URL = "/api/v2/model"

    @staticmethod
    def find(cluster: Cluster, name: str, version: int) -> 'ModelVersion':
        """
        Find a ModelVersion on the cluster by model name and a version.

        :param cluster: active cluster
        :param name: name of the model
        :param version: version of the model
        :return: ModelVersion object
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version/{}/{}".format(name, version))
        handle_request_error(
            resp, f"Failed to find modelversion for name={name}, version={version}. {resp.status_code} {resp.text}")
        return ModelVersion._from_json(cluster=cluster, model_version=resp.json())

    @staticmethod
    def find_by_id(cluster: Cluster, id: int) -> 'ModelVersion':
        """
        Find a ModelVersion on the cluster by id.

        :param cluster: active cluster
        :param id: model version id
        :return: ModelVersion object
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version")
        handle_request_error(
            resp, f"Failed to find modelversion by id={id}. {resp.status_code} {resp.text}")
        for modelversion_json in resp.json():
            if modelversion_json['id'] == id:
                return ModelVersion._from_json(cluster, modelversion_json)

    @staticmethod
    def _from_json(cluster: Cluster, model_version: dict) -> 'ModelVersion':
        """
        Internal method used for deserealization of a ModelVersion from a json object.

        :param cluster: active cluster
        :param model_version: json response from the cluster
        :return: ModelVersion object
        """
        id = model_version["id"]
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
            id=id,
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
    def create_externalmodel(cluster: Cluster, name: str, contract: ModelContract, 
               metadata: Optional[dict] = None, training_data: Optional[str] = None) -> 'ModelVersion':
        """
        Create an external ModelVersion on the cluster. 

        :param cluster: active cluster
        :param name: name of model
        :param contract: contract of the model
        :param metadata: metadata for the model
        :return: ModelVersion object
        """
        model = {
            "name": name,
            "contract": ModelContract_to_contract_dict(contract),
            "metadata": metadata
        }
        resp = cluster.request(method="POST", url="/api/v2/externalmodel", json=model)
        handle_request_error(
            resp, f"Failed to create an external model. {resp.status_code} {resp.text}")
        return ModelVersion._from_json(cluster, resp.json())

    @staticmethod
    def list_all(cluster: Cluster) -> List['ModelVersion']:
        """
        List all model versions on the cluster.

        :param cluster: active cluster
        :return: list of ModelVersions 
        """
        resp = cluster.request("GET", ModelVersion.BASE_URL + "/version")
        handle_request_error(
            resp, f"Failed to list model versions. {resp.status_code} {resp.text}")
        return [ModelVersion._from_json(cluster=cluster, model_version=model_version_json)
                for model_version_json in resp.json()]

    @staticmethod
    def list_all_by_model_name(cluster: Cluster, model_name: str) -> list:
        """
        List all model versions on the cluster filtered by model_name, sorted in ascending 
        order by version.

        :param cluster: active cluster
        :param model_name: model name
        :return: list of Models versions for provided model name
        """
        all_models = ModelVersion.list_all(cluster=cluster)
        modelversions_by_name = [model for model in all_models if model.name == model_name]
        sorted_by_version = sorted(modelversions_by_name, key=lambda model: model.version)
        return sorted_by_version
    
    def _update_status(self):
        self.status = self.find(self.cluster, self.name, self.version).status

    def lock_till_released(self) -> bool:
        """
        Lock till the model completes assembling.
        
        :raises ModelVersion.ReleaseFailed: if model failed to be released
        """
        events_steam = cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        self._update_status()
        if not ModelVersionStatus.is_assembling(self.status) and 
                ModelVersionStatus.is_released(self.status): 
            return None
        try:
            for event in events_client.events():
                if event.event == "ModelUpdate":
                    data = json.loads(event.data)
                    if data.get("id") == self.id:
                        self.status = data.get("status")
                        if ModelVersionStatus.is_released(self.status):
                            return None
                        raise ModelVersion.ReleaseFailed()
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
            raise BadRequest(
                f"Failed to assign metrics for {self}. Monitored model failed to release."
            )
        
        for metric in metrics:
            modelversion = metric.model.upload(self.cluster)
            if wait and not modelversion.lock_till_released():
                raise BadRequest(
                    f"Failed to assign metrics for {self}. Monitoring model {modelversion} failed to release."
                )

            msc = MetricSpecConfig(
                modelversion_id=modelversion.id,
                threshold=metric.threshold,
                threshold_op=metric.comparator
            )
            ms = MetricSpec.create(
                cluster=self.cluster,
                name=modelversion.name,
                modelversion_id=self.id,
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
    
    def as_metric(self, threshold: int, comparator: ThresholdCmpOp) -> MetricModel:
        """
        Converts model to MetricModel.

        :param threshold:
        :param comparator:
        :return: MetricModel
        """
        return MetricModel(model=self, threshold=threshold, comparator=comparator)

    def __init__(self, cluster: Cluster, id: int, model_id: int, name: str, version: int, 
                 contract: ModelContract, status: Optional[ModelVersionStatus], image: Optional[dict], 
                 runtime: Optional[DockerImage], is_external: bool, metadata: Optional[dict] = None, 
                 install_command: Optional[str] = None, training_data: Optional[str] = None):
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

    class ReleaseFailed(BaseException):
        pass


class DataProfileStatus(Enum):
    SUCCESS = "Success"
    FAILURE = "Failure"
    PROCESSING = "Processing"
    NOT_REGISTERED = "NotRegistered"


class DataUploadResponse:
    """
    Handle processing status of the training data upload. 

    :Example:

    Wait till data processing gets finished.
    >>> from hydrosdk.cluster import Cluster
    >>> cluster = Cluster("http-cluster-endpoint")
    >>> modelversion = ModelVerison.find(cluster, "my-model", 1)
    >>> modelversion.training_data = "s3://bucket/path/to/training-data.csv"
    >>> data_upload_response = modelversion.upload_training_data()
    >>> try: 
            data_upload_response.wait(retry=3, sleep=30) 
        except DataUploadResponse.TimeOut:
            print("Timed out waiting for data processing getting finished")
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
    
    @staticmethod
    def __tick(retry: int, sleep: int) -> Tuple[bool, int]:
        if retry == 0:
            return False, retry
        retry -= 1
        time.sleep(sleep)
        return True, retry

    def get_status(self) -> DataProfileStatus:
        resp = self.cluster.request('GET', self.url)
        handle_request_error(
            resp, f"Failed to get status for modelversion_id={self.modelversion_id}. {resp.status_code} {resp.text}")
        return response.json()['kind']

    def wait(self, retry=12, sleep=30):
        """
        Wait till data processing gets finished. 
        
        :param retry: Number of retries before giving up waiting.
        :param sleep: Sleep interval between retries.
        """
        while True:
            status = self.get_status()
            if status == DataProfileStatus.SUCCESS.value:
                break
            elif status == DataProfileStatus.PROCESSING.value:
                can_be_polled, retry = self.__tick(retry, sleep)
                if can_be_polled:
                    continue
                raise DataUploadResponse.DataProcessingNotFinished
            elif status == DataProfileStatus.NOT_REGISTERED.value:
                can_be_polled, retry = self.__tick(retry, sleep)
                if can_be_polled:
                    continue
                raise DataUploadResponse.DataProcessingNotRegistered
            raise DataUploadResponse.DataProcessingFailed

    class NotRegistered(Exception):
        pass

    class TimeOut(Exception):
        pass

    class Failed(Exception):
        pass
