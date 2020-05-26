import datetime
import json
import logging
import os
import tarfile
from enum import Enum
from typing import Optional, List

import sseclient
import yaml
from hydro_serving_grpc.contract import ModelContract
from hydro_serving_grpc.manager import ModelVersion as grpc_ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.cluster import Cluster
from hydrosdk.contract import ModelContract_to_contract_dict, contract_dict_to_ModelContract, \
    contract_yaml_to_ModelContract
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, MetricModel


def resolve_paths(path, payload):
    """
    Appends each element of payload to the path and makes {resolved_path: payload_element} dict

    :param path: absolute path
    :param payload: list of relative paths
    :return: dict with {resolved_path: payload_element}
    """
    return {os.path.normpath(os.path.join(path, v)): v for v in payload}


def read_yaml(path):
    """
    Deserializes LocalModel from yaml definition

    :param path:
    :raises InvalidYAMLFile: if passed yamls are invalid
    :return: LocalModel obj
    """
    logger = logging.getLogger('read_yaml')
    with open(path, 'r') as f:
        model_docs = [x for x in yaml.safe_load_all(f) if x.get("kind").lower() == "model"]
    if not model_docs:
        raise InvalidYAMLFile(path, "Couldn't find proper documents (kind: model)")
    if len(model_docs) > 1:
        logger.warning("Multiple YAML documents detected. Using the first one.")
        logger.debug(model_docs[0])
    model_doc = model_docs[0]
    name = model_doc.get('name')
    if not name:
        raise InvalidYAMLFile(path, "name is not defined")
    folder = os.path.dirname(path)
    original_payload = model_doc.get('payload')
    if not original_payload:
        raise InvalidYAMLFile(path, "payload is not defined")
    payload = resolve_paths(folder, original_payload)
    full_runtime = model_doc.get('runtime')
    if not full_runtime:
        raise InvalidYAMLFile(path, "runtime is not defined")
    split = full_runtime.split(":")
    runtime = DockerImage(
        name=split[0],
        tag=split[1],
        sha256=None
    )
    contract = model_doc.get('contract')

    if contract:
        protocontract = contract_yaml_to_ModelContract(model_name=name, yaml_contract=contract)
    else:
        protocontract = None

    model = LocalModel(
        name=name,
        contract=protocontract,
        runtime=runtime,
        payload=payload,
        path=path,
        install_command=model_doc.get('install-command'),
        training_data=model_doc.get('training-data'),
        metadata=model_doc.get('metadata')
    )
    return model


class Metricable:
    """
    Every model can be monitored with a set of metrics (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#metrics)
    """

    def __init__(self):
        self.metrics: [Metricable] = []

    def as_metric(self, threshold: int, comparator: MetricSpec) -> MetricModel:
        """
        Turns model into Metric Model

        :param threshold:
        :param comparator:
        :return: MetricModel
        """

        return MetricModel(model=self, threshold=threshold, comparator=comparator)

    def with_metrics(self, metrics: list):
        """
        Adds metrics to the model

        :param metrics: list of metrics
        :return: self Metricable
        """

        self.metrics = metrics
        return self


class LocalModel(Metricable):
    """
    Local Model
    A model is a machine learning model or a processing function that consumes provided inputs
    and produces predictions or transformations

    https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models
    """

    @staticmethod
    def create(path, name, runtime, contract=None, install_command=None):
        """
        Validates contract
        :param path:
        :param name:
        :param runtime:
        :param contract:
        :param install_command:
        :return: None
        """
        if contract:
            contract.validate()
        else:
            pass  # infer the contract
        pass

    @staticmethod
    def from_file(path):
        """
        Reads model definition from .yaml file or serving.py
        :param path:
        :raises ValueError: If not yaml or py
        :return: LocalModel obj
        """
        ext = os.path.splitext(path)[1]
        if ext in ['.yml', '.yaml']:
            return read_yaml(path)
        elif ext == '.py':
            raise NotImplementedError(".py file parsing is not supported yet")
        else:
            raise ValueError("Unsupported file extension: {}".format(ext))

    def __init__(self, name, runtime, payload, contract=None, path=None, metadata=None, install_command=None,
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
        return "LocalModel {}".format(self.name)

    def __upload(self, cluster: Cluster) -> 'UploadResponse':
        """
        Direct implementation of uploading one model to the server. For internal usage

        :param cluster: active cluster
        :raises ValueError: If server returned not 200
        :return: UploadResponse obj
        """
        logger = logging.getLogger("ModelDeploy")

        events_response = cluster.request("GET", "/api/v2/events", stream=True)
        sse_client = sseclient.SSEClient(events_response)

        now_time = datetime.datetime.now()
        hs_folder = ".hs"
        os.makedirs(hs_folder, exist_ok=True)
        tarballname = "{}-{}".format(self.name, now_time)
        tarpath = os.path.join(hs_folder, tarballname)
        logger.debug("Creating payload tarball {} for {} model".format(tarpath, self.name))
        with tarfile.open(tarpath, "w:gz") as tar:
            for source, target in self.payload.items():
                logger.debug("Archiving %s as %s", source, target)
                tar.add(source, arcname=target)
        # TODO upload it to the manager

        meta = {
            "name": self.name,
            "runtime": {"name": self.runtime.name,
                        "tag": self.runtime.tag,
                        "sha256": self.runtime.sha256},
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
        result = cluster.request("POST", "/api/v2/model/upload",
                                 data=encoder,
                                 headers={'Content-Type': encoder.content_type})
        if result.ok:
            model_version_json = result.json()

            modelversion = ModelVersion.from_json(cluster=cluster, model_version=model_version_json)
            return UploadResponse(modelversion=modelversion, sse_client=sse_client)
        else:
            raise ValueError("Error during model upload. {}".format(result.text))

    def upload(self, cluster: Cluster, wait: bool = True) -> dict:
        """
        Uploads Local Model
        :param cluster: active cluster
        :param wait: wait till model version is released
        :return: {model_obj: upload_resp}
        """

        # TODO divide into two different methods, more details @kmakarychev
        root_model_upload_response = self.__upload(cluster)

        # if wait flag == True and uploading failed we raise an error, otherwise continue execution
        if not root_model_upload_response.lock_till_released() and wait:
            raise ModelVersion.BadRequest(
                (f"Model version {root_model_upload_response.modelversion.id} has upload status: "
                 f"{ModelVersionStatus.Failed.value}"))

        models_dict = {self: root_model_upload_response}
        if self.metrics:
            for metric in self.metrics:
                upload_response = metric.model.__upload(cluster)

                # if wait flag == True and uploading failed we raise an error, otherwise continue execution
                if not upload_response.lock_till_released() and wait:
                    raise ModelVersion.BadRequest(
                        (f"Model version {upload_response.modelversion.id} has upload status: "
                         f"{ModelVersionStatus.Failed.value}"))

                msc = MetricSpecConfig(model_version_id=upload_response.modelversion.id,
                                       threshold=metric.threshold,
                                       threshold_op=metric.comparator)

                ms = MetricSpec.create(cluster=upload_response.modelversion.cluster,
                                       name=upload_response.modelversion.name,
                                       model_version_id=root_model_upload_response.modelversion.id,
                                       config=msc)

                models_dict[metric] = upload_response

        return models_dict


class ModelVersionStatus(Enum):
    """
    Model building statuses
    """
    Assembling = "Assembling"
    Released = "Released"
    Failed = "Failed"


class ModelVersion(Metricable):
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
                f"Failed to find ModelVersion for name={name}, version={version} . {resp.status_code} {resp.text}")

    @staticmethod
    def find_by_id(cluster: Cluster, id_) -> 'ModelVersion':
        """
        Finds a modelversion on server by id

        :param cluster: active cluster
        :param id_: model version id
        :raises Exception: if server returned not 200
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
        Internal method used for deserealization of a Model from json object
        :param cluster:
        :param model_version: a dictionary
        :return: A Model instance
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
    def create(cluster, name: str, contract: ModelContract, metadata: Optional[dict] = None) -> 'ModelVersion':
        """
        Creates modelversion on the server
        :param cluster: active cluster
        :param name: name of model
        :param contract:
        :param metadata:
        :raises Exception: If server returned not 200
        :return: modelversion
        """
        model = {
            "name": name,
            "contract": ModelContract_to_contract_dict(contract),
            "metadata": metadata
        }

        resp = cluster.request(method="POST", url="/api/v2/externalmodel", json=model)
        if resp.ok:
            resp_json = resp.json()
            mv_obj = ModelVersion.from_json(cluster=cluster, model_version=resp_json)
            return mv_obj
        raise ModelVersion.BadRequest(
            f"Failed to create external model. External model = {model}. {resp.status_code} {resp.text}")

    @staticmethod
    def delete_by_model_id(cluster: Cluster, model_id: int) -> dict:
        """
        Deletes modelversion by model id
        :param cluster: active cluster
        :param model_id: model version id
        :return: if 200, json. Otherwise None
        """
        res = cluster.request("DELETE", ModelVersion.BASE_URL + "/{}".format(model_id))
        if res.ok:
            return res.json()

        raise ModelVersion.BadRequest(f"Failed to list delete by id modelversions. {res.status_code} {res.text}")

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

        raise ModelVersion.BadResponse(
            f"Failed to list model versions. {resp.status_code} {resp.text}")

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

    def to_proto(self):
        """
        Turns Model to Model version

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

    def __init__(self, id: int, model_id: int, name: str, version: int, contract: ModelContract, cluster: Cluster,
                 status: Optional[ModelVersionStatus], image: Optional[dict],
                 runtime: Optional[DockerImage], is_external: bool, metadata: dict = None, install_command: str = None):
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

    def __repr__(self):
        return "ModelVersion {}:{}".format(self.name, self.version)

    class NotFound(Exception):
        pass

    class BadRequest(Exception):
        pass

    class BadResponse(Exception):
        pass


class UploadResponse:
    """
    Class that wraps assembly status and logs logic.

    Check the status of the assembly using `get_status()`
    Check the build logs using `logs()`
    """

    def __init__(self, modelversion: ModelVersion, sse_client: sseclient.SSEClient):
        self.cluster = modelversion.cluster
        self.modelversion = modelversion
        self.sse_client = sse_client

    def logs(self):
        """
        Sends request, saves and returns logs iterator
        :return: log iterator
        """
        url = "/api/v2/model/version/{}/logs".format(self.modelversion.id)
        logs_response = self.modelversion.cluster.request("GET", url, stream=True)
        return sseclient.SSEClient(logs_response).events()

    def poll_modelversion(self) -> None:
        """
        Checks last log record and sets upload status
        :raises StopIteration: If something went wrong with iteration over logs
        :return: None
        """
        self.modelversion = ModelVersion.find(self.cluster, self.modelversion.name,
                                              self.modelversion.version)

    def get_status(self):
        """
        Gets current status of upload

        :return: status
        """
        self.poll_modelversion()
        return self.modelversion.status

    def not_ok(self) -> bool:
        """
        Checks current status and returns if it is not ok
        :return: if not uploaded
        """
        return self.get_status() == ModelVersionStatus.Failed

    def ok(self) -> bool:
        """
        Checks current status and returns if it is ok
        :return: if uploaded
        """
        return self.get_status() == ModelVersionStatus.Released

    def building(self) -> bool:
        """
        Checks current status and returns if it is building
        :return: if building
        """
        return self.get_status() == ModelVersionStatus.Assembling

    # TODO: Add logging
    def lock_till_released(self) -> bool:
        """
        Waits till the model is released
        :raises ModelVersion.BadRequest: if model upload fails
        :return:
        """
        for event in self.sse_client.events():
            if event.event == "ModelUpdate":
                data = json.loads(event.data)

                if data.get("id") == self.modelversion.id:
                    status = data.get("status")
                    if status == ModelVersionStatus.Failed.value:
                        return False
                    elif status == ModelVersionStatus.Released.value:
                        return True
