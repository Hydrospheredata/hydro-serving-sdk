import datetime
import json
import logging
import os
import tarfile
import time
from enum import Enum
from typing import Optional

import sseclient
import yaml
from hydro_serving_grpc.contract import ModelContract
from hydro_serving_grpc.manager import ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.contract import ModelContract_to_contract_dict, contract_dict_to_ModelContract, contract_yaml_to_ModelContract
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.exceptions import MetricSpecException
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


def parse_model_from_json_dict(cluster, json_dict):
    """
    Parses json dict which could contain ExternalModel or Model
    :param cluster:
    :param json_dict:
    :return:
    """

    # assumption that a model is internal by default.
    # Externals are obligated to have "isExternal": true
    # if status is not in dict, then it's external
    # internal otherwise
    is_external = json_dict.get("isExternal", "status" not in json_dict)
    if is_external:
        return ExternalModel.ext_model_json_to_ext_model(json_dict)
    else:
        return Model.from_json(cluster, json_dict)


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
    def model_json_to_upload_response(cluster, model_json):
        """
        Deserialize model json into UploadResponse object

        :param cluster:
        :param model_json:
        :return: UploadResponse obj
        """
        model = Model.from_json(cluster, model_json)
        return UploadResponse(model=model)

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

    def __upload(self, cluster):
        """
        Direct implementation of uploading one model to the server. For internal usage

        :param cluster: active cluster
        :raises ValueError: If server returned not 200
        :return: UploadResponse obj
        """
        logger = logging.getLogger("ModelDeploy")
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
            json_res = result.json()
            return LocalModel.model_json_to_upload_response(cluster=cluster, model_json=json_res)
        else:
            raise ValueError("Error during model upload. {}".format(result.text))

    def upload(self, cluster) -> dict:
        """
        Uploads Local Model
        :param cluster: active cluster
        :raises MetricSpecException: If model not uploaded yet
        :return: {model_obj: upload_resp}
        """
        root_model_upload_response = self.__upload(cluster)
        models_dict = {self: root_model_upload_response}
        if self.metrics:
            for metric in self.metrics:
                upload_response = metric.model.__upload(cluster)

                msc = MetricSpecConfig(model_version_id=upload_response.model.id,
                                       threshold=metric.threshold,
                                       threshold_op=metric.comparator)
                ms_created = False

                while not ms_created:
                    try:
                        ms = MetricSpec.create(cluster=upload_response.model.cluster,
                                               name=upload_response.model.name,
                                               model_version_id=root_model_upload_response.model.id,
                                               config=msc)
                        ms_created = True
                    except MetricSpecException:
                        time.sleep(1)
                models_dict[metric] = upload_response

        return models_dict


class ModelStatus(Enum):
    """
    Model building statuses
    """
    Assembling = "Assembling"
    Released = "Released"
    Failed = "Failed"


class Model(Metricable):
    """
    Model (A model is a machine learning model or a processing function that consumes provided inputs
    and produces predictions or transformations
    https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)
    """
    BASE_URL = "/api/v2/model"

    @staticmethod
    def find(cluster, name, version):
        """
        Finds a model on server by name and model version

        :param cluster: active cluster
        :param name: model name
        :param version: model version
        :raises Exception: if server returned not 200
        :return: Model obj
        """
        resp = cluster.request("GET", Model.BASE_URL + "/version/{}/{}".format(name, version))

        if resp.ok:
            model_json = resp.json()
            return parse_model_from_json_dict(cluster, model_json)

        else:
            raise Exception(
                f"Failed to find Model for name={name}, version={version} . {resp.status_code} {resp.text}")

    @staticmethod
    def find_by_id(cluster, model_id):
        """
        Finds a model on server by id

        :param cluster: active cluster
        :param model_id: model id
        :raises Exception: if server returned not 200
        :return: Model obj
        """
        resp = cluster.request("GET", Model.BASE_URL + "/version")

        if resp.ok:
            for model_json in resp.json():
                if model_json['id'] == model_id:
                    return parse_model_from_json_dict(cluster, model_json)

        raise Exception(
            f"Failed to find_by_id Model for model_id={model_id}. {resp.status_code} {resp.text}")

    @staticmethod
    def from_json(cluster, model_json):
        """
        Internal method used for deserealization of a Model from json object
        :param cluster:
        :param model_json: a dictionary
        :return: A Model instance
        """

        model_id = model_json["id"]
        model_name = model_json["model"]["name"]
        model_version = model_json["modelVersion"]
        model_contract = contract_dict_to_ModelContract(model_json["modelContract"])

        # external model deserialization handling
        # TODO: get its own endpoint for external model
        if not model_json.get("runtime"):
            model_json["runtime"] = {}

        model_runtime = DockerImage(model_json["runtime"].get("name"), model_json["runtime"].get("tag"),
                                    model_json["runtime"].get("sha256"))
        model_image = model_json.get("image")
        model_cluster = cluster

        status = ModelStatus[model_json['status']]
        metadata = model_json['metadata']

        return Model(
            id=model_id,
            name=model_name,
            version=model_version,
            contract=model_contract,
            runtime=model_runtime,
            image=model_image,
            cluster=model_cluster,
            status=status,
            metadata=metadata,
        )

    @staticmethod
    def delete_by_id(cluster, model_id):
        """
        Deletes model by id
        :param cluster: active cluster
        :param model_id: model id
        :return: if 200, json. Otherwise None
        """
        res = cluster.request("DELETE", Model.BASE_URL + "/{}".format(model_id))
        if res.ok:
            return res.json()
        return None

    @staticmethod
    def list_models(cluster) -> list:
        """
        List all models on server

        :param cluster: active cluster
        :return: list of extModel and Model
        """
        resp = cluster.request("GET", Model.BASE_URL + "/version")

        if resp.ok:
            model_versions_json = resp.json()

            models = []

            for model_version_json in model_versions_json:
                model = parse_model_from_json_dict(cluster, model_version_json)
                models.append(model)
            return models

        raise Exception(
            f"Failed to list model versions. {resp.status_code} {resp.text}")

    @staticmethod
    def list_models_by_model_name(cluster, model_name: str) -> list:
        """
        List all models on server filtered by model_name, sorted in ascending order by version

        :param cluster: active cluster
        :param model_name: model name
        :return: list of Models for provided model name
        """
        all_models = Model.list_models(cluster=cluster)

        models_by_name = [model for model in all_models if model.name == model_name]

        sorted_by_version = sorted(models_by_name, key=lambda model: model.version)

        return sorted_by_version

    def to_proto(self):
        """
        Turns Model to Model version

        :return: model version obj
        """
        return ModelVersion(
            _id=self.id,
            name=self.name,
            version=self.version,
            contract=self.contract,
            runtime=self.runtime,
            image=DockerImageProto(name=self.image.name, tag=self.image.tag),
            image_sha=self.image.sha256
        )

    def __init__(self, id, name, version, contract, runtime, image, cluster, status, metadata=None,
                 install_command=None):
        super().__init__()

        self.name = name
        self.runtime = runtime
        self.contract = contract
        self.cluster = cluster
        self.id = id
        self.version = version
        self.image = image

        self.status = status

        self.metadata = metadata
        self.install_command = install_command

    def __repr__(self):
        return "Model {}:{}".format(self.name, self.version)


class ExternalModel:
    """
    External models running outside of the Hydrosphere platform

    https://hydrosphere.io/serving-docs/latest/how-to/monitoring-external-models.html
    """
    BASE_URL = "/api/v2/externalmodel"

    @staticmethod
    def ext_model_json_to_ext_model(ext_model_json: dict):
        """
        Deserializes external model json to external model

        :param ext_model_json: external model json
        :return: external model obj
        """
        return ExternalModel(name=ext_model_json["model"]["name"],
                             id_=ext_model_json["model"]["id"],
                             contract=contract_dict_to_ModelContract(ext_model_json["modelContract"]),
                             metadata=ext_model_json.get("metadata"), version=ext_model_json["modelVersion"])

    @staticmethod
    def create(cluster, name: str, contract: ModelContract, metadata: Optional[dict] = None):
        """
        Creates external model on the server

        :param cluster: active cluster
        :param name: name of ext model
        :param contract:
        :param metadata:
        :raises Exception: If server returned not 200
        :return: external model
        """
        ext_model = {
            "name": name,
            "contract": ModelContract_to_contract_dict(contract),
            "metadata": metadata
        }

        resp = cluster.request(method="POST", url=ExternalModel.BASE_URL, json=ext_model)
        if resp.ok:
            resp_json = resp.json()
            return ExternalModel.ext_model_json_to_ext_model(resp_json)
        raise Exception(
            f"Failed to create external model. External model = {ext_model}. {resp.status_code} {resp.text}")

    # TODO: return ExternalModel rather than Model
    @staticmethod
    def find_by_name(cluster, name, version):
        """
        Finds ext model on server by name and version

        :param cluster: active cluster
        :param name:
        :param version:
        :return: Model
        """
        found_model = Model.find(cluster=cluster, name=name, version=version)
        return found_model

    @staticmethod
    def delete_by_id(cluster, model_id):
        """
        Deletes external model by model id

        :param cluster: active cluster
        :param model_id:
        :return: None
        """
        Model.delete_by_id(cluster=cluster, model_id=model_id)

    def __init__(self, name, id_, contract, version, metadata):
        self.name = name
        self.contract = contract
        self.version = version
        self.metadata = metadata
        self.id_ = id_


class UploadResponse:
    """
    Class that wraps assembly status and logs logic.

    Check the status of the assembly using `get_status()`
    Check the build logs using `logs()`
    """

    def __init__(self, model):
        self.cluster = model.cluster
        self.model = model

    def logs(self):
        """
        Sends request, saves and returns logs iterator
        :return: log iterator
        """
        url = "/api/v2/model/version/{}/logs".format(self.model.id)
        logs_response = self.model.cluster.request("GET", url, stream=True)
        return sseclient.SSEClient(logs_response).events()

    def poll_model(self) -> None:
        """
        Checks last log record and sets upload status
        :raises StopIteration: If something went wrong with iteration over logs
        :return: None
        """
        self.model = Model.find(self.cluster, self.model.name, self.model.version)

    def get_status(self):
        """
        Gets current status of upload

        :return: status
        """
        self.poll_model()
        return self.model.status

    def not_ok(self) -> bool:
        """
        Checks current status and returns if it is not ok
        :return: if not uploaded
        """
        return self.get_status() == ModelStatus.Failed

    def ok(self) -> bool:
        """
        Checks current status and returns if it is ok
        :return: if uploaded
        """
        return self.get_status() == ModelStatus.Released

    def building(self) -> bool:
        """
        Checks current status and returns if it is building
        :return: if building
        """
        return self.get_status() == ModelStatus.Assembling
