import datetime
import json
import logging
import os
import tarfile
from enum import Enum
from numbers import Number

import sseclient
import yaml
from hydro_serving_grpc import DT_INVALID, TensorShapeProto, DataType
from hydro_serving_grpc.contract import DataProfileType, ModelField, ModelSignature, ModelContract
from hydro_serving_grpc.manager import ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.contract import name2dtype, contract_from_dict, contract_to_dict
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.image import DockerImage
from hydrosdk.monitoring import MetricModel, MetricSpec


def resolve_paths(path, payload):
    return {os.path.normpath(os.path.join(path, v)): v for v in payload}


def read_yaml(path):
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
        protocontract = contract_from_dict(contract)
    else:
        protocontract = None
    model = LocalModel(
        name=name,
        contract=protocontract,
        runtime=runtime,
        payload=payload,
        path=path
    )
    return model


def read_py(path):
    pass


class BaseModel:
    """
    Base class for LocalModel and Model
    """
    def as_metric(self, threshold: int, comparator: MetricSpec) -> MetricModel:
        """
        Turns model into Metric
        """
        return MetricModel(model=self, threshold=threshold, comparator=comparator, name=self.name)

    def with_metrics(self, metrics: list):
        """
        Adds metrics to the model
        """
        self.metrics = metrics
        return self


class LocalModel(BaseModel):
    @staticmethod
    def create(path, name, runtime, contract=None, install_command=None):
        if contract:
            contract.validate()
        else:
            pass  # infer the contract
        pass

    @staticmethod
    def from_file(path):
        """
        Reads model definition from .yaml file or serving.py
        """
        ext = os.path.splitext(path)[1]
        if ext in ['.yml', '.yaml']:
            return read_yaml(path)
        elif ext == '.py':
            return read_py(path)
        else:
            raise ValueError("Unsupported file extension: {}".format(ext))

    def __init__(self, name, contract, runtime, payload, path=None):
        if not isinstance(name, str):
            raise TypeError("name is not a string")
        self.name = name
        if not isinstance(runtime, DockerImage):
            raise TypeError("runtime is not a DockerImage")
        self.runtime = runtime
        if contract and not isinstance(contract, ModelContract):
            raise TypeError("contract is not a ModelContract")
        self.contract = contract
        if isinstance(payload, list):
            self.payload = resolve_paths(path=path, payload=payload)
            self.path = path
        if isinstance(payload, dict):
            self.payload = payload

    def __repr__(self):
        return "LocalModel {}".format(self.name)

    def upload(self, cluster):
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
            "contract": contract_to_dict(self.contract)
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
            print(json_res)
            version_id = json_res['id']
            model = Model(
                id=version_id,
                name=json_res['model']['name'],
                version=json_res['modelVersion'],
                contract=contract_to_dict(self.contract),
                runtime=self.runtime,
                image=DockerImage(json_res['image'].get('name'), json_res['image'].get('tag'), json_res['image'].get('sha256')),
                cluster=cluster)
            return UploadResponse(model=model, version_id=version_id)
        else:
            raise ValueError("Error during model upload. {}".format(result.text))


class Model(BaseModel):
    BASE_URL = "/api/v2/model"

    @staticmethod
    def find(cluster, name=None, version=None, id=None):
        pass

    @staticmethod
    def find_by_id(cluster, id=None):
        pass

    @staticmethod
    def from_proto(proto, cluster):
        Model(
            id=proto.id,
            name=proto.name,
            version=proto.version,
            contract=proto.contract,
            runtime=proto.runtime,
            image=DockerImage(name=proto.image.name, tag=proto.image.tag, sha256=proto.image_sha),
            cluster=cluster
        )

    @staticmethod
    def delete_by_id(cluster, model_id):
        res = cluster.request("DELETE", Model.BASE_URL + "/{}".format(model_id))
        if res.ok:
            return res.json()
        return None

    @staticmethod
    def list_models(cluster):
        result = cluster.request("GET", Model.BASE_URL)
        return result.json()

    def to_proto(self):
        return ModelVersion(
            _id=self.id,
            name=self.name,
            version=self.version,
            contract=self.contract,
            runtime=self.runtime,
            image=DockerImageProto(name=self.image.name, tag=self.image.tag),
            image_sha=self.image.sha256
        )

    def __init__(self, id, name, version, contract, runtime, image, cluster):
        self.cluster = cluster
        self.id = id
        self.name = name
        self.version = version
        self.runtime = runtime
        self.image = image
        self.contract = contract

    def __repr__(self):
        return "Model {}:{}".format(self.name, self.version)


class BuildStatus(Enum):
    BUILDING = "BUILDING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class UploadResponse:
    def __init__(self, model, version_id):
        self.cluster = model.cluster
        self.model = model
        self.model_version_id = version_id
        self.cluster = self.model.cluster
        self._logs_iterator = self.logs()
        self.last_log = ""
        self._status = ""

    def logs(self):
        logger = logging.getLogger("ModelDeploy")
        try:
            url = "/api/v2/model/version/{}/logs".format(self.model_version_id)
            logs_response = self.model.cluster.request("GET", url, stream=True)
            self._logs_iterator = sseclient.SSEClient(logs_response).events()
        except RuntimeError:
            logger.exception("Unable to get build logs")
            self._logs_iterator = None
        return self._logs_iterator

    def set_status(self) -> None:
        try:
            if self.last_log.startswith("Successfully tagged"):
                self._status = BuildStatus.FINISHED
            else:
                self.last_log = next(self._logs_iterator).data
                self._status = BuildStatus.BUILDING
        except StopIteration:
            if not self._status == BuildStatus.FINISHED:
                self._status = BuildStatus.FAILED

    def get_status(self):
        return self._status

    def not_ok(self) -> bool:
        self.set_status()
        return self.get_status() == BuildStatus.FAILED

    def ok(self):
        self.set_status()
        return self.get_status() == BuildStatus.FINISHED

    def building(self):
        self.set_status()
        return self.get_status() == BuildStatus.BUILDING

    def request_model(self):
        return self.cluster.request("GET", f"api/v2/model/{self.model.id}")
