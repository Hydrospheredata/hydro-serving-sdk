import datetime
import json
import logging
import os
import tarfile
from numbers import Number

import sseclient
import yaml
from hydro_serving_grpc import DT_INVALID, TensorShapeProto
from hydro_serving_grpc.contract import DataProfileType, ModelField, ModelSignature, ModelContract
from hydro_serving_grpc.manager import ModelVersion, DockerImage as DockerImageProto
from requests_toolbelt.multipart.encoder import MultipartEncoder

from hydrosdk.contract import name2dtype
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.image import DockerImage


def shape_to_proto(user_shape):
    if user_shape == "scalar":
        shape = TensorShapeProto()
    elif user_shape is None:
        shape = None
    elif isinstance(user_shape, list):
        dims = []
        for dim in user_shape:
            if not isinstance(dim, Number):
                raise TypeError("shape_list contains incorrect dim", user_shape, dim)
            converted = TensorShapeProto.Dim(size=dim)
            dims.append(converted)
        shape = TensorShapeProto(dim=dims)
    else:
        raise ValueError("Invalid shape value", user_shape)
    return shape


def field_from_dict(name, data_dict):
    shape = data_dict.get("shape")
    dtype = data_dict.get("type")
    subfields = data_dict.get("fields")
    raw_profile = data_dict.get("profile", "NONE").upper()
    if raw_profile not in DataProfileType.keys():
        profile = "NONE"
    else:
        profile = raw_profile

    result_dtype = None
    result_subfields = None
    if dtype is None:
        if subfields is None:
            raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name, data_dict)
        else:
            subfields_buffer = []
            for k, v in subfields.items():
                subfield = field_from_dict(k, v)
                subfields_buffer.append(subfield)
            result_subfields = subfields_buffer
    else:
        result_dtype = name2dtype(dtype)
        if result_dtype == DT_INVALID:
            raise ValueError("Invalid contract: {} field has invalid datatype {}".format(name, dtype))

    if result_dtype is not None:
        result_field = ModelField(
            name=name,
            shape=shape_to_proto(shape),
            dtype=result_dtype,
            profile=profile
        )
    elif result_subfields is not None:
        result_field = ModelField(
            name=name,
            shape=shape_to_proto(shape),
            subfields=ModelField.Subfield(data=result_subfields),
            profile=profile
        )
    else:
        raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name, data_dict)
    return result_field


def contract_from_dict(data_dict):
    name = data_dict.get("name", "Predict")
    inputs = []
    outputs = []
    for in_key, in_value in data_dict["inputs"].items():
        input = field_from_dict(in_key, in_value)
        inputs.append(input)
    for out_key, out_value in data_dict["outputs"].items():
        output = field_from_dict(out_key, out_value)
        outputs.append(output)
    signature = ModelSignature(
        signature_name=name,
        inputs=inputs,
        outputs=outputs
    )
    return ModelContract(model_name="model", predict=signature)


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


class LocalModel:
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
        if not isinstance(payload, dict):
            raise TypeError("payload is not a dict")
        self.payload = payload
        if not isinstance(path, str):
            raise TypeError("path is not a str")
        self.path = path

    def __repr__(self):
        return "LocalModel {}".format(self.name)

    def deploy(self, cluster):
        logger = logging.getLogger("ModelDeploy")
        now_time = datetime.datetime.now()
        hs_folder = ".hs"
        os.makedirs(hs_folder, exist_ok=True)
        tarballname = "{}-{}".format(self.name, now_time)
        tarpath = os.path.join(hs_folder, tarballname)
        logger.debug("Creating payload tarball {} for {} model".format(tarpath, self.name))
        with tarfile.open(tarpath, "w:gz") as tar:
            for source, target in self.payload.values():
                logger.debug("Archiving %s as %s", source, target)
                tar.add(source, arcname=target)
        # TODO upload it to the manager
        metadata = {}
        encoder = MultipartEncoder(
            fields={
                "payload": ("filename", open(tarpath, "rb")),
                "metadata": json.dumps(metadata.__dict__)
            }
        )
        result = cluster.request("POST", "/api/v2/model/upload",
                                 data=encoder,
                                 headers={'Content-Type': encoder.content_type})
        if result.ok:
            json_res = result.json()
            version_id = json_res['modelVersionId']
            try:
                url = "/api/v2/model/version/{}/logs".format(version_id)
                logs_response = cluster.request("GET", url, stream=True)
                logs_iterator = sseclient.SSEClient(logs_response).events()
            except RuntimeError:
                logger.exception("Unable to get build logs")
                logs_iterator = None
            model = Model(
                id=version_id,
                name=json_res['name'],
                version=json_res['version'],
                contract=json_res['contract'],
                runtime=json_res['runtime'],
                image=json_res['image'],
                cluster=cluster)
            return model, logs_iterator
        else:
            raise ValueError("Error during model upload. {}".format(result.text()))


class Model:
    BASE_URL = "/api/v2/model"

    @staticmethod
    def find(cluster, name=None, version=None, id=None):
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
