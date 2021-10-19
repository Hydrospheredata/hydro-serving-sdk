import os
import random
from hydro_serving_grpc.serving.runtime.api_pb2 import PredictRequest, PredictResponse

import pytest
from hydro_serving_grpc.serving.contract.tensor_pb2 import Tensor, TensorShape
from hydro_serving_grpc.serving.contract.types_pb2 import DataType

from hydrosdk.cluster import Cluster
from hydrosdk.signature import ModelField, ModelSignature, SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import ModelVersion, ModelVersionStatus, ModelVersionBuilder, \
    MonitoringConfiguration, resolve_paths, _analyze
from hydrosdk.exceptions import HydrosphereException, SignatureViolationException
from tests.common_fixtures import *
from tests.config import *


def test_resolve_paths():
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]
    resolved = resolve_paths(path, payload)
    assert resolved == {'/home/user/folder/model/cool/src/func_main.py': './src/func_main.py',
                        '/home/user/folder/model/cool/data/*': './data/*',
                        '/home/user/folder/model/cool/model/snapshot.proto': './model/snapshot.proto'}


def test_model_create_programmatically():
    name = config.default_model_name
    runtime = config.runtime
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './requirements.txt',
        './data/*',
        './model/snapshot.proto'
    ]
    resolved_payload = [
        '/home/user/folder/model/cool/src/func_main.py',
        '/home/user/folder/model/cool/requirements.txt',
        '/home/user/folder/model/cool/data/*',
        '/home/user/folder/model/cool/model/snapshot.proto',
    ]
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], ProfilingType.NUMERICAL) \
        .with_output('out1', 'double', [-1], ProfilingType.NUMERICAL).build()
    monitoring_configuration = MonitoringConfiguration(batch_size=10)
    metadata = {"key": "value"}
    install_command = "pip install -r requirements.txt"
    training_data = "s3://bucket/path/to/training-data"
    model_version_builder = ModelVersionBuilder(name, path) \
        .with_runtime(runtime) \
        .with_payload(payload) \
        .with_signature(signature) \
        .with_metadata(metadata) \
        .with_install_command(install_command) \
        .with_training_data(training_data) \
        .with_monitoring_configuration(monitoring_configuration)
    assert model_version_builder.name == name
    assert model_version_builder.runtime == runtime
    assert model_version_builder.path == path
    assert list(model_version_builder.payload.values()) == payload
    assert list(model_version_builder.payload.keys()) == resolved_payload
    assert model_version_builder.signature == signature
    assert model_version_builder.monitoring_configuration == monitoring_configuration

def test_model_relative_training_data():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    training_data = "./training-data.csv"
    try:
        b = ModelVersionBuilder(name, path) \
            .with_training_data(training_data)
        pytest.fail(f"Should be unreachable: {b.training_data}")
    except FileNotFoundError as err:
        assert str(err) == "Can't find training data file /home/user/folder/model/cool/training-data.csv"

def test_model_absolute_training_data():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    training_data = "/home/data/training-data.csv"
    try:
        b = ModelVersionBuilder(name, path) \
            .with_training_data(training_data)
        pytest.fail(f"Should be unreachable: {b.training_data}")
    except FileNotFoundError as err:
        assert str(err) == "Can't find training data file /home/data/training-data.csv"

def test_model_create_signature_validation():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    signature = SignatureBuilder('infer').build()
    with pytest.raises(SignatureViolationException):
        ModelVersionBuilder(name, path).with_signature(signature)


def test_model_version_builder_build(cluster: Cluster):
    name = config.default_model_name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], ProfilingType.NONE) \
        .with_output('out1', 'double', [-1], ProfilingType.NONE).build()
    batch_size = 10
    model_version_builder = ModelVersionBuilder(name, model_path) \
        .with_runtime(config.runtime) \
        .with_payload(['./src/func_main.py']) \
        .with_signature(signature) \
        .with_metadata({"key": "value"}) \
        .with_monitoring_configuration(MonitoringConfiguration(batch_size=batch_size))
    mv: ModelVersion = model_version_builder.build(cluster)
    assert mv.status is ModelVersionStatus.Assembling
    mv.lock_till_released(timeout=config.lock_timeout)
    assert mv.status is ModelVersionStatus.Released
    assert mv.monitoring_configuration.batch_size == batch_size


def test_modelversion_find_by_id(cluster: Cluster, model_version_builder: ModelVersionBuilder):
    mv: ModelVersion = model_version_builder.build(cluster)
    mv_found: ModelVersion = ModelVersion.find_by_id(cluster, mv.id)
    assert mv.id == mv_found.id


def test_modelversion_find(cluster: Cluster, model_version_builder: ModelVersionBuilder):
    mv: ModelVersion = model_version_builder.build(cluster)
    mv_found: ModelVersion = ModelVersion.find(cluster, mv.name, mv.version)
    assert mv.id == mv_found.id


def test_lock_till_released_failed(cluster: Cluster, runtime: DockerImage, 
                                   payload: list, signature: ModelSignature):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    model_version_builder = ModelVersionBuilder(config.default_model_name, model_path) \
        .with_runtime(runtime) \
        .with_payload(payload) \
        .with_signature(signature) \
        .with_install_command("exit 1")
    mv: ModelVersion = model_version_builder.build(cluster)
    with pytest.raises(ModelVersion.ReleaseFailed):
        mv.lock_till_released(timeout=config.lock_timeout)
        

def test_build_logs_not_empty(cluster: Cluster, model_version_builder: ModelVersionBuilder):
    mv: ModelVersion = model_version_builder.build(cluster)
    mv.lock_till_released(timeout=config.lock_timeout)
    i = 0
    for _ in mv.build_logs():
        i += 1
    assert i > 0


def test_modelversion_list(cluster: Cluster, model_version_builder: ModelVersionBuilder):
    mv: ModelVersion = model_version_builder.build(cluster)
    assert mv.id in [modelversion.id for modelversion in ModelVersion.list(cluster)]

    
def test_ModelField_dt_invalid_input():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    signature = ModelSignature(
        signature_name="test", 
        inputs=[ModelField(name="input", shape=TensorShape())],
        outputs=[ModelField(name="output", shape=TensorShape(), dtype=DataType.Name(2))]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model with invalid dtype in the signature input.*"):
        _ = ModelVersionBuilder(name, path).with_signature(signature)


def test_ModelField_dt_invalid_output():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    signature = ModelSignature(
        signature_name="test",
        inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())],
        outputs=[ModelField(name="test", shape=TensorShape())]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model with invalid dtype in the signature output.*"):
        _ = ModelVersionBuilder(name, path).with_signature(signature)


def test_ModelField_contact_signature_name_none():
    name = config.default_model_name
    path = "/home/user/folder/model/cool/"
    signature = ModelSignature(
        inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())],
        outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model without signature_name is not allowed.*"):
        _ = ModelVersionBuilder(name, path).with_signature(signature)


def test_model_json_parser_for_internal_models(cluster: Cluster, modelversion_json: dict):
    modelversion_json["id"] = 420
    mv: ModelVersion = ModelVersion._from_json(cluster, modelversion_json)
    assert mv.id == 420


def test_model_json_parser_for_external_models(cluster: Cluster, external_modelversion_json: dict):
    external_modelversion_json["id"] = 420
    mv: ModelVersion = ModelVersion._from_json(cluster, external_modelversion_json)
    assert mv.id == 420


def test_list_models_by_model_name(cluster: Cluster, runtime: DockerImage, 
                                   payload: list, signature: ModelSignature):
    def create_model_version_builder(name: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'resources/identity_model/')
        return ModelVersionBuilder(name, model_path) \
            .with_runtime(runtime) \
            .with_payload(payload) \
            .with_signature(signature)

    name1 = f"{config.default_model_name}-one-{random.randint(0, 1e5)}"
    name2 = f"{config.default_model_name}-two-{random.randint(0, 1e5)}"
    mv1: ModelVersion = create_model_version_builder(name1).build(cluster)
    mv2: ModelVersion = create_model_version_builder(name1).build(cluster)
    _ = create_model_version_builder(name2).build(cluster)
    
    mvs = ModelVersion.find_by_model_name(cluster, name1)
    assert len(mvs) == 2
    # test sorting
    assert mvs[0].id == mv1.id
    assert mvs[1].id == mv2.id

from hydro_serving_grpc.monitoring.sonar import entities_pb2
class Mock:
    def Analyze(c, a):
        entities_pb2.ExecutionInformation.SerializeToString(a)
        pass

def test_model_analyze_response():
    cl = Cluster("asdasd:9091", "asdasd:9090", check_connection=False)
    model = ModelVersion(
        cluster=cl,
        id=1,
        model_id=2,
        name="test",
        version=3,
        signature=ModelSignature(),
        status = None,
        image = "",
        runtime="",
        is_external=True
    )
    req_id = "asdsad"
    request = PredictRequest(
        inputs = {
            "a": Tensor()
        }
    )
    response = PredictResponse(
        outputs = {
            "b": Tensor()
        }
    )
    res = _analyze(model, Mock(), req_id, request, response)
    assert res.metadata.request_id == req_id
    assert res.metadata.model_version_id == 1
    assert res.metadata.model_name == "test"
    assert res.metadata.model_version == 3 
    assert res.error == ""
    assert res.response == response

def test_model_analyze_without_grpc():
    cl = Cluster("asdasd:9091", check_connection=False)
    model = ModelVersion(
        cluster=cl,
        id=1,
        model_id=2,
        name="test",
        version=3,
        signature=ModelSignature(),
        status = None,
        image = "",
        runtime="",
        is_external=True
    )
    req_id = "asdsad"
    request = PredictRequest()
    error = "error" 
    with pytest.raises(HydrosphereException):
        model.analyze(request_id=req_id, request = request, error = error)

def test_model_analyze_error():
    cl = Cluster("asdasd:9091", "asdasd:9090", check_connection=False)
    model = ModelVersion(
        cluster=cl,
        id=1,
        model_id=2,
        name="test",
        version=3,
        signature=ModelSignature(),
        status = None,
        image = "",
        runtime="",
        is_external=True
    )
    req_id = "asdsad"
    request = PredictRequest()
    error = "error"
    res = _analyze(model, Mock(), req_id, request, error=error)
    assert res.metadata.request_id == req_id
    assert res.metadata.model_version_id == 1
    assert res.metadata.model_name == "test"
    assert res.metadata.model_version == 3 
    assert res.error == "error" 
    assert res.response == PredictResponse()

def test_external_model_parse():
    import json
    json_str = """
    {
    "id": 138,
    "created": "2021-09-24T11:29:09.624Z",
    "finished": "2021-09-24T11:29:09.624Z",
    "modelVersion": 1,
    "modelSignature": {
        "signatureName": "predict",
        "inputs": [{
            "name": "in",
            "dtype": "DT_DOUBLE",
            "shape": {
                "dims": []
            },
            "profile": "NONE"
        }],
        "outputs": [{
            "name": "out",
            "dtype": "DT_DOUBLE",
            "shape": {
                "dims": []
            },
            "profile": "NONE"
        }]
    },
    "model": {
        "id": 18,
        "name": "external-model"
    },
    "status": "Released",
    "metadata": {},
    "applications": [],
    "image": null,
    "runtime": null,
    "monitoringConfiguration": {
        "batchSize": 100
    },
    "isExternal": true
    }"""
    json_dict = json.loads(json_str)
    cl = Cluster("asdasd:9091", "asdasd:9090", check_connection=False)

    mv = ModelVersion._from_json(cl, json_dict)
    print(mv)
    assert False