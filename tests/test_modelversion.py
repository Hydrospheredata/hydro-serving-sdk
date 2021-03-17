import os
import random

import pytest
from hydro_serving_grpc.serving.contract.tensor_pb2 import TensorShape
from hydro_serving_grpc.serving.contract.types_pb2 import DataType

from hydrosdk.cluster import Cluster
from hydrosdk.signature import ModelField, ModelSignature, SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import ModelVersion, ModelVersionStatus, LocalModel, \
    MonitoringConfiguration, resolve_paths
from hydrosdk.monitoring import ThresholdCmpOp
from hydrosdk.exceptions import SignatureViolationException
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
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './requirements.txt',
        './data/*',
        './model/snapshot.proto'
    ]
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], ProfilingType.NUMERICAL) \
        .with_output('out1', 'double', [-1], ProfilingType.NUMERICAL).build()
    monitoring_configuration = MonitoringConfiguration(batch_size=10)
    metadata = {"key": "value"}
    install_command = "pip install -r requirements.txt"
    training_data = "s3://bucket/path/to/training-data"
    local_model = LocalModel(name, runtime, path, payload, signature, metadata, 
                             install_command, training_data, monitoring_configuration)
    assert local_model.name == name
    assert local_model.runtime == runtime
    assert local_model.path == path
    assert list(local_model.payload.values()) == payload
    assert local_model.signature == signature
    assert local_model.monitoring_configuration == monitoring_configuration


def test_model_create_signature_validation():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './requirements.txt',
        './data/*',
        './model/snapshot.proto'
    ]
    signature = SignatureBuilder('infer').build()
    metadata = {"key": "value"}
    install_command = "pip install -r requirements.txt"
    training_data = "s3://bucket/path/to/training-data"
    with pytest.raises(SignatureViolationException):
        local_model = LocalModel(name, runtime, path, payload, signature, metadata, 
                                install_command, training_data)


def test_local_model_upload(cluster: Cluster):
    name = DEFAULT_MODEL_NAME
    payload = ['./src/func_main.py']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], ProfilingType.NONE) \
        .with_output('out1', 'double', [-1], ProfilingType.NONE).build()
    batch_size = 10
    monitoring_configuration = MonitoringConfiguration(batch_size=batch_size)
    metadata = {"key": "value"}
    local_model = LocalModel(name, runtime, model_path, payload, signature, 
        metadata, monitoring_configuration=monitoring_configuration)
    mv: ModelVersion = local_model.upload(cluster)
    assert mv.status is ModelVersionStatus.Assembling
    mv.lock_till_released(timeout=LOCK_TIMEOUT)
    assert mv.status is ModelVersionStatus.Released
    assert mv.monitoring_configuration.batch_size == batch_size


def test_modelversion_find_by_id(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv_found: ModelVersion = ModelVersion.find_by_id(cluster, mv.id)
    assert mv.id == mv_found.id


def test_modelversion_find(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv_found: ModelVersion = ModelVersion.find(cluster, mv.name, mv.version)
    assert mv.id == mv_found.id


def test_lock_till_released_failed(cluster: Cluster, runtime: DockerImage, 
                                   payload: list, signature: ModelSignature):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    local_model = LocalModel(
        DEFAULT_MODEL_NAME, runtime, model_path, payload, signature, install_command="exit 1")
    mv: ModelVersion = local_model.upload(cluster)
    with pytest.raises(ModelVersion.ReleaseFailed):
        mv.lock_till_released(timeout=LOCK_TIMEOUT)
        

def test_build_logs_not_empty(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released(timeout=LOCK_TIMEOUT)
    i = 0
    for _ in mv.build_logs():
        i += 1
    assert i > 0


def test_modelversion_list(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    assert mv.id in [modelversion.id for modelversion in ModelVersion.list(cluster)]

    
def test_ModelField_dt_invalid_input():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(
        signature_name="test", 
        inputs=[ModelField(name="input", shape=TensorShape())],
        outputs=[ModelField(name="output", shape=TensorShape(), dtype=DataType.Name(2))]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model with invalid dtype in the signature input.*"):
        LocalModel(name, runtime, path, payload, signature)


def test_ModelField_dt_invalid_output():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(
        signature_name="test",
        inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())],
        outputs=[ModelField(name="test", shape=TensorShape())]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model with invalid dtype in the signature output.*"):
        LocalModel(name, runtime, path, payload, signature)


def test_ModelField_contact_signature_name_none():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(
        inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())],
        outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShape())]
    )
    with pytest.raises(SignatureViolationException, match=r"Creating model without signature_name is not allowed.*"):
        LocalModel(name, runtime, path, payload, signature)


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
    def create_local_model(name: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'resources/identity_model/')
        return LocalModel(name, runtime, model_path, payload, signature)

    name1 = f"{DEFAULT_MODEL_NAME}-one-{random.randint(0, 1e5)}"
    name2 = f"{DEFAULT_MODEL_NAME}-two-{random.randint(0, 1e5)}"
    mv1: ModelVersion = create_local_model(name1).upload(cluster)
    mv2: ModelVersion = create_local_model(name1).upload(cluster)
    mv3: ModelVersion = create_local_model(name2).upload(cluster)
    
    mvs = ModelVersion.find_by_model_name(cluster, name1)
    assert len(mvs) == 2
    # test sorting
    assert mvs[0].id == mv1.id
    assert mvs[1].id == mv2.id
