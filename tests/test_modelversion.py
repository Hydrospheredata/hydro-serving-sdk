import os
import random

import pytest
from hydro_serving_grpc import TensorShapeProto, DataType

from hydrosdk.cluster import Cluster
from hydrosdk.contract import ModelContract, ModelField, ModelSignature, SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import ModelVersion, ModelVersionStatus, LocalModel, resolve_paths
from hydrosdk.monitoring import ThresholdCmpOp
from hydrosdk.exceptions import ContractViolationException
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
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './requirements.txt',
        './data/*',
        './model/snapshot.proto'
    ]
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()
    contract = ModelContract(predict=signature)
    metadata = {"key": "value"}
    install_command = "pip install -r requirements.txt"
    training_data = "s3://bucket/path/to/training-data"
    local_model = LocalModel(name, runtime, path, payload, contract, metadata, 
                             install_command, training_data)
    assert local_model.name == name
    assert local_model.runtime == runtime
    assert local_model.path == path
    assert list(local_model.payload.values()) == payload
    assert local_model.contract == contract


def test_model_create_contract_validation():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = [
        './src/func_main.py',
        './requirements.txt',
        './data/*',
        './model/snapshot.proto'
    ]
    signature = SignatureBuilder('infer').build()
    contract = ModelContract(predict=signature)
    metadata = {"key": "value"}
    install_command = "pip install -r requirements.txt"
    training_data = "s3://bucket/path/to/training-data"
    with pytest.raises(ContractViolationException):
        local_model = LocalModel(name, runtime, path, payload, contract, metadata, 
                                install_command, training_data)


def test_local_model_upload(cluster: Cluster):
    name = DEFAULT_MODEL_NAME
    payload = ['./src/func_main.py']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/model_1/')
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()
    contract = ModelContract(predict=signature)
    metadata = {"key": "value"}
    local_model = LocalModel(name, runtime, model_path, payload, contract, metadata)
    mv: ModelVersion = local_model.upload(cluster)
    assert mv.status is ModelVersionStatus.Assembling
    mv.lock_till_released()
    assert mv.status is ModelVersionStatus.Released


def test_modelversion_find_by_id(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv_found: ModelVersion = ModelVersion.find_by_id(cluster, mv.id)
    assert mv.id == mv_found.id


def test_modelversion_find(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv_found: ModelVersion = ModelVersion.find(cluster, mv.name, mv.version)
    assert mv.id == mv_found.id


def test_lock_till_released_failed(cluster: Cluster, runtime: DockerImage, 
                                   payload: list, contract: ModelContract):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/model_1/')
    local_model = LocalModel(
        DEFAULT_MODEL_NAME, runtime, model_path, payload, contract, install_command="exit 1")
    mv: ModelVersion = local_model.upload(cluster)
    with pytest.raises(ModelVersion.ReleaseFailed):
        mv.lock_till_released()
        

def test_build_logs_not_empty(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    i = 0
    for _ in mv.build_logs():
        i += 1
    assert i > 0


def test_modelversion_list_all(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    assert mv.id in [modelversion.id for modelversion in ModelVersion.list_all(cluster)]

    
def test_ModelField_dt_invalid_input():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(signature_name="test", inputs=[ModelField(name="test", shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())])
    contract = ModelContract(predict=signature)
    with pytest.raises(ContractViolationException, match=r"Creating model with invalid dtype in contract-input.*"):
        LocalModel(name, runtime, path, payload, contract)


def test_ModelField_dt_invalid_output():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(signature_name="test",
                               inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", shape=TensorShapeProto())])
    contract = ModelContract(predict=signature)
    with pytest.raises(ContractViolationException, match=r"Creating model with invalid dtype in contract-output.*"):
        LocalModel(name, runtime, path, payload, contract)


def test_ModelField_contract_predict_None():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = []
    contract = ModelContract(predict=None)
    with pytest.raises(ContractViolationException, match=r"Creating model without contract.predict is not allowed.*"):
        LocalModel(name, runtime, path, payload, contract)


def test_ModelField_contact_signature_name_none():
    name = DEFAULT_MODEL_NAME
    runtime = DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None)
    path = "/home/user/folder/model/cool/"
    payload = []
    signature = ModelSignature(inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())])
    contract = ModelContract(predict=signature)
    with pytest.raises(ContractViolationException, match=r"Creating model without contract.predict.signature_name is not allowed.*"):
        LocalModel(name, runtime, path, payload, contract)


def test_model_json_parser_for_internal_models(cluster: Cluster, modelversion_json: dict):
    modelversion_json["id"] = 420
    mv: ModelVersion = ModelVersion._from_json(cluster, modelversion_json)
    assert mv.id == 420


def test_model_json_parser_for_external_models(cluster: Cluster, external_modelversion_json: dict):
    external_modelversion_json["id"] = 420
    mv: ModelVersion = ModelVersion._from_json(cluster, external_modelversion_json)
    assert mv.id == 420


def test_list_models_by_model_name(cluster: Cluster, runtime: DockerImage, 
                                   payload: list, contract: ModelContract):
    def create_local_model(name: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'resources/model_1/')
        return LocalModel(name, runtime, model_path, payload, contract)

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
