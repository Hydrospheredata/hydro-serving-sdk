import os

import pytest
from hydro_serving_grpc import TensorShapeProto, DataType
from hydro_serving_grpc.contract import ModelContract, ModelField, ModelSignature

from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import ModelVersion, LocalModel, resolve_paths
from hydrosdk.monitoring import TresholdCmpOp
from tests.resources.test_config import HTTP_CLUSTER_ENDPOINT, GRPC_CLUSTER_ENDPOINT, PATH_TO_SERVING


def create_test_payload():
    return {os.path.dirname(os.path.abspath(__file__)) + '/resources/model_1/src/func_main.py': './src/func_main.py'}


def create_test_cluster(http_address=HTTP_CLUSTER_ENDPOINT, grpc_address=GRPC_CLUSTER_ENDPOINT):
    return Cluster(http_address=http_address, grpc_address=grpc_address)


def create_test_contract(signature=None):
    if not signature:
        signature = create_test_signature()
    return ModelContract(predict=signature)


def create_test_local_model(name="upload-model-test", contract=None, payload=None, path=None):
    if payload is None:
        payload = create_test_payload()

    if not contract:
        contract = create_test_contract()

    local_model = LocalModel(
        name=name,
        contract=contract,
        runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
        payload=payload,
        path=path  # build programmatically
    )

    return local_model


def create_test_signature():
    signature = SignatureBuilder('infer') \
        .with_input('input', 'int64', [1], 'numerical') \
        .with_output('output', 'int64', [1], 'numerical').build()
    return signature


def test_local_model_file_deserialization():
    model = LocalModel.from_file(PATH_TO_SERVING)
    assert model is not None


def test_modelversion_find_by_id():
    # mock answer from server
    # check model objects
    cluster = create_test_cluster()
    loc_model = create_test_local_model()

    upload_response = loc_model._LocalModel__upload(cluster)

    modelversion_by_id = ModelVersion.find_by_id(cluster, upload_response.modelversion.id)

    assert modelversion_by_id.id == upload_response.modelversion.id


def test_model_find_by_name_modelversion():
    cluster = create_test_cluster()
    signature = create_test_signature()

    contract = ModelContract(predict=signature)

    loc_model = create_test_local_model(contract=contract)
    upload_response = loc_model._LocalModel__upload(cluster)

    modelversion = ModelVersion.find(cluster, upload_response.modelversion.name, upload_response.modelversion.version)
    assert modelversion.id == upload_response.modelversion.id


def test_model_create_payload_dict():
    test_model = create_test_local_model()
    assert test_model.payload == create_test_payload()


def test_model_create_payload_list():
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]

    path = "/home/user/folder/model/cool/"

    test_model = create_test_local_model(payload=payload, path=path)

    assert test_model.payload == {'/home/user/folder/model/cool/src/func_main.py': './src/func_main.py',
                                  '/home/user/folder/model/cool/data/*': './data/*',
                                  '/home/user/folder/model/cool/model/snapshot.proto': './model/snapshot.proto'}


def test_model_create_programmatically():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)
    test_model = create_test_local_model()


def test_local_model_upload():
    # mock answer from server
    # check that correct JSON is sent to cluster

    m1 = create_test_local_model("linear_regression_1").as_metric(threshold=100, comparator=TresholdCmpOp.GREATER_EQ)
    m2 = create_test_local_model("linear_regression_2").as_metric(threshold=100, comparator=TresholdCmpOp.LESS_EQ)

    production_model = create_test_local_model("linear_regression_prod").with_metrics([m1, m2])

    upload_responses_dict = production_model.upload(create_test_cluster())
    upload_response_obj = upload_responses_dict[production_model]

    assert upload_response_obj


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_fail():
    pass


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_logs_fail():
    pass


# TODO: add asserts, add model
def test_model_list():
    cluster = create_test_cluster()

    local_model = create_test_local_model("linear_regression_prod")

    upload_responses_dict = local_model.upload(create_test_cluster())
    upload_response_obj = upload_responses_dict[local_model]

    res_list = ModelVersion.list_model_versions(cluster)

    assert res_list


def test_ModelField_dt_invalid_input():
    signature = ModelSignature(signature_name="test", inputs=[ModelField(name="test", shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())])

    contract = create_test_contract(signature=signature)

    with pytest.raises(ValueError, match=r"Creating model with invalid dtype in contract-input.*"):
        create_test_local_model(contract=contract)


def test_ModelField_dt_invalid_output():
    signature = ModelSignature(signature_name="test",
                               inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", shape=TensorShapeProto())])

    contract = create_test_contract(signature=signature)

    with pytest.raises(ValueError, match=r"Creating model with invalid dtype in contract-output.*"):
        create_test_local_model(contract=contract)


def test_ModelField_contract_predict_None():
    contract = ModelContract(predict=None)
    with pytest.raises(ValueError, match=r"Creating model without contract.predict is not allowed.*"):
        create_test_local_model(contract=contract)


def test_ModelField_contact_signatue_name_none():
    signature = ModelSignature(inputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())],
                               outputs=[ModelField(name="test", dtype=DataType.Name(2), shape=TensorShapeProto())])

    contract = create_test_contract(signature=signature)

    with pytest.raises(ValueError, match=r"Creating model without contract.predict.signature_name is not allowed.*"):
        create_test_local_model(contract=contract)


def test_modelversion_delete_by_id():
    cluster = create_test_cluster()

    local_model = create_test_local_model("linear_regression_prod")

    upload_responses_dict = local_model.upload(create_test_cluster())
    upload_response_obj = upload_responses_dict[local_model]

    ModelVersion.delete_by_model_id(cluster=cluster, model_id=upload_response_obj.modelversion.model_id)

    with pytest.raises(ModelVersion.NotFound):
        ModelVersion.find(cluster=cluster, name=upload_response_obj.modelversion.name, version=upload_response_obj.modelversion.version)


def test_resolve_paths():
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]
    folder = "/home/user/dev/model/cool/"
    result = resolve_paths(folder, payload)
    assert result['/home/user/dev/model/cool/src/func_main.py'] == './src/func_main.py'


MODEL_JSONS = [
    {'applications': [],
     'model': {'id': 1, 'name': 'ext-model-test'},
     'finished': '2020-04-17T07:02:43.429Z',
     'modelContract': {
         'modelName': '',
         'predict': {
             'signatureName': 'infer',
             'inputs': [
                 {'profile': 'NONE', 'dtype': 'DT_DOUBLE', 'name': 'in1', 'shape': {
                     'dim': [{'size': -1, 'name': ''},
                             {'size': 2, 'name': ''}],
                     'unknownRank': False}}],
             'outputs': [
                 {'profile': 'NONE', 'dtype': 'DT_DOUBLE', 'name': 'out1', 'shape': {
                     'dim': [
                         {'size': -1, 'name': ''}],
                     'unknownRank': False}}]}
     },
     'isExternal': True,
     'id': 3,
     'status': 'Released',
     'metadata': {'additionalProp1': 'prop'},
     'modelVersion': 3,
     'created': '2020-04-17T07:02:43.429Z'
     },
    {'applications': [],
     'model': {'id': 1, 'name': 'ext-model-test'},
     'finished': '2020-04-17T07:02:43.429Z',
     'modelContract': {
         'modelName': '',
         'predict': {
             'signatureName': 'infer',
             'inputs': [
                 {'profile': 'NONE', 'dtype': 'DT_DOUBLE', 'name': 'in1', 'shape': {
                     'dim': [{'size': -1, 'name': ''},
                             {'size': 2, 'name': ''}],
                     'unknownRank': False}}],
             'outputs': [
                 {'profile': 'NONE', 'dtype': 'DT_DOUBLE', 'name': 'out1', 'shape': {
                     'dim': [
                         {'size': -1, 'name': ''}],
                     'unknownRank': False}}]}
     },
     'id': 3,
     'metadata': {'additionalProp1': 'prop'},
     'modelVersion': 3,
     'created': '2020-04-17T07:02:43.429Z'
     }
]


@pytest.mark.parametrize('input', MODEL_JSONS)
@pytest.mark.parametrize('cluster', [Cluster(HTTP_CLUSTER_ENDPOINT)])
def test_model_json_parser(cluster, input):
    result = ModelVersion.from_json(cluster=cluster, model_version=input)
    assert result


def test_list_models_by_model_name():
    cluster = create_test_cluster()

    # make sure you don't have existing models named same
    loc_model = create_test_local_model(name="model-one")
    loc_model2 = create_test_local_model(name="model-two")
    loc_model3 = create_test_local_model(name="model-one")

    upload_response1 = loc_model._LocalModel__upload(cluster)
    upload_response2 = loc_model2._LocalModel__upload(cluster)
    upload_response3 = loc_model3._LocalModel__upload(cluster)

    found_modelversions = ModelVersion.list_models_by_model_name(cluster, upload_response1.modelversion.name)

    assert found_modelversions
    assert len(found_modelversions) == 2
    # test sorting
    assert found_modelversions[0].id == upload_response1.modelversion.id
    assert found_modelversions[1].id == upload_response3.modelversion.id
