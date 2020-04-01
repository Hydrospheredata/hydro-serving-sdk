import os

import pytest
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, resolve_paths, ExternalModel
from hydrosdk.monitoring import TresholdCmpOp
from tests.resources.test_config import CLUSTER_ENDPOINT, PATH_TO_SERVING


def get_payload():
    return {os.path.dirname(os.path.abspath(__file__)) + '/resources/model_1/src/func_main.py': './src/func_main.py'}


def get_cluster():
    return Cluster(CLUSTER_ENDPOINT)


def get_contract():
    return ModelContract(predict=get_signature())


def get_local_model(name="upload-model-test", contract=None, payload=None, path=None):
    if payload is None:
        payload = get_payload()

    if not contract:
        contract = get_contract()

    local_model = LocalModel(
        name=name,
        contract=contract,
        runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
        payload=payload,
        path=path  # build programmatically
    )

    return local_model


def get_ext_model_fields() -> tuple:
    name = "ext-model-test"
    contract = get_contract()
    metadata = {"additionalProp1": "prop"}

    return name, contract, metadata


def get_signature():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()
    return signature


def test_external_model_create():
    cluster = get_cluster()

    name, contract, metadata = get_ext_model_fields()

    created_model = ExternalModel.create(cluster=cluster, name=name, contract=contract, metadata=metadata)
    found_model = ExternalModel.find_by_name(cluster=cluster, name=created_model.name,
                                             version=created_model.version)
    assert found_model


def test_external_model_find_by_name():
    cluster = get_cluster()

    name, contract, metadata = get_ext_model_fields()

    created_model = ExternalModel.create(cluster=cluster, name=name, contract=contract, metadata=metadata)
    found_model = ExternalModel.find_by_name(cluster=cluster, name=created_model.name,
                                             version=created_model.version)

    assert found_model


def test_external_model_delete():
    cluster = get_cluster()

    name, contract, metadata = get_ext_model_fields()

    created_model = ExternalModel.create(cluster=cluster, name=name, contract=contract, metadata=metadata)
    ExternalModel.delete_by_id(cluster=cluster, model_id=created_model.id_)

    with pytest.raises(Exception, match=r"Failed to find Model for name.*"):
        found_model = ExternalModel.find_by_name(cluster=cluster, name=created_model.name,
                                                 version=created_model.version)


def test_local_model_file_deserialization():
    model = LocalModel.from_file(PATH_TO_SERVING)
    assert model is not None


def test_model_find_in_cluster():
    # mock answer from server
    # check model objects
    cluster = Cluster(CLUSTER_ENDPOINT)
    loc_model = get_local_model()

    upload_response = loc_model._LocalModel__upload(cluster)

    model_by_id = Model.find_by_id(cluster, upload_response.model.id)

    assert model_by_id.id == upload_response.model.id


def test_model_find():
    cluster = Cluster(CLUSTER_ENDPOINT)
    signature = get_signature()

    contract = ModelContract(predict=signature)

    loc_model = get_local_model(contract=contract)
    upload_response = loc_model._LocalModel__upload(cluster)

    model = Model.find(cluster, upload_response.model.name, upload_response.model.version)
    assert model.id == upload_response.model.id


def test_model_create_payload_dict():
    test_model = get_local_model()
    assert test_model.payload == get_payload()


def test_model_create_payload_list():
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]

    path = "/home/user/folder/model/cool/"

    test_model = get_local_model(payload=payload, path=path)

    assert test_model.payload == {'/home/user/folder/model/cool/src/func_main.py': './src/func_main.py',
                                  '/home/user/folder/model/cool/data/*': './data/*',
                                  '/home/user/folder/model/cool/model/snapshot.proto': './model/snapshot.proto'}


def test_model_create_programmatically():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)
    test_model = get_local_model()


def test_local_model_upload():
    # mock answer from server
    # check that correct JSON is sent to cluster

    m1 = get_local_model("linear_regression_1").as_metric(threshold=100, comparator=TresholdCmpOp.GREATER_EQ)
    m2 = get_local_model("linear_regression_2").as_metric(threshold=100, comparator=TresholdCmpOp.LESS_EQ)

    production_model = get_local_model("linear_regression_prod").with_metrics([m1, m2])

    progress = production_model.upload(get_cluster())

    while progress[m1].building():
        pass
    assert progress[m1].ok()


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_fail():
    pass


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_logs_fail():
    pass


# TODO: add asserts, add model
def test_model_list():
    cluster = Cluster(CLUSTER_ENDPOINT)

    name, contract, metadata = get_ext_model_fields()
    created_model = ExternalModel.create(cluster=cluster, name=name, contract=contract, metadata=metadata)

    res_list = Model.list_models(cluster)

    assert res_list


def test_model_delete_by_id():
    cluster = Cluster(CLUSTER_ENDPOINT)
    Model.delete_by_id(cluster, model_id=420)


def test_resolve_paths():
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]
    folder = "/home/user/dev/model/cool/"
    result = resolve_paths(folder, payload)
    assert result['/home/user/dev/model/cool/src/func_main.py'] == './src/func_main.py'
