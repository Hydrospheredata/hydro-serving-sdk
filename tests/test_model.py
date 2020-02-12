import pytest
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, resolve_paths
from hydrosdk.monitoring import CustomModelMetricSpec as metric_spec
from tests.resources.test_config import CLUSTER_ENDPOINT, PATH_TO_SERVING


def get_payload():
    return {'/home/user/folder/src/file.py': './src/file.py'}


def get_cluster():
    return Cluster(CLUSTER_ENDPOINT)


def get_contract():
    return ModelContract(predict=get_signature())


def get_local_model(name, payload=None, path=None):
    if payload is None:
        payload = get_payload()

    local_model = LocalModel(
        name=name,
        contract=get_contract(),
        runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
        payload=payload,
        path=path  # build programmatically
    )

    return local_model


def get_signature():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    return signature


def test_local_model_file_deserialization():
    model = LocalModel.from_file(PATH_TO_SERVING)
    print(model)
    assert model is not None


def test_model_find_in_cluster():
    # mock answer from server
    # check model objects
    cluster = Cluster(CLUSTER_ENDPOINT)
    model = Model.find(cluster, name="test_model", version=1)
    model_by_id = Model.find_by_id(12)


def test_model_create_payload_dict():
    test_model = get_local_model("test_model")
    assert test_model.payload == get_payload()


def test_model_create_payload_list():
    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]

    path = "/home/user/folder/model/cool/"

    test_model = get_local_model("test_model", payload=payload, path=path)

    assert test_model.payload == {'/home/user/folder/model/cool/src/func_main.py': './src/func_main.py',
                                  '/home/user/folder/model/cool/data/*': './data/*',
                                  '/home/user/folder/model/cool/model/snapshot.proto': './model/snapshot.proto'}


def test_model_create_programmatically():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)
    test_model = get_local_model("test_image")


def test_local_model_upload():
    # mock answer from server
    # check that correct JSON is sent to cluster

    m1 = get_local_model("linear_regression_1").as_metric(threshold=100, comparator=metric_spec.LESS_EQ)
    m2 = get_local_model("linear_regression_2").as_metric(threshold=100, comparator=metric_spec.LESS_EQ)

    production_model = get_local_model("linear_regression_prod").with_metrics([m1, m2])

    progress = production_model.upload(get_cluster())

    while progress.building():
        pass

    assert progress.ok()


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_fail():
    pass


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_logs_fail():
    pass


def test_model_list():
    cluster = Cluster(CLUSTER_ENDPOINT)
    res_list = Model.list_models(cluster)


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
    print(result)
    assert result['/home/user/dev/model/cool/src/func_main.py'] == './src/func_main.py'
