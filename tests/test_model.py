import pytest
from hydro_serving_grpc.contract import ModelContract
from .config import CLUSTER_HTTP
from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, resolve_paths

PATH_TO_SERVING = "./tests/resources/model_1/serving.yaml"

def test_local_model_file_deserialization():
    model = LocalModel.from_file(PATH_TO_SERVING)
    print(model)
    assert model is not None


def test_model_find_in_cluster():
    # mock answer from server
    # check model objects
    cluster = Cluster(CLUSTER_HTTP)
    model = Model.find(cluster, name="test-model", version=1)
    model_by_id = Model.find_by_id(12)


def test_model_create_payload_dict():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)

    test_model = LocalModel(
        name="TestModel",
        contract=contract,
        runtime=DockerImage("TestImage", "latest", None),
        payload={'/home/user/folder/src/file.py': './src/file.py'},
        path=None  # build programmatically
    )
    assert test_model.payload == {'/home/user/folder/src/file.py': './src/file.py'}


def test_model_create_payload_list():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)

    payload = [
        './src/func_main.py',
        './data/*',
        './model/snapshot.proto'
    ]
    path = "/home/user/folder/model/cool/"

    test_model = LocalModel(
        name="TestModel",
        contract=contract,
        runtime=DockerImage("TestImage", "latest", None),
        payload=payload,
        path=path  # build programmatically
    )
    assert test_model.payload == {'/home/user/folder/model/cool/src/func_main.py': './src/func_main.py',
                                  '/home/user/folder/model/cool/data/*': './data/*',
                                  '/home/user/folder/model/cool/model/snapshot.proto': './model/snapshot.proto'}


def test_model_create_programmatically():
    signature = SignatureBuilder('infer') \
        .with_input('in1', 'double', [-1, 2], 'numerical') \
        .with_output('out1', 'double', [-1], 'numerical').build()

    contract = ModelContract(predict=signature)

    test_model = LocalModel(
        name="testModel",
        contract=contract,
        runtime=DockerImage("TestImage", "latest", None),
        payload={'/home/user/folder/src/file.py': './src/file.py'},
        path=None  # build programmatically
    )


def test_local_model_upload():
    import hydrosdk.monitroing as m

    # mock answer from server
    # check that correct JSON is sent to cluster
    cluster = Cluster(CLUSTER_HTTP)

    m1 = LocalModel.from_file("../m1").as_metric(threshold=100, cmp=m.LESS_EQ)
    m2 = LocalModel.from_file("../m2").as_metric(threshold=100, cmp=m.EQ)

    production_model = LocalModel.from_file("../prod").with_metrics([m1, m2])
    progress = production_model.upload(cluster)

    all_logs = list(progress.logs())  # NB check list correctness
    assert all_logs == []
    assert all_logs is not None

    status = progress.wait()

    assert status.ok


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_fail():
    pass


@pytest.mark.skip("IMPLEMENT LATER")
def test_upload_logs_fail():
    pass


def test_model_list():
    cluster = Cluster(CLUSTER_HTTP)
    res_list = Model.list_models(cluster)


def test_model_delete_by_id():
    cluster = Cluster(CLUSTER_HTTP)
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
