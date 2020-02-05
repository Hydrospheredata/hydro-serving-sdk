import unittest

from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, resolve_paths


class ModelCases(unittest.TestCase):
    def test_local_model_file_deserialization(self):
        model = LocalModel.from_file("./resources/model_1/serving.yaml")
        print(model)
        self.assertIsNotNone(model)

    def test_model_find_in_cluster(self):
        # mock answer from server
        # check model objects
        cluster = Cluster("https://kek")
        model = Model.find(cluster, name="test-model", version=1)
        model_by_id = Model.find_by_id(12)

    def test_model_create_programmatically(self):
        contract = SignatureBuilder("predict")\
            .with_input("asdasd", "adsas", [])\
            .with_input("asdasd", "adsas", [])\
            .with_output("asdasd", "adsas", [])
        model = LocalModel(
            name="name",
            contract=contract,
            runtime=DockerImage("asdasd", "latest", None),
            payload=[
                "./test_model.py"
            ],
            path=None # build programmatically
        )

    def test_local_model_upload(self):
        # mock answer from server
        # check that correct JSON is sent to cluster
        cluster = Cluster("https://kek")
        model = LocalModel.from_file("./resources/model_1/serving.yaml")
        model.deploy(cluster)

    def test_model_list(self):
        cluster = Cluster("https://kek")
        list = Model.list(cluster)

    def test_model_delete(self):
        cluster = Cluster("https://kek")
        Model.delete(cluster, name="test-model", version=1)
        Model.delete_by_id(cluster, id=420)
        model = Model.find_by_id(cluster, id=123)
        model.delete()

    def test_resolve_paths(self):
        payload = [
            './src/func_main.py',
            './data/*',
            './model/snapshot.proto'
        ]
        folder = "/home/user/dev/model/cool/"
        result = resolve_paths(folder, payload)
        print(result)
        self.assertEqual(result['/home/user/dev/model/cool/src/func_main.py'], './src/func_main.py')
