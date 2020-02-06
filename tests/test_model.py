import unittest

from hydro_serving_grpc.contract import ModelContract

from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.model import Model, LocalModel, resolve_paths


PATH_TO_SERVING = "/serving.yaml"
PATH_TO_MODEL = "/model.py"
CLUSTER_HTTP = "http://localhost:80"


class ModelCases(unittest.TestCase):
    def test_local_model_file_deserialization(self):
        model = LocalModel.from_file(PATH_TO_SERVING)
        print(model)
        self.assertIsNotNone(model)

    def test_model_find_in_cluster(self):
        # mock answer from server
        # check model objects
        cluster = Cluster(CLUSTER_HTTP)
        model = Model.find(cluster, name="test-model", version=1)
        model_by_id = Model.find_by_id(12)

    def test_model_create_payload_dict(self):
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

    def test_model_create_payload_list(self):
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

    def test_model_create_programmatically(self):

        signature = SignatureBuilder('infer') \
            .with_input('in1', 'double', [-1, 2], 'numerical') \
            .with_output('out1', 'double', [-1], 'numerical').build()

        contract = ModelContract(predict=signature)

        test_model = LocalModel(
            name="testModel",
            contract=contract,
            runtime=DockerImage("TestImage", "latest", None),
            payload={'/home/user/folder/src/file.py': './src/file.py'},
            path=None # build programmatically
        )

    def test_local_model_upload(self):
        # mock answer from server
        # check that correct JSON is sent to cluster
        cluster = Cluster(CLUSTER_HTTP)
        model = LocalModel.from_file(PATH_TO_SERVING)

        model.deploy(cluster)

    def test_model_list(self):
        cluster = Cluster(CLUSTER_HTTP)
        res_list = Model.list_models(cluster)

    def test_model_delete_by_id(self):
        cluster = Cluster(CLUSTER_HTTP)
        Model.delete_by_id(cluster, model_id=420)


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
