import unittest

from hydrosdk.model import LocalModel, resolve_paths


class ModelCases(unittest.TestCase):
    def test_local_model_file_deserialization(self):
        model = LocalModel.from_file("./resources/model_1/serving.yaml")
        self.assertIsNotNone(model)

    def test_local_model_upload(self):
        pass

    def test_model_find(self):
        pass

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