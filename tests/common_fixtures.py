import os

import pytest
from grpc import ssl_channel_credentials

from hydrosdk.contract import ModelContract, SignatureBuilder, ProfilingType
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import LocalModel
from tests.config import *
from tests.utils import *


@pytest.fixture(scope="session")
def cluster():
    if GRPC_CLUSTER_ENDPOINT_SSL:
        credentials = ssl_channel_credentials()
        return Cluster(HTTP_CLUSTER_ENDPOINT, GRPC_CLUSTER_ENDPOINT, ssl=True, grpc_credentials=ssl_channel_credentials())
    else:
        return Cluster(HTTP_CLUSTER_ENDPOINT, GRPC_CLUSTER_ENDPOINT)


@pytest.fixture(scope="session")
def signature():
    return SignatureBuilder('infer') \
        .with_input('input', 'int64', 'scalar', ProfilingType.NUMERICAL) \
        .with_output('output', 'int64', 'scalar', ProfilingType.NUMERICAL).build()


@pytest.fixture(scope="session")
def contract(signature):
    return ModelContract(predict=signature)


@pytest.fixture(scope="session")
def payload():
    return ['./src/func_main.py']


@pytest.fixture(scope="session")
def runtime():
    return DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)


@pytest.fixture(scope="session")
def local_model(payload, contract, runtime):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    return LocalModel(DEFAULT_MODEL_NAME, runtime, model_path, payload, contract)


@pytest.fixture(scope="session")
def tensor_local_model(payload, runtime):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    signature = SignatureBuilder('infer') \
        .with_input('input', 'int64', [1], ProfilingType.NONE) \
        .with_output('output', 'int64', [1], ProfilingType.NONE).build()
    contract = ModelContract(predict=signature)
    return LocalModel(DEFAULT_MODEL_NAME, runtime, model_path, payload, contract)


@pytest.fixture(scope="session")
def modelversion_json():
    return {
        "applications": [],
        "model": {
            "id": 1,
            "name": "my-model"
        },
        "image": {
            "name": "registry/my-model",
            "tag": "1",
            "sha256": "7e28dfe693edaee29c57124e9cf01da80089f4e0408eb072c5222e1d2c3a8e7b"
        },
        "finished": "2020-02-20T12:19:23.240Z",
        "modelContract": {
            "modelName": "model",
            "predict": {
                "signatureName": "infer",
                "inputs": [
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_INT32",
                        "name": "input_1",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    },
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_INT32",
                        "name": "input_2",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    }
                ],
                "outputs": [
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_DOUBLE",
                        "name": "output",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    }
                ]
            }
        },
        "isExternal": False,
        "id": 1,
        "status": "Released",
        "metadata": {
            "git.branch.head.date": "Mon Nov 25 13:16:13 2019",
            "git.branch.head.sha": "4a9e1ef5e32b5d76b0cd3659090de08c1d8308d0",
            "git.branch": "master",
            "git.branch.head.author.name": "Your Name",
            "git.is-dirty": "False",
            "git.branch.head.author.email": "your.email@example.com"
        },
        "modelVersion": 1,
        "monitoringConfiguration": {"batchSize": 100},
        "runtime": {
            "name": "hydrosphere/serving-runtime-python-3.6",
            "tag": "2.3.2"
        },
        "created": "2020-02-20T12:18:56.115Z"
    }


@pytest.fixture(scope="session")
def external_modelversion_json():
    return {
        "applications": [],
        "model": {
            "id": 2,
            "name": "my-external-model"
        },
        "finished": "2020-05-22T12:38:05.021Z",
        "modelContract": {
            "modelName": "my-external-model",
            "predict": {
                "signatureName": "predict",
                "inputs": [
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_INT64",
                        "name": "input_1",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    },
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_INT64",
                        "name": "input_2",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    },
                ],
                "outputs": [
                    {
                        "profile": "NUMERICAL",
                        "dtype": "DT_DOUBLE",
                        "name": "output",
                        "shape": {
                            "dim": [],
                            "unknownRank": False
                        }
                    }
                ]
            }
        },
        "isExternal": True,
        "id": 2,
        "status": "Released",
        "metadata": {
            "key": "value"
        },
        "modelVersion": 1,
        "monitoringConfiguration": {"batchSize": 100},
        "created": "2020-05-22T12:38:05.021Z"
    }
