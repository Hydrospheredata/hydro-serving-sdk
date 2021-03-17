import os

import pytest
from grpc import ssl_channel_credentials

from hydrosdk.cluster import Cluster
from hydrosdk.deployment_configuration import DeploymentConfigurationBuilder
from hydrosdk.signature import SignatureBuilder, ProfilingType
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import LocalModel, MonitoringConfiguration
from tests.config import *


@pytest.fixture(scope="session")
def cluster():
    if GRPC_CLUSTER_ENDPOINT_SSL:
        credentials = ssl_channel_credentials()
        return Cluster(HTTP_CLUSTER_ENDPOINT, GRPC_CLUSTER_ENDPOINT, ssl=True, grpc_credentials=credentials)
    else:
        return Cluster(HTTP_CLUSTER_ENDPOINT, GRPC_CLUSTER_ENDPOINT)


@pytest.fixture(scope="session")
def signature():
    return SignatureBuilder('infer') \
        .with_input('input', 'int64', 'scalar', ProfilingType.NUMERICAL) \
        .with_output('output', 'int64', 'scalar', ProfilingType.NUMERICAL).build()


@pytest.fixture(scope="session")
def payload():
    return ['./src/func_main.py']


@pytest.fixture(scope="session")
def runtime():
    return DockerImage(DEFAULT_RUNTIME_IMAGE, DEFAULT_RUNTIME_TAG, None)


@pytest.fixture(scope="session")
def monitoring_configuration():
    return MonitoringConfiguration(batch_size=10)


@pytest.fixture(scope="session")
def local_model(payload, signature, runtime, monitoring_configuration):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    return LocalModel(DEFAULT_MODEL_NAME, runtime, model_path, 
        payload, signature, monitoring_configuration=monitoring_configuration)


@pytest.fixture(scope="session")
def tensor_local_model(payload, runtime, monitoring_configuration):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'resources/identity_model/')
    signature = SignatureBuilder('infer') \
        .with_input('input', 'int64', [1], ProfilingType.NONE) \
        .with_output('output', 'int64', [1], ProfilingType.NONE).build()
    return LocalModel(DEFAULT_MODEL_NAME, runtime, model_path, 
        payload, signature, monitoring_configuration=monitoring_configuration)


@pytest.fixture(scope="session")
def modelversion_json():
    return {
        "id": 1,
        "created": "2021-03-17T13:12:20.680Z",
        "finished": "2021-03-17T13:12:24.318Z",
        "modelVersion": 1,
        "modelSignature": {
            "signatureName": "infer",
            "inputs": [{
                "name": "input",
                "dtype": "DT_INT64",
                "shape": {
                    "dims": []
                },
                "profile": "NUMERICAL"
            }],
            "outputs": [{
                "name": "output",
                "dtype": "DT_INT64",
                "shape": {
                    "dims": []
                },
                "profile": "NUMERICAL"
            }]
        },
        "model": {
            "id": 1,
            "name": "my-model"
        },
        "status": "Released",
        "metadata": {},
        "applications": [],
        "image": {
            "name": "my-model",
            "tag": "1",
            "sha256": "1c714d62b9450b2c3467d67855e443c2fe61c6718733ead3d4af89c7ed4515c4"
        },
        "runtime": {
            "name": "hydrosphere/serving-runtime-python-3.6",
            "tag": "2.4.0",
            "sha256": None,
        },
        "monitoringConfiguration": {
            "batchSize": 10
        },
        "isExternal": False,
    }


@pytest.fixture(scope="session")
def external_modelversion_json():
    return {
        "id": 2,
        "created": "2021-03-17T13:12:20.680Z",
        "finished": "2021-03-17T13:12:24.318Z",
        "modelVersion": 1,
        "modelSignature": {
            "signatureName": "infer",
            "inputs": [{
                "name": "input",
                "dtype": "DT_INT64",
                "shape": {
                    "dims": []
                },
                "profile": "NUMERICAL"
            }],
            "outputs": [{
                "name": "output",
                "dtype": "DT_INT64",
                "shape": {
                    "dims": []
                },
                "profile": "NUMERICAL"
            }]
        },
        "model": {
            "id": 1,
            "name": "my-external-model"
        },
        "status": "Released",
        "metadata": {},
        "applications": [],
        "image": {
            "name": "my-external-model",
            "tag": "1",
            "sha256": "1c714d62b9450b2c3467d67855e443c2fe61c6718733ead3d4af89c7ed4515c4"
        },
        "runtime": {
            "name": "hydrosphere/serving-runtime-python-3.6",
            "tag": "2.4.0",
            "sha256": None,
        },
        "monitoringConfiguration": {
            "batchSize": 10
        },
        "isExternal": True,
    }
