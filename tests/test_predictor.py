import os
import time
from random import random

import numpy as np
import pytest
from hydro_serving_grpc.contract import ModelContract
from pandas import DataFrame

from hydrosdk.contract import SignatureBuilder
from hydrosdk.data.types import PredictorDT
from hydrosdk.image import DockerImage
from hydrosdk.model import LocalModel
from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model, get_signature


@pytest.fixture
def tensor_servable():
    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()

    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    servable = Servable.create(model_name=upload_resp[model].model.name,
                               model_version=upload_resp[model].model.version, cluster=http_cluster,
                               grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)
    return servable


@pytest.fixture
def scalar_servable():
    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = SignatureBuilder('infer') \
        .with_input('input', 'int64', "scalar", 'numerical') \
        .with_output('output', 'int64', "scalar", 'numerical').build()

    payload = {os.path.dirname(os.path.abspath(__file__)) + '/resources/scalar_identity_model/src/func_main.py': './src/func_main.py'}

    contract = ModelContract(predict=signature)

    local_model = LocalModel(
        name="scalar_model",
        contract=contract,
        runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
        payload=payload,
        path=None
    )

    upload_resp = local_model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    servable = Servable.create(model_name=upload_resp[local_model].model.name,
                               model_version=upload_resp[local_model].model.version,
                               cluster=http_cluster,
                               grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)

    return servable


# TODO: Add more valid assert
def test_predict_list(tensor_servable):
    value = int(random() * 1e5)
    predictor_client = tensor_servable.predictor(return_type=PredictorDT.DICT_PYTHON)

    inputs = {'input': [value]}
    predictions = predictor_client.predict(inputs)

    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], list)
    assert predictions['output'] == [value]


def test_predict_nparray(tensor_servable):
    value = int(random() * 1e5)

    predictor_client = tensor_servable.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    inputs = {'input': np.array([value])}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], np.ndarray)
    assert predictions['output'] == np.array([value])


def test_predict_np_scalar_type(scalar_servable):
    value = np.int(random() * 1e5)

    predictor_client = scalar_servable.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    inputs = {'input': value}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], np.ScalarType)
    assert predictions['output'] == value


def test_predict_python_scalar_type(scalar_servable):
    value = int(random() * 1e5)

    predictor_client = scalar_servable.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    inputs = {'input': value}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], int)
    assert predictions['output'] == value


def test_predict_df(tensor_servable):
    value = int(random() * 1e5)

    predictor_client = tensor_servable.predictor(return_type=PredictorDT.DF)
    inputs_dict = {'input': [value]}
    inputs_df = DataFrame(inputs_dict)
    predictions = predictor_client.predict(inputs=inputs_df)

    assert isinstance(predictions, DataFrame)
    assert predictions.equals(DataFrame({'output': [value]}))
