import os
import time
from random import random

import numpy as np
import pytest
from hydro_serving_grpc.contract import ModelContract
from pandas import DataFrame

from hydrosdk.application import ApplicationStatus
from hydrosdk.contract import SignatureBuilder
from hydrosdk.data.types import PredictorDT
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import LocalModel
from hydrosdk.servable import Servable
from tests.test_application import create_test_application
from tests.test_modelversion import create_test_cluster, create_test_local_model, create_test_signature


# TODO: add servable Unmonitored tests

@pytest.fixture
def tensor_servable():
    cluster = create_test_cluster()

    signature = create_test_signature()

    contract = ModelContract(predict=signature)

    model = create_test_local_model(contract=contract)

    upload_resp = model.upload(cluster=cluster)

    servable = Servable.create(model_name=upload_resp[model].modelversion.name,
                               version=upload_resp[model].modelversion.version, cluster=cluster)

    # wait for servable to assemble
    time.sleep(20)
    return servable


@pytest.fixture
def scalar_servable():
    cluster = create_test_cluster()

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

    upload_resp = local_model.upload(cluster)

    servable = Servable.create(model_name=upload_resp[local_model].modelversion.name,
                               version=upload_resp[local_model].modelversion.version,
                               cluster=cluster)

    # wait for servable to assemble
    time.sleep(20)

    return servable


def test_predict_application():
    cluster = create_test_cluster()
    model = create_test_local_model()

    upload_response = model.upload(cluster=cluster)

    tensor_application = create_test_application(cluster=cluster, upload_response=upload_response, local_model=model)

    value = int(random() * 1e5)

    while tensor_application.status != ApplicationStatus.READY:
        tensor_application.update_status()

    for servable in Servable.list(cluster=cluster):
        if upload_response[model].modelversion.version == servable.modelversion.version:
            # TODO: Add to servables status and then del sleep
            time.sleep(20)
            break

    predictor_client = tensor_application.predictor(return_type=PredictorDT.DICT_PYTHON)

    inputs = {'input': [value]}

    predictions = predictor_client.predict(inputs)

    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], list)
    assert predictions['output'] == [value]


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
    assert isinstance(predictions['output'], np.int64)
    assert predictions['output'] == value


def test_predict_df(tensor_servable):
    value = int(random() * 1e5)

    predictor_client = tensor_servable.predictor(return_type=PredictorDT.DF)
    inputs_dict = {'input': [value]}
    inputs_df = DataFrame(inputs_dict)
    predictions = predictor_client.predict(inputs=inputs_df)

    assert isinstance(predictions, DataFrame)
    assert predictions.equals(DataFrame({'output': [value]}))
