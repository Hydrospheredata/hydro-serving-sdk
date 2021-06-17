import random
import time 

import pytest
import numpy as np
import pandas as pd

from hydrosdk.data.types import PredictorDT
from hydrosdk.modelversion import ModelVersionBuilder, ModelVersion, MonitoringConfiguration
from hydrosdk.cluster import Cluster
from hydrosdk.application import Application, ApplicationBuilder, ExecutionStageBuilder
from tests.common_fixtures import *
from tests.config import *
from tests.test_dataupload import training_data

# TODO: add servable Unmonitored tests

@pytest.fixture
def value(): 
    return random.randint(0, 1e5)


@pytest.yield_fixture(scope="module")
def app_tensor(cluster: Cluster, tensor_model_version_builder: ModelVersionBuilder):
    mv: ModelVersion = tensor_model_version_builder.build(cluster)
    mv.lock_till_released(timeout=LOCK_TIMEOUT)
    stage = ExecutionStageBuilder().with_model_variant(mv, 100).build()
    app = ApplicationBuilder(f"{DEFAULT_APP_NAME}-{random.randint(0, 1e5)}") \
        .with_stage(stage).build(cluster)
    app.lock_while_starting(timeout=LOCK_TIMEOUT)
    time.sleep(5)
    yield app
    Application.delete(cluster, app.name)


@pytest.yield_fixture(scope="module")
def app_scalar(cluster: Cluster, model_version_builder: ModelVersionBuilder, training_data: str):
    model_version_builder.with_monitoring_configuration(MonitoringConfiguration(batch_size=10))
    mv: ModelVersion = model_version_builder.build(cluster)
    mv.training_data = training_data
    data_upload_response = mv.upload_training_data()
    data_upload_response.wait(sleep=5)
    mv.lock_till_released(timeout=LOCK_TIMEOUT)
    stage = ExecutionStageBuilder().with_model_variant(mv, 100).build()
    app: Application = ApplicationBuilder(f"{DEFAULT_APP_NAME}-{random.randint(0, 1e5)}") \
        .with_stage(stage).build(cluster)
    app.lock_while_starting(timeout=LOCK_TIMEOUT)
    time.sleep(5)
    yield app
    Application.delete(cluster, app.name)

    
def test_predict(app_tensor: Application, value: int):
    predictor = app_tensor.predictor(return_type=PredictorDT.DICT_PYTHON)
    predictions = predictor.predict({"input": [value]})
    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], list)
    assert predictions['output'] == [value]


def test_predict_nparray(app_tensor: Application, value: int):
    predictor = app_tensor.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    predictions = predictor.predict({"input": np.array([value])})
    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], np.ndarray)
    assert predictions['output'] == np.array([value])


def test_predict_df(app_tensor: Application, value: int):
    predictor = app_tensor.predictor(return_type=PredictorDT.DF)
    predictions = predictor.predict(pd.DataFrame({'input': [value]}))
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.equals(pd.DataFrame({'output': [value]}))


def test_predict_np_scalar_type(app_scalar: Application, value: int):
    predictor = app_scalar.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    predictions = predictor.predict({'input': value})
    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], np.ScalarType)
    assert predictions['output'] == value


def test_predict_python_scalar_type(app_scalar: Application, value: int):
    predictor = app_scalar.predictor(return_type=PredictorDT.DICT_NP_ARRAY)
    predictions = predictor.predict({'input': value})
    assert isinstance(predictions, dict)
    assert isinstance(predictions['output'], np.int64)
    assert predictions['output'] == value
