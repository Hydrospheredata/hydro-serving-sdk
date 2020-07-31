import random

import pytest
import numpy as np
import pandas as pd

from hydrosdk.data.types import PredictorDT
from hydrosdk.modelversion import LocalModel, ModelVersion
from hydrosdk.cluster import Cluster
from hydrosdk.application import Application, ApplicationBuilder, ExecutionStageBuilder
from tests.common_fixtures import *
from tests.utils import *
from tests.config import *

# TODO: add servable Unmonitored tests

@pytest.fixture
def value(): 
    return random.randint(0, 1e5)


@pytest.fixture(scope="module")
def app_tensor(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    stage = ExecutionStageBuilder().with_model_variant(mv, 100).build()
    app = ApplicationBuilder(cluster, f"{DEFAULT_APP_NAME}-{random.randint(0, 1e5)}") \
        .with_stage(stage).build()
    application_lock_till_ready(cluster, app.name)
    yield app


@pytest.fixture(scope="module")
def app_scalar(cluster: Cluster, scalar_local_model: LocalModel):
    mv: ModelVersion = scalar_local_model.upload(cluster)
    mv.lock_till_released()
    stage = ExecutionStageBuilder().with_model_variant(mv, 100).build()
    app = ApplicationBuilder(cluster, f"{DEFAULT_APP_NAME}-{random.randint(0, 1e5)}") \
        .with_stage(stage).build()
    application_lock_till_ready(cluster, app.name)
    yield app

    
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
