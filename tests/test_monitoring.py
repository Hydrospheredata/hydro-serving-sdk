import time

import pytest

from hydrosdk.cluster import Cluster
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, TresholdCmpOp
from tests.resources.test_config import HTTP_CLUSTER_ENDPOINT
from tests.test_modelversion import create_test_local_model


@pytest.fixture
def cluster():
    return Cluster(HTTP_CLUSTER_ENDPOINT)


def test_create(cluster):
    model1 = create_test_local_model()
    model2 = create_test_local_model()
    upload_resp1 = model1.upload(cluster)
    upload_resp2 = model2.upload(cluster)

    ms_config = MetricSpecConfig(upload_resp2[model2].modelversion.id, 10, TresholdCmpOp.NOT_EQ)
    result = MetricSpec.create(cluster, "test", upload_resp1[model1].modelversion.id, ms_config)
    assert isinstance(result, MetricSpec)
    assert result.name == "test"
    assert result.cluster == cluster
    assert result.model_version_id == upload_resp1[model1].modelversion.id


def test_list_all(cluster):
    result = MetricSpec.list_all(cluster)
    assert result
    assert isinstance(result[0], MetricSpec)


# FIXME: assert []
def test_list_for_model_verison(cluster):
    result = MetricSpec.list_for_model(cluster, 3)
    assert result
    assert isinstance(result[0], MetricSpec)


# FIXME: IndexError: list index out of range
def test_delete(cluster):
    result = MetricSpec.list_for_model(cluster, 3)[0]
    result.delete()
