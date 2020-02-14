import pytest

from hydrosdk.cluster import Cluster
from hydrosdk.model import LocalModel
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, TresholdCmpOp
from tests.resources.test_config import CLUSTER_ENDPOINT, PATH_TO_SERVING


@pytest.fixture
def cluster():
    return Cluster(CLUSTER_ENDPOINT)


def test_create(cluster):
    model = LocalModel.from_file(PATH_TO_SERVING)
    model1 = model.upload(cluster)
    model2 = model.upload(cluster)
    ms_config = MetricSpecConfig(model2.model_version_id, 10, TresholdCmpOp.NOT_EQ)
    result = MetricSpec.create(cluster, "test", model1.model_version_id, ms_config)
    assert isinstance(result, MetricSpec)
    assert result.name == "test"
    assert result.cluster == cluster
    assert result.model_version_id == model1.model_version_id


def test_list_all(cluster):
    result = MetricSpec.list_all(cluster)
    print(result)
    assert result
    assert isinstance(result[0], MetricSpec)


def test_list_for_model_verison(cluster):
    result = MetricSpec.list_for_model(cluster, 3)
    print(result)
    assert result
    assert isinstance(result[0], MetricSpec)


def test_delete(cluster):
    result = MetricSpec.list_for_model(cluster, 3)[0]
    result.delete()
