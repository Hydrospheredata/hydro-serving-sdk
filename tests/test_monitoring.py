import pytest

from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion, LocalModel
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, ThresholdCmpOp
from tests.common_fixtures import *


@pytest.fixture(scope="module")
def root_mv(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    return mv


@pytest.fixture(scope="module")
def monitoring_mv(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    return mv


def test_create_low_level(cluster: Cluster, root_mv: ModelVersion, monitoring_mv: ModelVersion):
    ms_config = MetricSpecConfig(monitoring_mv.id, 10, ThresholdCmpOp.NOT_EQ)
    ms = MetricSpec.create(cluster, "test", root_mv.id, ms_config)
    ms_found = MetricSpec.find_by_id(cluster, ms.id)
    assert ms_found.name == ms.name
    assert ms_found.modelversion_id == ms.modelversion_id


def test_create_high_level(cluster: Cluster, root_mv: ModelVersion, monitoring_mv: ModelVersion):
    metric = monitoring_mv.as_metric(10, ThresholdCmpOp.NOT_EQ)
    root_mv.assign_metrics([metric])
    assert monitoring_mv.name in [ms.name for ms in MetricSpec.find_by_modelversion(cluster, root_mv.id)]


def test_list(cluster: Cluster, root_mv: ModelVersion, monitoring_mv: ModelVersion):
    ms_config = MetricSpecConfig(monitoring_mv.id, 10, ThresholdCmpOp.NOT_EQ)
    ms = MetricSpec.create(cluster, "test_list", root_mv.id, ms_config)
    assert ms.name in [item.name for item in MetricSpec.list(cluster)]


def test_list_for_modelversion(cluster: Cluster, local_model: LocalModel, monitoring_mv: ModelVersion):
    new_root_mv: ModelVersion = local_model.upload(cluster)
    ms_config = MetricSpecConfig(monitoring_mv.id, 10, ThresholdCmpOp.NOT_EQ)
    ms1 = MetricSpec.create(cluster, "test_list_for_modelversion", new_root_mv.id, ms_config)
    ms2 = MetricSpec.create(cluster, "test_list_for_modelversion", new_root_mv.id, ms_config)
    mss = MetricSpec.find_by_modelversion(cluster, new_root_mv.id)
    assert len(mss) == 2
    assert ms1.id in [ms.id for ms in mss]
    assert ms2.id in [ms.id for ms in mss]
    

def test_delete(cluster: Cluster, local_model: LocalModel, monitoring_mv: ModelVersion):
    new_root_mv: ModelVersion = local_model.upload(cluster)
    ms_config = MetricSpecConfig(monitoring_mv.id, 10, ThresholdCmpOp.NOT_EQ)
    ms1 = MetricSpec.create(cluster, "test_list_for_modelversion", new_root_mv.id, ms_config)
    ms2 = MetricSpec.create(cluster, "test_list_for_modelversion", new_root_mv.id, ms_config)
    MetricSpec.delete(cluster, ms2.id)
    mss = MetricSpec.find_by_modelversion(cluster, new_root_mv.id)
    assert len(mss) == 1
    assert ms1.id in [ms.id for ms in mss]
    assert ms2.id not in [ms.id for ms in mss]
