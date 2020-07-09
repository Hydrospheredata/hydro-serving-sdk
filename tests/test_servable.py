import json
import time

import pytest
import sseclient

from hydrosdk.modelversion import LocalModel, ModelVersion
from hydrosdk.exceptions import BadRequest
from hydrosdk.servable import Servable, ServableStatus
from tests.common_fixtures import *
from tests.utils import *


def test_servable_create(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    assert Servable.find_by_name(cluster, sv.name)


def test_servable_list_all(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    assert sv.name in [servable.name for servable in Servable.list_all(cluster)]


def test_servable_find_by_name(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    sv_found: Servable = Servable.find_by_name(cluster, sv.name) 
    assert sv.name == sv_found.name


def test_servable_delete(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    Servable.delete(cluster, sv.name)
    with pytest.raises(BadRequest):
        Servable.find_by_name(cluster, sv.name)


def test_servable_status(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    sv: Servable = Servable.find_by_name(cluster, sv.name)
    assert sv.status == ServableStatus.SERVING


def test_servable_logs_not_empty(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    i = 0
    for _ in sv.logs():
        i += 1
    assert i > 0


def test_servable_logs_follow_not_empty(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    i = 0
    timeout_messages = 3
    for event in sv.logs(follow=True):
        if not event.data:
            if timeout_messages < 0:
                break
            timeout_messages -= 1
        else:
            i += 1
            break
    assert i > 0
