import time

from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model


def test_servable_list_all():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    Servable.create(model_name=upload_resp[model].model.name,
                    model_version=upload_resp[model].model.version, cluster=cluster)

    assert Servable.list(cluster=cluster)


def test_servable_find_by_name():
    pass


def test_servable_list_for_modelversion():
    pass


def test_servable_delete():
    cluster = get_cluster()
    model = get_local_model()
    ur = model.upload(cluster)

    created_servable = Servable.create(model_name=ur[model].model.name,
                                       model_version=ur[model].model.version, cluster=cluster)
    time.sleep(1)

    deleted_servable = Servable.delete(cluster, created_servable.name)

    time.sleep(3)

    assert not Servable.get(cluster, created_servable.name)


def test_servable_create():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=cluster)
    found_servable = Servable.get(cluster, created_servable.name)

    assert found_servable
