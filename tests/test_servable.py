import time

from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model


def test_servable_list_all():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    servable = Servable(cluster=cluster, model=model, servable_name="servable_name")
    servable.create(model_name=upload_resp[model].model.name, model_version=upload_resp[model].model.version)

    assert servable.list()


def test_servable_find_by_name():
    pass


def test_servable_list_for_modelversion():
    pass


def test_servable_delete():
    cluster = get_cluster()
    model = get_local_model()
    ur = model.upload(cluster)
    servable = Servable(cluster=cluster, model=model, servable_name="servable_name")
    created_servable = servable.create(model_name=ur[model].model.name, model_version=ur[model].model.version)
    time.sleep(1)

    deleted_servable = servable.delete(created_servable.name)

    time.sleep(3)

    assert not servable.get(created_servable.name)


def test_servable_create():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    servable = Servable(cluster=cluster, model=model, servable_name="servable_name")
    created_servable = servable.create(model_name=upload_resp[model].model.name, model_version=upload_resp[model].model.version)
    found_servable = servable.get(created_servable.name)

    assert found_servable
