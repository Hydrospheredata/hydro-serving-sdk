from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model


def test_servable_list_all():
    cluster = get_cluster()
    model = get_local_model()
    ur = model.upload(cluster)

    servable = Servable(cluster=cluster, model=model, servable_name="servable_name", host=None, port=None)
    servable.create(model_name=ur[model].model.name, model_version=ur[model].model.version)

    assert servable.list()


def test_servable_find_by_name():
    pass


def test_servable_list_for_modelversion():
    pass


def test_servable_delete():
    pass


def test_servable_create():
    cluster = get_cluster()
    model = get_local_model()
    ur = model.upload(cluster)

    servable = Servable(cluster=cluster, model=model, servable_name="servable_name", host=None, port=None)
    created_servable = servable.create(model_name=ur[model].model.name, model_version=ur[model].model.version)
    found_servable = servable.get(created_servable['fullName'])

    assert found_servable

