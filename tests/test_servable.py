import time

import pytest
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.exceptions import ServableException
from hydrosdk.servable import Servable, ServableStatus
from tests.test_model import get_cluster, get_local_model
from tests.test_model import get_signature


def create_test_servable():
    http_cluster = get_cluster()

    signature = get_signature()
    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster)

    return created_servable


def test_servable_list_all():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    time.sleep(3)
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

    time.sleep(3)
    created_servable = Servable.create(model_name=ur[model].model.name,
                                       model_version=ur[model].model.version, cluster=cluster,
                                       metadata={"additionalProp1": "prop"})
    time.sleep(1)

    deleted_servable = Servable.delete(cluster, created_servable.name)

    time.sleep(3)

    with pytest.raises(ServableException):
        found_servable = Servable.get(cluster, created_servable.name)


def test_servable_create():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    time.sleep(3)
    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=cluster)
    found_servable = Servable.get(cluster, created_servable.name)

    assert found_servable


def test_servable_status():
    cluster = get_cluster()
    model = get_local_model()
    upload_resp = model.upload(cluster)

    time.sleep(3)
    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=cluster)

    assert created_servable.status == ServableStatus.STARTING

    time.sleep(10)
    found_servable = Servable.get(cluster, created_servable.name)

    assert found_servable.status == ServableStatus.SERVING
