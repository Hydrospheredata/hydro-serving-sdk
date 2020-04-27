import time

import pytest
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.exceptions import ServableException
from hydrosdk.servable import Servable, ServableStatus
from tests.test_model import create_test_cluster, create_test_local_model
from tests.test_model import create_test_signature


def create_test_servable():
    http_cluster = create_test_cluster()

    signature = create_test_signature()
    contract = ModelContract(predict=signature)

    model = create_test_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster)

    return created_servable


def test_servable_list_all():
    cluster = create_test_cluster()
    model = create_test_local_model()
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
    cluster = create_test_cluster()
    model = create_test_local_model()
    ur = model.upload(cluster)

    time.sleep(3)
    created_servable = Servable.create(model_name=ur[model].model.name,
                                       model_version=ur[model].model.version, cluster=cluster,
                                       metadata={"additionalProp1": "prop"})
    time.sleep(1)

    deleted_servable = Servable.delete(cluster, created_servable.name)

    time.sleep(3)

    with pytest.raises(ServableException):
        found_servable = Servable.find(cluster, created_servable.name)


def test_servable_create():
    cluster = create_test_cluster()
    model = create_test_local_model()
    upload_resp = model.upload(cluster)

    time.sleep(3)
    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=cluster)
    found_servable = Servable.find(cluster, created_servable.name)

    assert found_servable


def test_servable_status():
    cluster = create_test_cluster()
    model = create_test_local_model()
    upload_resp = model.upload(cluster)

    time.sleep(3)
    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=cluster)

    assert created_servable.status == ServableStatus.STARTING

    time.sleep(10)
    found_servable = Servable.find(cluster, created_servable.name)

    assert found_servable.status == ServableStatus.SERVING
