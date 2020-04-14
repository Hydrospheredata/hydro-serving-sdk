import os
import time
import pytest
import yaml

from hydrosdk.application import Application, ApplicationStatus
from tests.test_model import get_cluster, get_local_model


DEFAULT_APP_NAME = "test-app"


def create_test_application(cluster, upload_response, model):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/application.yml') as f:
        d = yaml.safe_load(f)
        app = Application.parse_application(d)
        app_as_dict = app._asdict()
        app_as_dict["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersionId"] = upload_response[model].model_version_id

        while upload_response[model].building():
            print("building")

        time.sleep(3)

        application = Application.create(cluster, app_as_dict)
        return application


def test_list_all():
    cluster = get_cluster()
    local_model = get_local_model()
    upload_response = local_model.upload(cluster=cluster)

    created_application = create_test_application(cluster=cluster, model=local_model, upload_response=upload_response)
    all_applications = Application.list_all(cluster)

    assert all_applications is not None


def test_find_by_name():
    cluster = get_cluster()
    local_model = get_local_model()
    upload_response = local_model.upload(cluster=cluster)

    created_application = create_test_application(cluster=cluster, model=local_model, upload_response=upload_response)
    found_application = Application.find_by_name(cluster=cluster, app_name=DEFAULT_APP_NAME)
    assert found_application.name == DEFAULT_APP_NAME


def test_delete():
    cluster = get_cluster()
    local_model = get_local_model()
    upload_response = local_model.upload(cluster=cluster)

    created_application = create_test_application(cluster=cluster, model=local_model,
                                                  upload_response=upload_response)
    deleted_application = Application.delete(cluster=cluster, app_name=DEFAULT_APP_NAME)

    with pytest.raises(Exception, match=r"Failed to find by name.*"):
        found_application = Application.find_by_name(cluster=cluster, app_name=DEFAULT_APP_NAME)


def test_create():
    cluster = get_cluster()
    local_model = get_local_model()
    upload_response = local_model.upload(cluster=cluster)

    created_application = create_test_application(cluster=cluster, model=local_model, upload_response=upload_response)

    all_applications = Application.list_all(cluster=cluster)

    found_application = False
    for application in all_applications:
        if application.name == DEFAULT_APP_NAME:
            found_application = True
            break

    assert found_application


def test_application_status():
    cluster = get_cluster()
    local_model = get_local_model()
    upload_response = local_model.upload(cluster=cluster)

    Application.delete(cluster, DEFAULT_APP_NAME)
    created_application = create_test_application(cluster=cluster, model=local_model,
                                                  upload_response=upload_response)

    assert created_application.status == ApplicationStatus.ASSEMBLING

    time.sleep(10)

    found_application = Application.find_by_name(cluster=cluster, app_name=DEFAULT_APP_NAME)

    assert found_application.status == ApplicationStatus.READY
