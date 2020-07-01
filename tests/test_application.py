import os

import pytest
import time
import yaml

from hydrosdk.application import Application, ApplicationStatus
from tests.resources.test_config import DEFAULT_APP_NAME
from tests.test_modelversion import create_test_cluster, create_test_local_model


def create_test_application(cluster, upload_response: dict = None, local_model=None):
    with open(os.path.dirname(os.path.abspath(__file__)) + '/resources/application.yml') as f:
        d = yaml.safe_load(f)
        app = Application.parse_application(d)
        app_as_dict = app._asdict()

        if not local_model and not upload_response:
            # name of model in applcation.yml should match model name in app_as_dict
            local_model = create_test_local_model(name=app_as_dict['name'])
            upload_response = local_model.upload(cluster=cluster)

        app_as_dict["executionGraph"]["stages"][0]["modelVariants"][0]["modelVersionId"] = upload_response[
            local_model].modelversion.version

        try:
            application = Application.create(cluster, app_as_dict)
        except Exception as e:
            print("While creating App, there was an exception: " + str(e))

            # if app already existed, delete first
            Application.delete(cluster=cluster, application_name=DEFAULT_APP_NAME)
            time.sleep(3)
            application = Application.create(cluster, app_as_dict)

        return application


def test_list_all():
    cluster = create_test_cluster()

    created_application = create_test_application(cluster=cluster)
    all_applications = Application.list_all(cluster)

    assert all_applications is not None


def test_find_by_name():
    cluster = create_test_cluster()

    created_application = create_test_application(cluster)
    found_application = Application.find_by_name(cluster=cluster, application_name=DEFAULT_APP_NAME)
    assert found_application.name == DEFAULT_APP_NAME


def test_delete():
    cluster = create_test_cluster()

    created_application = create_test_application(cluster=cluster)
    deleted_application = Application.delete(cluster=cluster, application_name=DEFAULT_APP_NAME)

    with pytest.raises(Exception, match=r"Failed to find by name.*"):
        found_application = Application.find_by_name(cluster=cluster, application_name=DEFAULT_APP_NAME)


def test_create():
    cluster = create_test_cluster()

    created_application = create_test_application(cluster=cluster)

    all_applications = Application.list_all(cluster=cluster)

    found_application = False
    for application in all_applications:
        if application.name == DEFAULT_APP_NAME:
            found_application = True
            break

    assert found_application


def test_application_status():
    cluster = create_test_cluster()

    Application.delete(cluster, DEFAULT_APP_NAME)
    created_application = create_test_application(cluster=cluster)

    assert created_application.status == ApplicationStatus.ASSEMBLING

    time.sleep(10)

    found_application = Application.find_by_name(cluster=cluster, application_name=DEFAULT_APP_NAME)

    assert found_application.status == ApplicationStatus.READY
