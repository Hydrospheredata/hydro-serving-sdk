import pytest

from hydrosdk.application import LocalApplication


@pytest.mark.parametrize("yaml_file", [
    "./tests/resources/simple-app.yml",
    "./tests/resources/app-with-local-model.yml"
])
def test_read_yaml(yaml_file):
    parsed_app = LocalApplication.from_file(yaml_file)


def test_list_all():
    pass


def test_find_by_name():
    pass


def test_find_by_id():
    pass


def test_delete():
    pass


def test_create():
    pass


def test_update():
    pass
