import random

from hydrosdk.cluster import Cluster
from hydrosdk.servable import Servable, ServableStatus
from hydrosdk.deployment_configuration import DeploymentConfigurationBuilder, DeploymentConfiguration
from hydrosdk.exceptions import BadRequestException
from hydrosdk.modelversion import ModelVersion
from tests.common_fixtures import *


@pytest.fixture(scope="module")
def mv(cluster: Cluster, model_version_builder: ModelVersionBuilder) -> ModelVersion:
    mv: ModelVersion = model_version_builder.build(cluster)
    mv.lock_till_released(timeout=config.lock_timeout)
    return mv


@pytest.fixture(scope="module")
def deployment_configuration_name(scope="module"):
    return f"deploy_config_{random.randint(0, 1e5)}"


@pytest.yield_fixture(scope="module")
def deployment_configuration(cluster: Cluster, deployment_configuration_name: str):
    deployment_configuration = DeploymentConfigurationBuilder(deployment_configuration_name) \
        .with_replicas(2) \
        .build(cluster)
    yield deployment_configuration
    DeploymentConfiguration.delete(cluster, deployment_configuration.name)


@pytest.yield_fixture(scope="module")
def servable(cluster: Cluster, mv: ModelVersion, deployment_configuration: DeploymentConfiguration):
    sv: Servable = Servable.create(cluster, mv.name, mv.version, deployment_configuration=deployment_configuration)
    sv.lock_while_starting(timeout=config.lock_timeout)
    yield sv
    Servable.delete(cluster, sv.name)


def test_servable_create(cluster: Cluster, servable: Servable):
    assert Servable.find_by_name(cluster, servable.name)


def test_servable_list(cluster: Cluster, servable: Servable):
    assert servable.name in [s.name for s in Servable.list(cluster)]


def test_servable_find_by_name(cluster: Cluster, servable: Servable):
    servable_found: Servable = Servable.find_by_name(cluster, servable.name)
    assert servable.name == servable_found.name


def test_servable_delete(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    sv.lock_while_starting(timeout=config.lock_timeout)
    Servable.delete(cluster, sv.name)
    with pytest.raises(BadRequestException):
        Servable.find_by_name(cluster, sv.name)


def test_servable_with_deployment_config(servable: Servable):
    assert servable.deployment_configuration_name is not None


def test_servable_logs_not_empty(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    sv.lock_while_starting(timeout=config.lock_timeout)
    i = 0
    for _ in sv.logs():
        i += 1
    try: 
        assert i > 0
    finally: 
        Servable.delete(cluster, sv.name)


def test_servable_logs_follow_not_empty(cluster: Cluster, mv):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    sv.lock_while_starting(timeout=config.lock_timeout)
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
    try: 
        assert i > 0
    finally: 
        Servable.delete(cluster, sv.name)
