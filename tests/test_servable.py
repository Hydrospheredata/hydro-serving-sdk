from hydrosdk.deployment_configuration import DeploymentConfigurationBuilder, DeploymentConfiguration
from hydrosdk.exceptions import BadRequest
from hydrosdk.modelversion import ModelVersion
from tests.common_fixtures import *
from tests.utils import *


@pytest.fixture(scope="module")
def mv(cluster: Cluster, local_model: LocalModel) -> ModelVersion:
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    return mv


@pytest.fixture(scope="module")
def deployment_configuration(cluster):
    try:
        deployment_configuration = DeploymentConfiguration.find(cluster, "servable_dep_config")
    except BadRequest:
        deployment_configuration = DeploymentConfigurationBuilder("servable_dep_config", cluster).with_replicas(2).build()

    yield deployment_configuration
    DeploymentConfiguration.delete(cluster, deployment_configuration.name)


@pytest.fixture(scope="module")
def servable(cluster: Cluster, mv: ModelVersion, deployment_configuration: DeploymentConfiguration):
    sv: Servable = Servable.create(cluster, mv.name, mv.version, deployment_configuration=deployment_configuration)
    servable_lock_till_serving(cluster, sv.name)
    yield Servable.find_by_name(cluster, sv.name)
    Servable.delete(cluster, sv.name)


def test_servable_create(cluster, servable: Servable):
    assert Servable.find_by_name(cluster, servable.name)


def test_servable_list(cluster: Cluster, servable: Servable):
    assert servable.name in [s.name for s in Servable.list(cluster)]


def test_servable_find_by_name(cluster: Cluster, servable: Servable):
    servable_found: Servable = Servable.find_by_name(cluster, servable.name)
    assert servable.name == servable_found.name


def test_servable_delete(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    Servable.delete(cluster, sv.name)
    with pytest.raises(BadRequest):
        Servable.find_by_name(cluster, sv.name)


def test_servable_status(servable: Servable):
    assert servable.status == ServableStatus.SERVING


def test_servable_with_deployment_config(servable: Servable):
    assert servable.deployment_configuration is not None


def test_servable_logs_not_empty(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    i = 0
    for _ in sv.logs():
        i += 1
    assert i > 0
    Servable.delete(cluster, sv.name)


def test_servable_logs_follow_not_empty(cluster: Cluster, mv):
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
    Servable.delete(cluster, sv.name)
