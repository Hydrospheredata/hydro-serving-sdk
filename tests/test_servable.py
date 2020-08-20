from hydrosdk.deployment_configuration import DeploymentConfigurationBuilder
from hydrosdk.exceptions import BadRequest
from hydrosdk.modelversion import ModelVersion
from tests.common_fixtures import *
from tests.utils import *


@pytest.fixture(scope="module")
def mv(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    return mv


@pytest.fixture(scope="module")
def deployment_configuration(cluster):
    deployment_configuration = DeploymentConfigurationBuilder("servable_dep_config", cluster).with_replicas(4).build()
    yield deployment_configuration
    # DeploymentConfiguration.delete(cluster, deployment_configuration.name)


def test_servable_create(cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    assert Servable.find_by_name(cluster, sv.name)


def test_servable_list_all(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    assert sv.name in [servable.name for servable in Servable.list_all(cluster)]


def test_servable_find_by_name(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    sv_found: Servable = Servable.find_by_name(cluster, sv.name)
    assert sv.name == sv_found.name


def test_servable_delete(cluster: Cluster, mv: ModelVersion):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    Servable.delete(cluster, sv.name)
    with pytest.raises(BadRequest):
        Servable.find_by_name(cluster, sv.name)


def test_servable_status(cluster: Cluster, mv):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    sv: Servable = Servable.find_by_name(cluster, sv.name)
    assert sv.status == ServableStatus.SERVING


def test_servable_with_deployment_config(cluster: Cluster, mv, deployment_configuration):
    sv: Servable = Servable.create(cluster, mv.name, mv.version, deployment_configuration=deployment_configuration)
    sv: Servable = Servable.find_by_name(cluster, sv.name)
    assert sv.deployment_configuration is not None


def test_servable_logs_not_empty(cluster: Cluster, mv):
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    i = 0
    for _ in sv.logs():
        i += 1
    assert i > 0


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
