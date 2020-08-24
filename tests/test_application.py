import random

from hydrosdk.application import ExecutionStageBuilder, ApplicationBuilder
from hydrosdk.deployment_configuration import DeploymentConfiguration, DeploymentConfigurationBuilder
from hydrosdk.modelversion import ModelVersion
from tests.common_fixtures import *
from tests.config import *
from tests.utils import *


@pytest.fixture(scope="module")
def modelversion(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    return mv


@pytest.fixture(scope="module")
def deployment_configuration(cluster):
    deployment_configuration = DeploymentConfigurationBuilder("app_dep_config", cluster).with_replicas(4).build()
    yield deployment_configuration
    DeploymentConfiguration.delete(cluster, "app_dep_config")


@pytest.fixture(scope="module")
def app(cluster: Cluster, modelversion: ModelVersion, deployment_configuration: DeploymentConfiguration):
    stage = ExecutionStageBuilder().with_model_variant(modelversion, 100, deployment_configuration).build()
    app = ApplicationBuilder(cluster, f"{DEFAULT_APP_NAME}-{random.randint(0, 1e5)}") \
        .with_stage(stage).with_metadata("key", "value").build()
    application_lock_till_ready(cluster, app.name)
    yield app
    Application.delete(cluster, app.name)


def test_model_variants_weights_sum_up_to_100(modelversion: ModelVersion):
    stage = ExecutionStageBuilder().with_model_variant(modelversion, 100).build()
    assert stage is not None


def test_model_variants_weights_sum_up_to_100_fail(modelversion: ModelVersion):
    with pytest.raises(ValueError):
        stage = ExecutionStageBuilder().with_model_variant(modelversion, 50).build()


def test_list_all_non_empty(cluster: Cluster, app: Application):
    apps = Application.list_all(cluster)
    assert app.name in [item.name for item in apps]
    assert app.id in [item.id for item in apps]


def test_find(cluster: Cluster, app: Application):
    app_found = Application.find(cluster, app.name)
    assert app_found.id == app.id


def test_execution_graph(app: Application, modelversion: ModelVersion):
    ex_graph = app.execution_graph
    assert len(ex_graph.stages) == 1
    assert len(ex_graph.stages[0].model_variants) == 1
    assert ex_graph.stages[0].model_variants[0].modelVersion.id == modelversion.id
    assert ex_graph.stages[0].model_variants[0].weight == 100


def test_deployment_config(app: Application):
    app_dep_config = app.execution_graph.stages[0].model_variants[0].deploymentConfig
    assert app_dep_config is not None
    assert app_dep_config.name == "app_dep_config"


@pytest.mark.xfail(reason="(HYD-399) Bug in the hydro-serving-manager")
def test_metadata(cluster: Cluster, app: Application):
    app_found = Application.find(cluster, app.name)
    assert app_found.metadata == {"key": "value"}
