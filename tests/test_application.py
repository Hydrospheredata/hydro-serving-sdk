import os
import time

import pytest
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.application import Application, ApplicationStatus, ExecutionStageBuilder, ApplicationBuilder
from hydrosdk.cluster import Cluster
from hydrosdk.contract import SignatureBuilder
from hydrosdk.image import DockerImage
from hydrosdk.modelversion import LocalModel
from hydrosdk.modelversion import ModelVersion
from tests.resources.test_config import DEFAULT_APP_NAME, HTTP_CLUSTER_ENDPOINT


@pytest.fixture(scope="module")
def cluster():
    return Cluster(HTTP_CLUSTER_ENDPOINT)


@pytest.fixture(scope="module")
def test_model(cluster):
    signature = SignatureBuilder('infer') \
        .with_input('x', 'double', "scalar") \
        .with_output('y', 'double', "scalar").build()

    contract = ModelContract(predict=signature)

    path = os.path.dirname(os.path.abspath(__file__)) + "/resources/sqrt_model/"
    payload = ['src/func_main.py']

    test_model = LocalModel(name="test_model",
                            contract=contract,
                            runtime=DockerImage("hydrosphere/serving-runtime-python-3.7", "2.3.2", None),
                            payload=payload,
                            path=path)
    test_model.upload(cluster, wait=True)
    test_model = ModelVersion.find(cluster, name="test_model", version=1)
    return test_model


def create_test_application(cluster, test_model, name=DEFAULT_APP_NAME, ):
    mv = ModelVersion.find_by_id(cluster, test_model.id)
    stage = ExecutionStageBuilder().with_model_variant(mv, 100).build()
    application = ApplicationBuilder(cluster, name).with_stage(stage).with_metadata("key", "value").build()
    return application


class TestApplicaton:

    @classmethod
    def setup_class(cls):
        cluster = Cluster(HTTP_CLUSTER_ENDPOINT)
        apps = Application.list_all(cluster)
        for app in apps:
            app.delete()

    @pytest.fixture(autouse=True)
    def create_delete_application(self, request, cluster, test_model):
        app = create_test_application(cluster, test_model)
        request.addfinalizer(app.delete)

    def test_list_all_non_empty(self, cluster):
        all_applications = Application.list_all(cluster)
        assert all_applications is not None
        assert len(all_applications) == 1

    def test_find_by_name(self, cluster):
        found_application = Application.find_by_name(cluster=cluster, name=DEFAULT_APP_NAME)
        assert found_application.name == DEFAULT_APP_NAME

    def test_application_status(self, cluster):
        app = Application.find_by_name(cluster=cluster, name=DEFAULT_APP_NAME)
        assert app.status == ApplicationStatus.ASSEMBLING
        time.sleep(10)
        app.update_status()
        assert app.status == ApplicationStatus.READY

    def test_execution_graph(self, cluster, test_model):
        app = Application.find_by_name(cluster=cluster, name=DEFAULT_APP_NAME)
        ex_graph = app.execution_graph
        assert ex_graph.stages
        assert len(ex_graph.stages) == 1
        assert len(ex_graph.stages[0].model_variants) == 1
        assert ex_graph.stages[0].model_variants[0].modelVersion.id == test_model.id
        assert ex_graph.stages[0].model_variants[0].weight == 100

    @pytest.mark.xfail(reason="(HYD-399) Bug in the hydro-serving-manager")
    def test_metadata(self, cluster):
        app = Application.find_by_name(cluster=cluster, name=DEFAULT_APP_NAME)
        assert app.metadata == {"key": "value"}
