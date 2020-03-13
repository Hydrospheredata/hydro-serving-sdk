import pytest
import requests_mock

from hydrosdk.cluster import Cluster
from tests.resources.test_config import config, CLUSTER_ENDPOINT


@pytest.fixture
def req_mock():
    def _mock():
        mock = requests_mock.Mocker()
        mock.get(CLUSTER_ENDPOINT + "/api/buildinfo",
                 json={"name": "serving-manager", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get(CLUSTER_ENDPOINT + "/gateway/buildinfo",
                 json={"name": "serving-gateway", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get(CLUSTER_ENDPOINT + "/monitoring/buildinfo",
                 json={"name": "serving-sonar", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        return mock
    return _mock


def test_cluster_init(req_mock):
    with req_mock():
        cluster = Cluster.connect(CLUSTER_ENDPOINT)
        print(cluster.build_info())
        assert cluster.address == CLUSTER_ENDPOINT


def test_cluster_requests(req_mock):
    with req_mock() as mock:
        mock.get(CLUSTER_ENDPOINT + "/" + config["api_endpoint"] + config["endpoint"]["model"]["list"], json=[])
        cluster = Cluster.connect(CLUSTER_ENDPOINT)
        response = cluster.request("GET", config["api_endpoint"] + config["endpoint"]["model"]["list"]).json()
        assert response == []


def test_cluster_insecure_grpc(req_mock):
    with req_mock():
        cluster = Cluster.connect(CLUSTER_ENDPOINT)
        channel = cluster.grpc_insecure()
        assert channel is not None


def test_cluster_secure_grpc(req_mock):
    with req_mock():
        cluster = Cluster.connect(CLUSTER_ENDPOINT)
        channel = cluster.grpc_secure()
        assert channel is not None
