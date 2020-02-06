import pytest

from hydrosdk.cluster import Cluster

import requests_mock


@pytest.fixture
def req_mock():
    def _mock():
        mock = requests_mock.Mocker()
        mock.get("https://localhost:9099/api/buildinfo",
                 json={"name": "serving-manager", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get("https://localhost:9099/gateway/buildinfo",
                 json={"name": "serving-gateway", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get("https://localhost:9099/monitoring/buildinfo",
                 json={"name": "serving-sonar", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        return mock
    return _mock


def test_cluster_init(req_mock):
    with req_mock():
        cluster = Cluster.connect("https://localhost:9099")
        print(cluster.build_info())
        assert cluster.address == "https://localhost:9099"


def test_cluster_requests(req_mock):
    with req_mock() as mock:
        mock.get("https://localhost:9099/api/v2/model", json=[])
        cluster = Cluster.connect("https://localhost:9099")
        response = cluster.request("GET", "api/v2/model").json()
        assert response == []


def test_cluster_insecure_grpc(req_mock):
    with req_mock():
        cluster = Cluster.connect("https://localhost:9099")
        channel = cluster.grpc_insecure()
        assert channel is not None


def test_cluster_secure_grpc(req_mock):
    with req_mock():
        cluster = Cluster.connect("https://localhost:9099")
        channel = cluster.grpc_secure()
        assert channel is not None
