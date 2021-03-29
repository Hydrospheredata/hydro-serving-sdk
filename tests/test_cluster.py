import pytest
import grpc
import requests_mock
from grpc import ssl_channel_credentials

from hydrosdk.cluster import Cluster
from tests.config import *


@pytest.fixture
def req_mock():
    def _mock():
        mock = requests_mock.Mocker()
        mock.get(HTTP_CLUSTER_ENDPOINT + "/api/buildinfo",
                 json={"name": "serving-manager", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get(HTTP_CLUSTER_ENDPOINT + "/gateway/buildinfo",
                 json={"name": "serving-gateway", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        mock.get(HTTP_CLUSTER_ENDPOINT + "/monitoring/buildinfo",
                 json={"name": "serving-sonar", "gitHeadCommit": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29",
                       "gitCurrentTags": [], "gitCurrentBranch": "master", "scalaVersion": "2.12.8",
                       "version": "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion": "1.2.8"})
        return mock
    return _mock


def test_cluster_init(req_mock):
    with req_mock():
        cluster = Cluster(HTTP_CLUSTER_ENDPOINT)
        assert cluster.http_address == HTTP_CLUSTER_ENDPOINT


@pytest.mark.skipif(GRPC_CLUSTER_ENDPOINT_SSL, reason="Cluster is running in secure mode")
def test_cluster_insecure_grpc(req_mock):
    with req_mock():
        cluster = Cluster(http_address=HTTP_CLUSTER_ENDPOINT, grpc_address=GRPC_CLUSTER_ENDPOINT)
        assert cluster.channel is not None


@pytest.mark.skipif(not GRPC_CLUSTER_ENDPOINT_SSL, reason="Cluster is running in insecure mode")
def test_cluster_secure_grpc(req_mock):
    with req_mock():
        cluster = Cluster(
            http_address=HTTP_CLUSTER_ENDPOINT, 
            grpc_address=GRPC_CLUSTER_ENDPOINT, 
            ssl=True,
            grpc_credentials=ssl_channel_credentials()
        )
        assert cluster.channel is not None
