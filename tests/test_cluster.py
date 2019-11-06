import unittest

from hydrosdk.cluster import Cluster

import requests_mock

class ClusterCases(unittest.TestCase):
    def test_cluster_init(self):
        with requests_mock.Mocker() as mock:
            mock.get("https://localhost:9099/api/buildinfo", json={"name" : "serving-manager", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/gateway/buildinfo", json={"name" : "serving-gateway", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/monitoring/buildinfo", json={"name" : "serving-sonar", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            cluster = Cluster.connect("https://localhost:9099")
            print(cluster.build_info())
            self.assertEqual(cluster.address, "https://localhost:9099")

    def test_cluster_requests(self):
        with requests_mock.Mocker() as mock:
            mock.get("https://localhost:9099/api/buildinfo", json={"name" : "serving-manager", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/gateway/buildinfo", json={"name" : "serving-gateway", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/monitoring/buildinfo", json={"name" : "serving-sonar", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            cluster = Cluster.connect("https://localhost:9099")

            mock.get("https://localhost:9099/api/v2/model", json=[])
            response = cluster.request("GET", "api/v2/model").json()
            self.assertListEqual(response, [])

    def test_cluster_insecure_grpc(self):
        with requests_mock.Mocker() as mock:
            mock.get("https://localhost:9099/api/buildinfo", json={"name" : "serving-manager", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/gateway/buildinfo", json={"name" : "serving-gateway", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/monitoring/buildinfo", json={"name" : "serving-sonar", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            cluster = Cluster.connect("https://localhost:9099")
            channel = cluster.grpc_insecure()
            self.assertIsNotNone(channel)

    def test_cluster_secure_grpc(self):
        with requests_mock.Mocker() as mock:
            mock.get("https://localhost:9099/api/buildinfo", json={"name" : "serving-manager", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/gateway/buildinfo", json={"name" : "serving-gateway", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            mock.get("https://localhost:9099/monitoring/buildinfo", json={"name" : "serving-sonar", "gitHeadCommit" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "gitCurrentTags" : [], "gitCurrentBranch" : "master", "scalaVersion" : "2.12.8", "version" : "6757bc137ab41e99b8d5bc13b3609bc2162c5c29", "sbtVersion" : "1.2.8"})
            cluster = Cluster.connect("https://localhost:9099")
            channel = cluster.grpc_secure()
            self.assertIsNotNone(channel)
