import json
import os
import random
from unittest.mock import MagicMock

import pytest
import yaml

from tests.common_fixtures import cluster
from hydrosdk.deployment_configuration import *
from hydrosdk.utils import BadRequestException
import uuid

@pytest.fixture(scope="module")
def deployment_configuration_name():
    return f"deploy_{uuid.uuid4()}"


@pytest.fixture()
def deployment_config_json():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deployment_config_json_path = os.path.join(current_dir, 'resources/deployment_configuration.json')

    with open(deployment_config_json_path, "r") as f:
        deployment_config_json = json.load(f)
    return deployment_config_json


@pytest.fixture()
def deployment_config_yaml():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deployment_config_yaml_path = os.path.join(current_dir, 'resources/deployment_configuration.yaml')

    with open(deployment_config_yaml_path, "r") as f:
        deployment_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        del deployment_config_yaml['kind']
    return deployment_config_yaml


def test_yaml_and_json_equality(deployment_config_json: Dict, deployment_config_yaml: Dict):
    assert json.dumps(deployment_config_json, sort_keys=True) == json.dumps(deployment_config_yaml, sort_keys=True)


def test_reading_from_camel_case_json(deployment_config_json: Dict):
    deployment_config = DeploymentConfiguration.parse_obj(deployment_config_json)

    assert deployment_config.name == "deploy_23432"

    assert deployment_config.deployment.replica_count == 4

    assert deployment_config.hpa.min_replicas == 2
    assert deployment_config.hpa.max_replicas == 10
    assert deployment_config.hpa.cpu_utilization == 80

    assert deployment_config.container.resources.requests['cpu'] == "250m"
    assert deployment_config.container.resources.requests['memory'] == "2G"
    assert deployment_config.container.resources.limits['cpu'] == "500m"
    assert deployment_config.container.resources.limits['memory'] == "4G"
    assert deployment_config.container.env["ENVIRONMENT"] == "1"

    assert deployment_config.pod.node_selector.keys() == {"foo", "key"}
    assert deployment_config.pod.node_selector["foo"] == "bar"
    assert deployment_config.pod.node_selector["key"] == "value"

def test_deployment_configuration_builder(cluster: Cluster, deployment_config_json: Dict, deployment_configuration_name: str):
    wpats = [
        WeightedPodAffinityTerm(
            weight=100,
            pod_affinity_term=PodAffinityTerm(
                topology_key="top", 
                namespaces=["namespace2"],
                label_selector=LabelSelector(
                    match_labels={"key": "value"},
                    match_expressions=[
                        LabelSelectorRequirement(key="one", operator="In", values=["a", "b"]),
                        LabelSelectorRequirement(key="two", operator="NotIn", values=["b"])
                    ]
                )
            )
        )
    ]
    pod_affinity_terms = [
        PodAffinityTerm(
            topology_key="top", 
            namespaces=["namespace1"], 
            label_selector=LabelSelector(
                match_expressions=[
                    LabelSelectorRequirement(key='exp3', operator="Exists"),
                    LabelSelectorRequirement(key='exp4', operator="NotIn", values=['a', 'b'])
                ]
            )
        )
    ]
    pod_affinity = PodAffinity(
        preferred_during_scheduling_ignored_during_execution=wpats,
        required_during_scheduling_ignored_during_execution=pod_affinity_terms
    )
    pod_anti_affinity_terms = [
        PodAffinityTerm(
            topology_key="top", 
            namespaces=['namespace1'], 
            label_selector=LabelSelector(
                match_expressions=[
                    LabelSelectorRequirement(key='one', operator="Exists"),
                    LabelSelectorRequirement(key='two', operator="NotIn", values=['a', 'b']),
                    LabelSelectorRequirement(key='three', operator="DoesNotExist")
                ]
            )
        )
    ]
    wpaats = [
        WeightedPodAffinityTerm(
            weight=100,
            pod_affinity_term=PodAffinityTerm(
                topology_key="top", 
                namespaces=["namespace2"],
                label_selector=LabelSelector(
                    match_labels={"key": "value"},
                    match_expressions=[
                        LabelSelectorRequirement(key="one", operator="In", values=["a", "b"]),
                        LabelSelectorRequirement(key="two", operator="NotIn", values=["b"])
                    ]
                )
            )
        )
    ]
    pod_anti_affinity = PodAntiAffinity(
        required_during_scheduling_ignored_during_execution=pod_anti_affinity_terms,
        preferred_during_scheduling_ignored_during_execution=wpaats
    )
    psts = [
        PreferredSchedulingTerm(
            weight=100, 
            preference=NodeSelectorTerm(
                match_expressions=[
                    NodeSelectorRequirement(key="exp2", operator="NotIn", values=["aaaa", "bbbb", "cccc"])
                ],
                match_fields=[
                    NodeSelectorRequirement(key="fields3", operator="NotIn", values=["aaaa", "bbbb", "cccc"])
                ]
            )
        )
    ]
    node_selector = NodeSelector(
        node_selector_terms=[
            NodeSelectorTerm(
                match_expressions=[
                    NodeSelectorRequirement(key="exp1", operator="Exists")
                ],
                match_fields=[
                    NodeSelectorRequirement(key="fields1", operator="Exists")
                ]
            )
        ]
    )
    node_affinity = NodeAffinity(
        required_during_scheduling_ignored_during_execution=node_selector,
        preferred_during_scheduling_ignored_during_execution=psts
    )
    affinity = Affinity(
        pod_affinity=pod_affinity, 
        pod_anti_affinity=pod_anti_affinity, 
        node_affinity=node_affinity
    )

    builder = DeploymentConfigurationBuilder(deployment_configuration_name)
    builder \
        .with_hpa(max_replicas=10, min_replicas=2, target_cpu_utilization_percentage=80) \
        .with_pod_node_selector({"key": "value", "foo": "bar"}) \
        .with_resource_requirements(limits={"cpu": "500m", "memory": "4G"}, requests={"cpu": "250m", "memory": "2G"}) \
        .with_env({"ENVIRONMENT": "1"}) \
        .with_replicas(replica_count=4) \
        .with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Equal", value="one") \
        .with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Exists") \
        .with_affinity(affinity)

    new_config = builder.build(cluster)
    camel_case_config = new_config.dict(by_alias=True, exclude_unset=True)
    deployment_config_json["name"] = deployment_configuration_name

    assert camel_case_config == deployment_config_json

    DeploymentConfiguration.delete(cluster, deployment_configuration_name)


def test_with_cluster(cluster: Cluster, deployment_configuration_name: str):

    builder = DeploymentConfigurationBuilder(deployment_configuration_name)
    deployment_config = builder.with_hpa(max_replicas=4).with_replicas(replica_count=2).build(cluster)

    assert deployment_config.name == deployment_configuration_name
    assert deployment_config.hpa.max_replicas == 4
    assert deployment_config.hpa.min_replicas == 1
    assert deployment_config.deployment.replica_count == 2

    DeploymentConfiguration.delete(cluster, deployment_configuration_name)
    with pytest.raises(BadRequestException):
        DeploymentConfiguration.find(cluster, deployment_configuration_name)


def test_list_deployment_configs(cluster):
    deployment_configs = DeploymentConfiguration.list(cluster)
    assert isinstance(deployment_configs, list)
