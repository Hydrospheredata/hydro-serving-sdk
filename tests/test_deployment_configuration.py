import json
import os
from unittest.mock import MagicMock

import pytest
import yaml

from hydrosdk.deployment_configuration import *


@pytest.fixture
def deployment_config_json() -> Dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deployment_config_json_path = os.path.join(current_dir, 'resources/deployment_configuration.json')

    with open(deployment_config_json_path, "r") as f:
        deployment_config_json = json.load(f)
    return deployment_config_json


@pytest.fixture
def deployment_config_yaml() -> Dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deployment_config_yaml_path = os.path.join(current_dir, 'resources/deployment_configuration.yaml')

    with open(deployment_config_yaml_path, "r") as f:
        deployment_config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        del deployment_config_yaml['kind']
    return deployment_config_yaml


def test_yaml_and_json_equality(deployment_config_json, deployment_config_yaml):
    assert json.dumps(deployment_config_json, sort_keys=True) == json.dumps(deployment_config_yaml, sort_keys=True)


def test_reading_from_camel_case_json(deployment_config_json):
    deployment_config = DeploymentConfiguration.from_camel_case_dict(deployment_config_json)

    assert deployment_config.name == "cool-deployment-config"

    assert deployment_config.deployment.replica_count == 4

    assert deployment_config.hpa.min_replicas == 2
    assert deployment_config.hpa.max_replicas == 10
    assert deployment_config.hpa.cpu_utilization == 80

    assert deployment_config.container.resources.requests['cpu'] == "2"
    assert deployment_config.container.resources.requests['memory'] == "2g"
    assert deployment_config.container.resources.limits['cpu'] == "4"
    assert deployment_config.container.resources.limits['memory'] == "4g"

    assert deployment_config.pod.node_selector.keys() == {"foo", "im"}
    assert deployment_config.pod.node_selector["foo"] == "bar"
    assert deployment_config.pod.node_selector["im"] == "a map"


def test_converting_back_to_camel_case(deployment_config_json):
    deployment_config = DeploymentConfiguration.from_camel_case_dict(deployment_config_json)
    camel_case_config = deployment_config.to_camel_case_dict()
    assert json.dumps(camel_case_config, sort_keys=True) == json.dumps(deployment_config_json, sort_keys=True)


def test_deployment_configuration_builder(deployment_config_json):
    wpats = [WeightedPodAffinityTerm(weight=100,
                                     pod_affinity_term=PodAffinityTerm(topology_key="toptop", namespaces=["namespace2"],
                                                                       label_selector=LabelSelector(
                                                                           match_labels={"key": "a"},
                                                                           match_expressions=[
                                                                               LabelSelectorRequirement(key="kek",
                                                                                                        operator="In",
                                                                                                        values=["a", "b"]),
                                                                               LabelSelectorRequirement(key="kek2",
                                                                                                        operator="NotIn",
                                                                                                        values=["b"])])))]
    pod_affinity_terms = [PodAffinityTerm(topology_key="top", namespaces=["namespace1"], label_selector=LabelSelector(
        match_expressions=[LabelSelectorRequirement(key='kek', operator="Exists"),
                           LabelSelectorRequirement(key='key', operator="NotIn", values=['a', 'b'])]))]
    pod_affinity = PodAffinity(preferred_during_scheduling_ignored_during_execution=wpats,
                               required_during_scheduling_ignored_during_execution=pod_affinity_terms)

    pod_anti_affinity_terms = [PodAffinityTerm(topology_key="top", namespaces=['namespace1'], label_selector=LabelSelector(
        match_expressions=[LabelSelectorRequirement(key='kek', operator="Exists"),
                           LabelSelectorRequirement(key='key2', operator="NotIn", values=['a', 'b']),
                           LabelSelectorRequirement(key='kek2', operator="DoesNotExist")]))]

    wpaats = [WeightedPodAffinityTerm(weight=100,
                                      pod_affinity_term=PodAffinityTerm(topology_key="toptop", namespaces=["namespace2"],
                                                                        label_selector=LabelSelector(
                                                                            match_labels={"key": "a"},
                                                                            match_expressions=[
                                                                                LabelSelectorRequirement(key="kek",
                                                                                                         operator="In",
                                                                                                         values=["a", "b"]),
                                                                                LabelSelectorRequirement(key="kek",
                                                                                                         operator="NotIn",
                                                                                                         values=["b"])])))]

    pod_anti_affinity = PodAntiAffinity(required_during_scheduling_ignored_during_execution=pod_anti_affinity_terms,
                                        preferred_during_scheduling_ignored_during_execution=wpaats)

    psts = [PreferredSchedulingTerm(weight=100, preference=NodeSelectorTerm(
        match_expressions=[NodeSelectorRequirement(key="exp2", operator="NotIn", values=["aaaa", "bvzv", "czxc"])],
        match_fields=[NodeSelectorRequirement(key="fields3", operator="NotIn", values=["aaa", "cccc", "zxcc"])]))]
    node_selector = NodeSelector(node_selector_terms=[
        NodeSelectorTerm(match_expressions=[NodeSelectorRequirement(key="exp1", operator="Exists")],
                         match_fields=[NodeSelectorRequirement(key="fields1", operator="Exists")])])
    node_affinity = NodeAffinity(required_during_scheduling_ignored_during_execution=node_selector,
                                 preferred_during_scheduling_ignored_during_execution=psts)

    affinity = Affinity(pod_affinity=pod_affinity, pod_anti_affinity=pod_anti_affinity, node_affinity=node_affinity)

    config_builder = DeploymentConfigBuilder(name="cool-deployment-config", cluster=None)
    config_builder.with_hpa(max_replicas=10, min_replicas=2, target_cpu_utilization_percentage=80). \
        with_pod_node_selector({"im": "a map", "foo": "bar"}). \
        with_resource_requirements(limits={"cpu": "4", "memory": "4g"}, requests={"cpu": "2", "memory": "2g"}). \
        with_replicas(replica_count=4). \
        with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Equal", value="kek"). \
        with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Exists"). \
        with_affinity(affinity)

    config_builder.build = MagicMock(return_value=DeploymentConfiguration(name=config_builder.name,
                                                                          hpa=config_builder.hpa,
                                                                          pod=config_builder.pod_spec,
                                                                          container=config_builder.container_spec,
                                                                          deployment=config_builder.deployment_spec))

    new_config = config_builder.build()
    camel_case_config = new_config.to_camel_case_dict()

    print(json.dumps(deployment_config_json, sort_keys=True))
    print(json.dumps(camel_case_config, sort_keys=True))

    assert json.dumps(camel_case_config, sort_keys=True) == json.dumps(deployment_config_json, sort_keys=True)
