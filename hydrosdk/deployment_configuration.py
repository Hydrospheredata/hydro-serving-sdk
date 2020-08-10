from dataclasses import dataclass, field
from typing import List, Dict

from hydrosdk.utils import handle_request_error, enable_camel_case

_BASE_URL = "kek"


@enable_camel_case
@dataclass
class HorizontalPodAutoScalerSpec:
    min_replicas: int
    max_replicas: int
    cpu_utilization: int = None


@enable_camel_case
@dataclass
class NodeAffinity:
    preferred_during_scheduling_ignored_during_execution: List = field(default_factory=list)
    required_during_scheduling_ignored_during_execution: Dict[str, str] = field(default_factory=dict)


@enable_camel_case
@dataclass
class PodAffinity:
    preferred_during_scheduling_ignored_during_execution: List = field(default_factory=list)
    required_during_scheduling_ignored_during_execution: List = field(default_factory=list)


@enable_camel_case
@dataclass
class PodAntiAffinity:
    preferred_during_scheduling_ignored_during_execution: List = field(default_factory=list)
    required_during_scheduling_ignored_during_execution: List = field(default_factory=list)


@enable_camel_case
@dataclass
class Affinity:
    node_affinity: NodeAffinity = None
    pod_affinity: PodAffinity = None
    pod_anti_affinity: PodAntiAffinity = None


@enable_camel_case
@dataclass
class Toleration:
    effect: str
    key: str
    operator: str
    toleration_seconds: int
    value: str = None


@enable_camel_case
@dataclass
class PodSpec:
    node_selector: Dict[str, str] = field(default_factory=dict)
    affinity: Affinity = None
    tolerations: List[Toleration] = field(default_factory=list)


@enable_camel_case
@dataclass
class DeploymentSpec:
    replica_count: str = None


@enable_camel_case
@dataclass
class ResourceRequirements:
    limits: Dict[str, str]
    requests: Dict[str, str]


@enable_camel_case
@dataclass
class ContainerSpec:
    resources: ResourceRequirements = None


@enable_camel_case
@dataclass
class DeploymentConfig:
    name: str
    hpa: HorizontalPodAutoScalerSpec = None
    pod: PodSpec = None
    container: ContainerSpec = None
    deployment: DeploymentSpec = None
    _BASE_URL: str = "deployment_config"

    @staticmethod
    def find(cluster, name):
        resp = cluster.request("GET", f"{DeploymentConfig._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to find Deployment Configuration {resp.status_code} {resp.text}")
        return DeploymentConfig.from_camel_case(resp.json())

    @staticmethod
    def delete(cluster, name):
        resp = cluster.request("DELETE", f"{DeploymentConfig._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to delete Deployment Configuration {resp.status_code} {resp.text}")
        return DeploymentConfig.from_camel_case(resp.json())


class DeploymentConfigBuilder:

    def __init__(self, name, cluster):
        self.name = name
        self.cluster = cluster

        self.hpa = None
        self.pod_spec = PodSpec()
        self.container_spec = ContainerSpec()
        self.deployment_spec = DeploymentSpec()

    def with_hpa(self, min_replicas, max_replicas, target_cpu_utilization_percentage=None):
        if self.hpa is not None:
            raise ValueError("HPA already set")
        self.hpa = HorizontalPodAutoScalerSpec(min_replicas, max_replicas, target_cpu_utilization_percentage)
        return self

    def with_pod_node_selector(self, node_selector: Dict[str, str]):
        if self.pod_spec is None:
            self.pod_spec = PodSpec(node_selector=node_selector)
        else:
            self.pod_spec.node_selector.update(node_selector)
        return self

    def with_affinity(self, affinity: Affinity):
        if self.pod_spec is None:
            self.pod_spec = PodSpec(affinity=affinity)
        elif self.pod_spec.affinity is not None:
            raise ValueError("Affinity is already set")
        else:
            self.pod_spec.affinity = affinity
        return self

    def with_resource_requirements(self, resource_requirements: ResourceRequirements):
        if self.container_spec is None:
            self.container_spec = ContainerSpec(resources=resource_requirements)
        elif self.container_spec.resources is not None:
            raise ValueError("Resource Requirements is already set")
        else:
            self.container_spec.resources = resource_requirements
        return self

    def with_toleration(self, toleration: Toleration):
        if self.pod_spec is None:
            self.pod_spec = PodSpec(tolerations=[toleration])
        else:
            self.pod_spec.tolerations.append(toleration)
        return self

    def with_replicas(self, replica_count):
        if self.deployment_spec is None:
            self.deployment_spec = DeploymentSpec(replica_count=replica_count)
        elif self.deployment_spec.replica_count is not None:
            raise ValueError("Replica Count is already set")
        else:
            self.deployment_spec.replica_count = replica_count
        return self

    def build(self):
        new_deployment_config = DeploymentConfig(name=self.name,
                                                 hpa=self.hpa,
                                                 pod=self.pod_spec,
                                                 container=self.container_spec,
                                                 deployment=self.deployment_spec)

        resp = self.cluster.request("POST", DeploymentConfig._BASE_URL_, json=new_deployment_config.to_camel_case())
        handle_request_error(resp, f"Failed to upload new Deployment Configuration {resp.status_code} {resp.text}")
        return DeploymentConfig.from_camel_case(resp.json())
