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
    """
    PodSpec contains all configurations related to a pod.
    """
    node_selector: Dict[str, str] = field(default_factory=dict)
    affinity: Affinity = None
    tolerations: List[Toleration] = field(default_factory=list)


@enable_camel_case
@dataclass
class DeploymentSpec:
    """
    DeploymentSpec contains all configurations related to a deployment.
    """
    replica_count: int = 1


@enable_camel_case
@dataclass
class ResourceRequirements:
    """
    :param limits: the maximum amount of compute resources allowedv
    :param requests: Requests describes the minimum amount of compute resources required.
                     If Requests is omitted for a container, it defaults to Limits if that is explicitly specified,
                     otherwise to an implementation-defined value.
    """
    limits: Dict[str, str]
    requests: Dict[str, str] = None


@enable_camel_case
@dataclass
class ContainerSpec:
    """
    ContainerSpec contains all configurations related to a container.
    """
    resources: ResourceRequirements = None


@enable_camel_case
@dataclass
class DeploymentConfig:
    """
    DeploymentConfig encapsulates information about how your Servables should run TODO

    :param name: Unique Name of a Deployment Configuration
    :param hpa: HorizontalPodAutoScaler specification
    :param pod: Pod specification
    :param container: Container specification
    :param deployment: Deployment specification
    """
    name: str
    hpa: HorizontalPodAutoScalerSpec = None
    pod: PodSpec = None
    container: ContainerSpec = None
    deployment: DeploymentSpec = None
    _BASE_URL: str = "deployment_config"

    @staticmethod
    def find(cluster, name):
        """
        Search a deployment configuration by name
        :param cluster:
        :param name:
        :return:
        """
        resp = cluster.request("GET", f"{DeploymentConfig._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to find Deployment Configuration {resp.status_code} {resp.text}")
        return DeploymentConfig.from_camel_case(resp.json())

    @staticmethod
    def delete(cluster, name):
        """
        Delete deployment configuration from Hydrosphere cluster
        :param cluster: Active Cluster
        :param name: Deployment configuration name
        """
        resp = cluster.request("DELETE", f"{DeploymentConfig._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to delete Deployment Configuration {resp.status_code} {resp.text}")


class DeploymentConfigBuilder:
    """
    Deployment Config Builder is used to create a new DeploymentConfig

    :Example:

    Create a new deployment configuration.
    >>> from hydrosdk import Cluster, DeploymentConfigBuilder
    >>> cluster = Cluster('http-cluster-endpoint')
    >>> config_builder = DeploymentConfigBuilder(name="new_config", cluster=cluster)
    >>> config = config_builder.with_hpa(max_replicas=10, min_replicas=2, target_cpu_utilization_percentage=80). \
                    with_pod_node_selector({"label1": "key1", "foo": "bar"}). \
                    with_resource_requirements(limits={"cpu": "4", "memory": "4g"}, \
                                               requests={"cpu": "2", "memory": "2g"}). \
                    with_replicas(replica_count=4). \
                    with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Exists").\
                    build()
    """

    def __init__(self, name, cluster):
        """
        Create new Deployment Config Builder
        :param name: Name of created deployment  configuration
        :param cluster: Cluster where new deployment configuration should be created
        """
        self.name = name
        self.cluster = cluster

        self.hpa = None
        self.pod_spec = None
        self.container_spec = None
        self.deployment_spec = None

    def with_hpa(self, max_replicas, min_replicas=1, target_cpu_utilization_percentage=None) -> 'DeploymentConfigBuilder':
        """
        Adds a HorizontalPodAutoScaler specs to a DeploymentConfig

        :param min_replicas: minReplicas is the lower limit for the number of replicas to which the autoscaler can scale down.
                             It defaults to 1 pod. minReplicas is allowed to be 0 if the alpha feature gate HPAScaleToZero
                             is enabled and at least one Object or External metric is configured. Scaling is active as long
                             as at least one metric value is available.
        :param max_replicas: upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
        :param target_cpu_utilization_percentage: target average CPU utilization (represented as a percentage of requested CPU)
         over all the pods; if not specified the default autoscaling policy will be used.
        """
        if self.hpa is not None:
            raise ValueError("HPA already set")
        self.hpa = HorizontalPodAutoScalerSpec(min_replicas, max_replicas, target_cpu_utilization_percentage)
        return self

    def with_pod_node_selector(self, node_selector: Dict[str, str]) -> 'DeploymentConfigBuilder':
        """
        Adds selector which must be true for the pod to fit on a node.
        :param node_selector: Selector which must match a node's labels for the pod to be scheduled on that node
        :return:
        """
        if self.pod_spec is None:
            self.pod_spec = PodSpec(node_selector=node_selector)
        else:
            self.pod_spec.node_selector.update(node_selector)
        return self

    def with_affinity(self, affinity: Affinity) -> 'DeploymentConfigBuilder':
        """
        Adds pod scheduling constraints
        :param affinity: Group of affinity scheduling rules
        :return:
        """
        if self.pod_spec is None:
            self.pod_spec = PodSpec(affinity=affinity)
        elif self.pod_spec.affinity is not None:
            raise ValueError("Affinity is already set")
        else:
            self.pod_spec.affinity = affinity
        return self

    def with_resource_requirements(self, limits: Dict[str, str], requests: Dict[str, str] = None) -> 'DeploymentConfigBuilder':
        """
        Specify resources required by this container.
        :param limits: the maximum amount of compute resources allowedv
        :param requests: Requests describes the minimum amount of compute resources required.
                         If Requests is omitted for a container, it defaults to Limits if that is explicitly specified,
                         otherwise to an implementation-defined value.
        :return:
        """
        new_resource_requirements = ResourceRequirements(limits=limits, requests=requests)
        if self.container_spec is None:
            self.container_spec = ContainerSpec(resources=new_resource_requirements)
        elif self.container_spec.resources is not None:
            raise ValueError(f"Cannot set Resource Requirements as {new_resource_requirements},"
                             f" it is already set to {self.container_spec.resources}")
        else:
            self.container_spec.resources = new_resource_requirements
        return self

    def with_toleration(self,
                        effect: str,
                        key: str,
                        operator: str,
                        toleration_seconds: int,
                        value: str = None) -> 'DeploymentConfigBuilder':
        """
        Specify pod's toleration
        :param effect: TODO
        :param key: TODO
        :param operator: TODO
        :param toleration_seconds: TODO
        :param value: TODO
        :return:
        """
        new_toleration = Toleration(effect=effect, key=key, operator=operator, toleration_seconds=toleration_seconds, value=value)
        if self.pod_spec is None:
            self.pod_spec = PodSpec(tolerations=[new_toleration])
        else:
            self.pod_spec.tolerations.append(new_toleration)
        return self

    def with_replicas(self, replica_count) -> 'DeploymentConfigBuilder':
        """
        Specify number of desired pods for a deployment
        :param replica_count: Number of desired pods. This is a pointer to distinguish between explicit zero and not specified. Defaults to 1.
        :return:
        """
        if self.deployment_spec is None:
            self.deployment_spec = DeploymentSpec(replica_count=replica_count)
        elif self.deployment_spec.replica_count is not None:
            raise ValueError("Replica Count is already set")
        else:
            self.deployment_spec.replica_count = replica_count
        return self

    def build(self) -> DeploymentConfig:
        """
        Create the Deployment Configuration in your Hydrosphere cluster.
        :return: Deployment Configuration object from the Hydrosphere cluster
        """
        new_deployment_config = DeploymentConfig(name=self.name,
                                                 hpa=self.hpa,
                                                 pod=self.pod_spec,
                                                 container=self.container_spec,
                                                 deployment=self.deployment_spec)

        resp = self.cluster.request("POST", DeploymentConfig._BASE_URL_, json=new_deployment_config.to_camel_case())
        handle_request_error(resp, f"Failed to upload new Deployment Configuration {resp.status_code} {resp.text}")
        return DeploymentConfig.from_camel_case(resp.json())
