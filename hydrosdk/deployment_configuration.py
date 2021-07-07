import re
from pydantic import BaseModel
from typing import List, Dict, Optional

from hydrosdk.cluster import Cluster
from hydrosdk.utils import handle_request_error, enable_camel_case
from hydrosdk.builder import AbstractBuilder

class K8SEntity(BaseModel):
    class Config:
        @staticmethod
        def to_camel_case(x: str):
            segments = x.split("_")
            return segments[0] + "".join([x.capitalize() for x in segments[1:]])

        alias_generator = to_camel_case
        allow_population_by_field_name = True

class HorizontalPodAutoScalerSpec(K8SEntity):
    """
    HPA Specification

    :param max_replicas: upper limit for the number of replicas to which the autoscaler can scale up. It cannot be less that minReplicas.
    :param min_replicas: lower limit for the number of replicas to which the autoscaler can scale down.
      It defaults to 1 pod. minReplicas is allowed to be 0 if the alpha feature gate HPAScaleToZero is enabled and at least
      one Object or External metric is configured. Scaling is active as long as at least one metric value is available.
    :param cpu_utilization: target average CPU utilization (represented as a percentage of requested CPU)
      over all the pods; if not specified the default autoscaling policy will be used.
    """
    min_replicas: int
    max_replicas: int
    cpu_utilization: int = None

class NodeSelectorRequirement(K8SEntity):
    """
    A node selector requirement is a selector that contains values, a key, and an operator that relates the key and values.

    :param operator: Represents a key's relationship to a set of values. Valid operators are In, NotIn, Exists, DoesNotExist. Gt, and Lt.
    :param key: The label key that the selector applies to.
    :param values: An array of string values. If the operator is In or NotIn, the values array must be non-empty.
     If the operator is Exists or DoesNotExist, the values array must be empty.
     If the operator is Gt or Lt, the values array must have a single element, which will be interpreted as an integer.
     This array is replaced during a strategic merge patch.
    """
    key: str
    operator: str
    values: List[str] = None

class NodeSelectorTerm(K8SEntity):
    """
    A null or empty node selector term matches no objects. The requirements of them are ANDed.
     The TopologySelectorTerm type implements a subset of the NodeSelectorTerm.
    """
    match_expressions: List[NodeSelectorRequirement] = None
    match_fields: List[NodeSelectorRequirement] = None

class PreferredSchedulingTerm(K8SEntity):
    """
    An empty preferred scheduling term matches all objects with implicit weight 0 (i.e. it's a no-op). A null preferred scheduling
     term matches no objects (i.e. is also a no-op).

    :param preference: A node selector term, associated with the corresponding weight.
    :param weight: Weight associated with matching the corresponding nodeSelectorTerm, in the range 1-100.
    """
    weight: int
    preference: NodeSelectorTerm

class NodeSelector(K8SEntity):
    """
    A node selector represents the union of the results of one or more label queries over a set of nodes; that is,
     it represents the OR of the selectors represented by the node selector terms.
    """
    node_selector_terms: List[NodeSelectorTerm] = []

class LabelSelectorRequirement(K8SEntity):
    """
    A label selector requirement is a selector that contains values, a key, and an operator that relates the key and values.

    :param key: key is the label key that the selector applies to.
    :param operator: operator represents a key's relationship to a set of values. Valid operators are In, NotIn, Exists and DoesNotExist.
    :param values: values is an array of string values. If the operator is In or NotIn, the values array must be non-empty.
     If the operator is Exists or DoesNotExist, the values array must be empty. This array is replaced during a strategic merge patch.
    """
    key: str
    operator: str
    values: Optional[List[str]] = None

class LabelSelector(K8SEntity):
    """
    A label selector is a label query over a set of resources. The result of matchLabels and matchExpressions are ANDed.
    An empty label selector matches all objects. A null label selector matches no objects.

    :param match_labels: map of {key,value} pairs. A single {key,value} in the matchLabels map is equivalent
      to an element of matchExpressions, whose key field is "key", the operator is "In", and the values array contains only "value".
      The requirements are ANDed.
    :param match_expressions: matchExpressions is a list of label selector requirements. The requirements are ANDed.
    """
    match_expressions: Optional[List[LabelSelectorRequirement]] = None
    match_labels: Optional[Dict[str, str]] = None

class PodAffinityTerm(K8SEntity):
    """
    Defines a set of pods (namely those matching the labelSelector relative to the given namespace(s)) that this pod should be co-located
    (affinity) or not co-located (anti-affinity) with, where co-located is defined as running on a node whose value of the label with key
    <topologyKey> matches that of any node on which a pod of the set of pods is running

    :param label_selector: A label query over a set of resources, in this case pods.
    :param namespaces: namespaces specifies which namespaces the labelSelector applies to (matches against);
     null or empty list means "this pod's namespace"
    :param topology_key: This pod should be co-located (affinity) or not co-located (anti-affinity) with the pods
     matching the labelSelector in the specified namespaces, where co-located is defined as running on a node whose
     value of the label with key topologyKey matches that of any node on which any of the selected pods is running.
     Empty topologyKey is not allowed.
    """
    label_selector: LabelSelector
    topology_key: str
    namespaces: Optional[List[str]] = None

class WeightedPodAffinityTerm(K8SEntity):
    """
    The weights of all of the matched WeightedPodAffinityTerm fields are added per-node to find the most preferred node(s)

    :param pod_affinity_term: Required. A pod affinity term, associated with the corresponding weight.
    :param weight: weight associated with matching the corresponding podAffinityTerm, in the range 1-100.
    """
    weight: int
    pod_affinity_term: PodAffinityTerm

class NodeAffinity(K8SEntity):
    """
    Group of node affinity scheduling rules.
    """
    preferred_during_scheduling_ignored_during_execution: List[PreferredSchedulingTerm] = []
    required_during_scheduling_ignored_during_execution: Optional[NodeSelector] = None

class PodAffinity(K8SEntity):
    """
    Pod affinity is a group of inter pod affinity scheduling rules.
    """
    preferred_during_scheduling_ignored_during_execution: List[WeightedPodAffinityTerm] = []
    required_during_scheduling_ignored_during_execution: List[PodAffinityTerm] = []

class PodAntiAffinity(K8SEntity):
    """
    Pod anti affinity is a group of inter pod anti affinity scheduling rules.
    """
    preferred_during_scheduling_ignored_during_execution: List[WeightedPodAffinityTerm] = []
    required_during_scheduling_ignored_during_execution: List[PodAffinityTerm] = []

class Affinity(K8SEntity):
    """
    Group of affinity scheduling rules.

    :param node_affinity: node affinity scheduling rules for the pod.
    :param pod_affinity: pod affinity scheduling rules (e.g. co-locate this pod in the same node, zone, etc. as some other pod(s)).
    :param pod_anti_affinity: pod anti-affinity scheduling rules
        (e.g. avoid putting this pod in the same node, zone, etc. as some other pod(s)).
    """
    node_affinity: Optional[NodeAffinity] = None
    pod_affinity: Optional[PodAffinity] = None
    pod_anti_affinity: Optional[PodAntiAffinity] = None

class Toleration(K8SEntity):
    """
    The pod this Toleration is attached to tolerates any taint that matches the triple <key,value,effect> using the matching operator <operator>.

    :param effect: Effect indicates the taint effect to match.
        Empty means match all taint effects. When specified, allowed values are NoSchedule, PreferNoSchedule and NoExecute.
    :param key: Key is the taint key that the toleration applies to. Empty means match all taint keys.
     If the key is empty, operator must be Exists; this combination means to match all values and all keys.
    :param operator: Operator represents a key's relationship to the value. Valid operators are Exists and Equal. Defaults to Equal.
     Exists is equivalent to wildcard for value, so that a pod can tolerate all taints of a particular category.
    :param toleration_seconds: TolerationSeconds represents the period of time the toleration (which must be of effect NoExecute,
     otherwise this field is ignored) tolerates the taint. By default, it is not set, which means tolerate the taint forever (do not evict).
     Zero and negative values will be treated as 0 (evict immediately) by the system.
    :param value: Value is the taint value the toleration matches to. If the operator is Exists, the value should be empty,
     otherwise just a regular string.
    """
    operator: str
    toleration_seconds: Optional[int] = None
    key: Optional[str] = None
    value: Optional[str] = None
    effect: Optional[str] = None

class PodSpec(K8SEntity):
    """
    PodSpec contains all configurations related to a pod.
    """
    node_selector: Dict[str, str] = {}
    tolerations: List[Toleration] = []
    affinity: Optional[Affinity] = None

class DeploymentSpec(K8SEntity):
    """
    DeploymentSpec contains all configurations related to a deployment.
    """
    replica_count: int = 1

class ResourceRequirements(K8SEntity):
    """
    Resources required by the container.

    :param limits: the maximum amount of compute resources allowed
    :param requests: Requests describes the minimum amount of compute resources required.
                     If Requests is omitted for a container, it defaults to Limits if that is explicitly specified,
                     otherwise to an implementation-defined value.
    """
    limits: Dict[str, str]
    requests: Dict[str, str] = None

    def __post_init__(self):
        quantity_validation_regex = r"^([+-]?[0-9.]+)([eEinumkKMGTP]*[-+]?[0-9]*)$"
        for resource, quantity in self.limits.items():
            if not re.match(quantity_validation_regex, quantity):
                raise ValueError(f"{resource} limit ({quantity}) is invalid. Quantity must match '{quantity_validation_regex}'")
        for resource, quantity in self.requests.items():
            if not re.match(quantity_validation_regex, quantity):
                raise ValueError(f"{resource} request ({quantity}) is invalid. Quantity must match '{quantity_validation_regex}'")

class ContainerSpec(K8SEntity):
    """
    ContainerSpec contains all configurations related to a container.
    """
    resources: Optional[ResourceRequirements] = None
    env: Optional[Dict[str, str]] = None

class DeploymentConfiguration(K8SEntity):
    """
    DeploymentConfiguration encapsulates k8s configs about how your Servables and ModelVariants should run.

    :param name: Unique Name of a Deployment Configuration
    :param hpa: HorizontalPodAutoScaler specification
    :param pod: Pod specification
    :param container: Container specification
    :param deployment: Deployment specification
    """
    name: str
    hpa: Optional[HorizontalPodAutoScalerSpec] = None
    pod: Optional[PodSpec] = None
    container: Optional[ContainerSpec] = None
    deployment: Optional[DeploymentSpec] = None

    _BASE_URL: str = "/api/v2/deployment_configuration"

    @staticmethod
    def list(cluster: Cluster) -> List['DeploymentConfiguration']:
        """
        List all deployment configurations

        :param cluster:
        :return:
        """
        resp = cluster.request("GET", DeploymentConfiguration._BASE_URL)
        handle_request_error(resp, f"Failed to get a list of Deployment Configurations - {resp.status_code} {resp.text}")
        return [DeploymentConfiguration.parse_obj(app_json) for app_json in resp.json()]

    @staticmethod
    def find(cluster: Cluster, name: str) -> 'DeploymentConfiguration':
        """
        Search a deployment configuration by name

        :param cluster:
        :param name:
        :return:
        """
        resp = cluster.request("GET", f"{DeploymentConfiguration._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to find Deployment Configuration named {name} {resp.status_code} {resp.text}")
        return DeploymentConfiguration.parse_obj(resp.json())

    @staticmethod
    def delete(cluster: Cluster, name: str) -> dict:
        """
        Delete deployment configuration from Hydrosphere cluster

        :param cluster: Active Cluster
        :param name: Deployment configuration name
        :return: json response from the server
        """
        resp = cluster.request("DELETE", f"{DeploymentConfiguration._BASE_URL}/{name}")
        handle_request_error(resp, f"Failed to delete Deployment Configuration named {name}: {resp.status_code} {resp.text}")
        return resp
    
    def to_dict(self):
        return self.dict(by_alias=True)


class DeploymentConfigurationBuilder(AbstractBuilder):
    """
    DeploymentConfigBuilder is used to create a new DeploymentConfiguration.

    :Example:

    Create a new deployment configuration.

    >>> from hydrosdk import Cluster, DeploymentConfigurationBuilder
    >>> cluster = Cluster('http-cluster-endpoint')
    >>> config_builder = DeploymentConfigurationBuilder(name="new_config")
    >>> config = config_builder.with_hpa(max_replicas=10, min_replicas=2, target_cpu_utilization_percentage=80). \
                    with_pod_node_selector({"label1": "key1", "foo": "bar"}). \
                    with_resource_requirements(limits={"cpu": "4", "memory": "4G"}, \
                                               requests={"cpu": "2", "memory": "2G"}). \
                    with_env({"ENVIRONMENT": "1"}).
                    with_replicas(replica_count=4). \
                    with_toleration(effect="PreferNoSchedule", key="equalToleration", toleration_seconds=30, operator="Exists").\
                    build(cluster)
    """

    def __init__(self, name: str) -> "DeploymentConfigurationBuilder":
        """
        Create new Deployment Config Builder

        :param name: Name of created deployment  configuration
        """
        self.name = name
        self.hpa = None
        self.pod_spec = None
        self.container_spec = None
        self.deployment_spec = None

    def with_hpa(self, max_replicas, min_replicas=1, target_cpu_utilization_percentage=None) -> 'DeploymentConfigurationBuilder':
        """
        Adds a HorizontalPodAutoScaler specs to a DeploymentConfiguration.

        :param min_replicas: min_replicas is the lower limit for the number of replicas to which the autoscaler can scale down.
                             It defaults to 1 pod. minReplicas is allowed to be 0 if the alpha feature gate HPAScaleToZero
                             is enabled and at least one Object or External metric is configured. Scaling is active as long
                             as at least one metric value is available.
        :param max_replicas: upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
        :param target_cpu_utilization_percentage: target average CPU utilization (represented as a percentage of requested CPU)
         over all the pods; if not specified the default autoscaling policy will be used.
        """
        if self.hpa is not None:
            raise ValueError("HorizontalPodAutoScalerSpec is already set")
        if max_replicas < min_replicas:
            raise ValueError(f"max_replicas ({max_replicas}) cannot be smaller than min_replicas ({min_replicas}).")
        self.hpa = HorizontalPodAutoScalerSpec(min_replicas=min_replicas, max_replicas=max_replicas, cpu_utilization=target_cpu_utilization_percentage)
        return self

    def with_pod_node_selector(self, node_selector: Dict[str, str]) -> 'DeploymentConfigurationBuilder':
        """
        Adds selector which must be true for the pod to fit on a node.

        :param node_selector: Selector which must match a node's labels for the pod to be scheduled on that node
        :return:
        """
        if self.pod_spec is None:
            self.pod_spec = PodSpec(node_selector=node_selector)
        elif self.pod_spec.node_selector is not None:
            raise ValueError(f"Cannot set Node Selector as {node_selector},"
                             f" it is already set to {self.pod_spec.node_selector}")
        else:
            self.pod_spec.node_selector = node_selector
        return self

    def with_affinity(self, affinity: Affinity) -> 'DeploymentConfigurationBuilder':
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

    def with_resource_requirements(self, limits: Dict[str, str], requests: Dict[str, str] = None) -> 'DeploymentConfigurationBuilder':
        """
        Specify resources required by this container.

        :param limits: the maximum amount of compute resources allowed
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

    def with_env(self, env: Dict[str, str]) -> 'DeploymentConfigurationBuilder':
        """
        Specify env for the container.

        :param env: dictionary with the required environment variables.
        """
        if self.container_spec is None:
            self.container_spec = ContainerSpec(env=env)
        elif self.container_spec.env is not None:
            raise ValueError(f"Cannot set env as {env},"
                             f" they are already set to {self.container_spec.env}")
        else:
            self.container_spec.env = env
        return self
        

    def with_toleration(self,
                        effect: str = None,
                        key: str = None,
                        operator: str = "Equal",
                        toleration_seconds: int = None,
                        value: str = None) -> 'DeploymentConfigurationBuilder':
        """
        Specify pod's toleration. The pod this Toleration is attached to tolerates any taint that matches the triple <key,value,effect>
         using the matching operator <operator>.

        :param effect: Effect indicates the taint effect to match.
            Empty means match all taint effects. When specified, allowed values are NoSchedule, PreferNoSchedule and NoExecute.
        :param key: Key is the taint key that the toleration applies to. Empty means match all taint keys.
         If the key is empty, operator must be Exists; this combination means to match all values and all keys.
        :param operator: Operator represents a key's relationship to the value. Valid operators are Exists and Equal. Defaults to Equal.
         Exists is equivalent to wildcard for value, so that a pod can tolerate all taints of a particular category.
        :param toleration_seconds: TolerationSeconds represents the period of time the toleration (which must be of effect NoExecute,
         otherwise this field is ignored) tolerates the taint. By default, it is not set, which means tolerate the taint forever (do not evict).
         Zero and negative values will be treated as 0 (evict immediately) by the system.
        :param value: Value is the taint value the toleration matches to. If the operator is Exists, the value should be empty,
         otherwise just a regular string.
        """
        new_toleration = Toleration(effect=effect, key=key, operator=operator, toleration_seconds=toleration_seconds, value=value)
        if self.pod_spec is None:
            self.pod_spec = PodSpec(tolerations=[new_toleration])
        else:
            self.pod_spec.tolerations.append(new_toleration)
        return self

    def with_replicas(self, replica_count=1) -> 'DeploymentConfigurationBuilder':
        """
        Specify number of desired pods for a deployment

        :param replica_count: Number of desired pods. This is a pointer to distinguish between explicit zero and not specified.
         Defaults to 1.
        :return:
        """
        if self.deployment_spec is None:
            self.deployment_spec = DeploymentSpec(replica_count=replica_count)
        elif self.deployment_spec.replica_count is not None:
            raise ValueError("Replica Count is already set")
        else:
            self.deployment_spec.replica_count = replica_count
        return self
    
    def _with_hpa_spec(self, hpa: Optional[HorizontalPodAutoScalerSpec] = None) -> 'DeploymentConfigurationBuilder':
        """
        Add a HorizontalPodAutoScalerSpec.
        """
        if hpa is None:
            return self
        if not isinstance(hpa, HorizontalPodAutoScalerSpec):
            raise ValueError("hpa should be of type HorizontalPodAutoScalerSpec")
        if self.hpa is not None:
            raise ValueError("HorizontalPodAutoScalerSpec is already set")
        self.hpa = hpa
        return self
    
    def _with_deployment_spec(self, deployment: Optional[DeploymentSpec] = None) -> 'DeploymentConfigurationBuilder':
        """
        Add a DeploymentSpec.
        """
        if deployment is None:
            return self
        if not isinstance(deployment, DeploymentSpec):
            raise ValueError("deployment should be of type HorizontalPodAutoScalerSpec")
        if self.deployment_spec is not None:
            raise ValueError("DeploymentSpec is already set")
        self.deployment_spec = deployment
        return self

    def _with_container_spec(self, container: Optional[ContainerSpec]) -> 'DeploymentConfigurationBuilder':
        """
        Add a ContainerSpec.
        """
        if container is None:
            return self
        if not isinstance(container, ContainerSpec):
            raise ValueError("container should be of type ContainerSpec")
        if self.container_spec is not None:
            raise ValueError("ContainerSpec is already set")
        self.container_spec = container
        return self
    
    def _with_pod_spec(self, pod: Optional[PodSpec]) -> 'DeploymentConfigurationBuilder':
        """
        Add a PodSpec.
        """
        if pod is None:
            return self
        if not isinstance(pod, PodSpec):
            raise ValueError("pod should be of type PodSpec")
        if self.pod_spec is not None:
            raise ValueError("PodSpec is already set")
        self.pod_spec = pod
        return self
    
    def build(self, cluster: Cluster) -> DeploymentConfiguration:
        """
        Create the Deployment Configuration in your Hydrosphere cluster.

        :return: Deployment Configuration object from the Hydrosphere cluster
        """
        config = DeploymentConfiguration(
            name=self.name,
            hpa=self.hpa,
            pod=self.pod_spec,
            container=self.container_spec,
            deployment=self.deployment_spec
        )
        resp = cluster.request("POST", DeploymentConfiguration._BASE_URL, json=config.to_dict())
        handle_request_error(resp, f"Failed to upload new Deployment Configuration. {resp.status_code} {resp.text}")
        return DeploymentConfiguration.parse_obj(resp.json())
