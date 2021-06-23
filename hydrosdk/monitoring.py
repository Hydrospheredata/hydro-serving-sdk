from typing import Union, List

from hydrosdk.cluster import Cluster
from hydrosdk.utils import handle_request_error


class ThresholdCmpOp:
    """
    Threshold comparison operator is used to check if ModelVersion is healthy
    Model is healthy if {metric_value}{TresholdCmpOp}{threshold}
    """
    EQ = "Eq"
    NOT_EQ = "NotEq"
    GREATER = "Greater"
    GREATER_EQ = "GreaterEq"
    LESS = "Less"
    LESS_EQ = "LessEq"


class MetricModel:
    """
    Model having extra metric fields
    """

    def __init__(self, modelversion: 'ModelVersion', threshold: float,
                 comparator: ThresholdCmpOp) -> 'MetricModel':
        self.modelversion = modelversion
        self.threshold = threshold
        self.comparator = comparator


class MetricSpecConfig:
    """
    Metric specification config
    """

    def __init__(self, modelversion_id: int, threshold: Union[int, float],
                 threshold_op: ThresholdCmpOp, servable=None) -> 'MetricSpecConfig':
        """
        Create a MetricSpecConfig for the specified ModelVersion.

        :param modelversion_id: an id of the ModelVersion, which will be used as a monitoring metric
        :param threshold: a threshold for the metric
        :param threshold_op: operator to be used to compare metric values against threshold
        """
        self.servable = servable
        self.threshold_op = threshold_op
        self.threshold = threshold
        self.modelversion_id = modelversion_id


class MetricSpec:
    _BASE_URL = "/api/v2/monitoring/metricspec"

    @staticmethod
    def list(cluster: Cluster) -> List['MetricSpec']:
        """
        Send request and returns list with all available metric specs.

        :param cluster: active cluster
        :return: list with all available metric specs
        """
        resp = cluster.request("GET", MetricSpec._BASE_URL)
        handle_request_error(
            resp, f"Failed to list MetricSpecs. {resp.status_code} {resp.text}")
        return [MetricSpec._from_json(cluster, x) for x in resp.json()]

    @staticmethod
    def find_by_id(cluster: Cluster, id: int) -> 'MetricSpec':
        """
        Find MetricSpec by id.

        :param cluster: active cluster
        :param id: 
        :return: MetricSpec
        """
        resp = cluster.request("GET", f"{MetricSpec._BASE_URL}/{id}")
        handle_request_error(
            resp, f"Failed to retrieve MetricSpec for id={id}. {resp.status_code} {resp.text}")
        return MetricSpec._from_json(cluster, resp.json())

    @staticmethod
    def find_by_modelversion(cluster: Cluster, modelversion_id: int) -> List['MetricSpec']:
        """
        Find MetricSpecs by model version.

        :param cluster: active cluster
        :param modelversion_id: ModelVersions for which to return metrics.
        :return: list of MetricSpec objects
        """
        resp = cluster.request("GET", f"{MetricSpec._BASE_URL}/modelversion/{modelversion_id}")
        handle_request_error(
            resp, f"Failed to list MetricSpecs for modelversion_id={modelversion_id}. {resp.status_code} {resp.text}")
        return [MetricSpec._from_json(cluster, x) for x in resp.json()]

    @staticmethod
    def delete(cluster: Cluster, id: int) -> dict:
        """
        Delete MetricSpec.

        :return: result of deletion
        """
        resp = cluster.request("DELETE", f"{MetricSpec._BASE_URL}/{id}")
        handle_request_error(
            resp, f"Failed to delete MetricSpec for id={id}. {resp.status_code} {resp.text}")
        return resp.json()

    @staticmethod
    def create(cluster: Cluster, name: str, modelversion_id: int,
               config: MetricSpecConfig) -> 'MetricSpec':
        """
        Create MetricSpec and returns corresponding instance.

        :param cluster: active cluster
        :param name: name of the metric
        :param modelversion_id: ModelVersion for which to create a MetricSpec
        :param config: MetricSpecConfig, describing MetricSpec
        :return: metricSpec
        """
        metric_spec_json = {
            'name': name,
            'modelVersionId': modelversion_id,
            'config': {
                'modelVersionId': config.modelversion_id,
                'threshold': config.threshold,
                'thresholdCmpOperator': {
                    'kind': config.threshold_op
                }
            }
        }
        resp = cluster.request("POST", MetricSpec._BASE_URL, json=metric_spec_json)
        handle_request_error(
            resp, f"Failed to create a MetricSpec for name={name}, modelversion_id={modelversion_id}. {resp.status_code} {resp.text}")
        return MetricSpec._from_json(cluster, resp.json())

    @staticmethod
    def _from_json(cluster: Cluster, json_dict: dict) -> 'MetricSpec':
        """
        Deserialize MetricSpec from json.

        :param cluster: active cluster
        :param json_dict:
        :return: MetricSpec obj
        """
        return MetricSpec(
            cluster=cluster,
            id=json_dict['id'],
            name=json_dict['name'],
            modelversion_id=json_dict['modelVersionId'],
            config=MetricSpecConfig(
                modelversion_id=json_dict['config']['modelVersionId'],
                threshold=json_dict['config']['threshold'],
                threshold_op=json_dict['config']['thresholdCmpOperator']['kind'],
                servable=json_dict['config'].get('servable')
            ),
        )

    def __init__(self, cluster: Cluster, id: int, name: str, modelversion_id: int,
                 config: MetricSpecConfig) -> 'MetricSpec':
        self.id = id
        self.cluster = cluster
        self.config = config
        self.modelversion_id = modelversion_id
        self.name = name
