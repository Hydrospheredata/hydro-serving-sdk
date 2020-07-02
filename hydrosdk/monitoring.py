from typing import Union, List
from urllib.parse import urljoin

from hydrosdk.cluster import Cluster
from hydrosdk.exceptions import MetricSpecException, handle_request_error


class TresholdCmpOp:
    """
    Threshold comparison operator
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
    def __init__(self, model, threshold, comparator):
        self.model = model
        self.threshold = threshold
        self.comparator = comparator


class MetricSpecConfig:
    """
    Metric specification config
    """
    def __init__(self, modelversion_id: int, threshold: Union[int, float], threshold_op: TresholdCmpOp, servable=None):
        self.servable = servable
        self.threshold_op = threshold_op
        self.threshold = threshold
        self.modelversion_id = modelversion_id


class MetricSpec:
    BASE_URL = "/api/v2/monitoring/metricspec"

    @staticmethod
    def list_all(cluster: Cluster) -> List['MetricSpec']:
        """
        Sends request and returns list with all available metric specs.

        :param cluster: active cluster
        :raises MetricSpecException: If server returned not 200
        :return: list with all available metric specs
        """
        resp = cluster.request("GET", MetricSpec.BASE_URL)
        handle_request_error(
            resp, f"Failed to list MetricSpecs. {resp.status_code} {resp.text}")
        return [MetricSpec.__parse_json(cluster, x) for x in resp.json()]

    @staticmethod
    def list_for_modelversion(cluster: Cluster, modelversion_id: int) -> List['MetricSpec']:
        """
        Sends request and returns list with specs by model version.

        :param cluster: active cluster
        :param modelversion_id:
        :raises MetricSpecException: If server returned not 200
        :return: list of metric spec objs
        """
        url = urljoin(MetricSpec.BASE_URL, f"modelversion/{modelversion_id}")
        resp = cluster.request("get", url)
        handle_request_error(
            resp, f"Failed to list MetricSpecs for modelversion_id={modelversion_id}. {resp.status_code} {resp.text}")
        return [MetricSpec.__parse_json(cluster, x) for x in resp.json()]

    @staticmethod
    def find_by_id(cluster: Cluster, metric_spec_id: int) -> 'MetricSpec':
        """
        Returns MetricSpec by id.

        :param cluster: active cluster
        :param metric_spec_id: 
        :return: MetricSpec
        """
        url = urljoin(MetricSpec.BASE_URL, str(metric_spec_id))
        resp = cluster.request("get", url)
        handle_request_error(
            resp, f"Failed to retrieve MetricSpec for metric_spec_id={metric_spec_id}. {resp.status_code} {resp.text}")
        return MetricSpec.__parse_json(cluster, resp.json())

    @staticmethod
    def delete(cluster: Cluster, metric_spec_id: int) -> dict:
        """
        Deletes MetricSpec.

        :return: result of deletion
        """
        url = urljoin(MetricSpec.BASE_URL, str(self.metric_spec_id))
        resp = cluster.request("delete", url)
        handle_request_error(
            resp, f"Failed to delete MetricSpec for metric_spec_id={metric_spec_id}. {resp.status_code} {resp.text}")
        return resp.json()
    
    @staticmethod
    def create(cluster: Cluster, name: str, modelversion_id: int, config: MetricSpecConfig) -> 'MetricSpec':
        """
        Create MetricSpec and returns corresponding instance.

        :param cluster: active cluster
        :param name: name of the metric
        :param modelversion_id: ModelVersion for which to create a MetricSpec
        :param config: config, describing MetricSpec
        :return: metricSpec
        """
        d = {
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
        resp = cluster.request("POST", MetricSpec.BASE_URL, json=d)
        handle_request_error(
            resp, f"Failed to create a MetricSpec for name={name}, modelversion_id={modelversion_id}. {resp.status_code} {resp.text}")
        return MetricSpec.__parse_json(cluster, resp.json())

    # FIXME: if modelversion upload failed it would not have a servable and will fail here (try runnning fail_upload tests before monitoring tests in tests_modelversions)
    @staticmethod
    def __parse_json(cluster: Cluster, json_dict: dict) -> 'MetricSpec':
        """
        Deserialize MetricSpec from json.

        :param cluster: active cluster
        :param json_dict:
        :return: MetricSpec obj
        """
        return MetricSpec(
            cluster=cluster,
            name=json_dict['name'],
            modelversion_id=json_dict['modelVersionId'],
            config=MetricSpecConfig(
                modelversion_id=json_dict['config']['modelVersionId'],
                threshold=json_dict['config']['threshold'],
                threshold_op=json_dict['config']['thresholdCmpOperator']['kind'],
                servable=json_dict['config']['servable']
            ),
            metric_spec_id=json_dict['id']
        )

    def __init__(self, cluster: Cluster, metric_spec_id: int, name: str, modelversion_id: int,
                 config: MetricSpecConfig) -> 'MetricSpec':
        self.cluster = cluster
        self.metric_spec_id = metric_spec_id
        self.config = config
        self.modelversion_id = modelversion_id
        self.name = name

