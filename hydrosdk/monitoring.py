from typing import Union
from urllib.parse import urljoin

from hydrosdk.cluster import Cluster
from hydrosdk.exceptions import MetricSpecException


class TresholdCmpOp:
    EQ = "Eq"
    NOT_EQ = "NotEq"
    GREATER = "Gt"
    GREATER_EQ = "GtEq"
    LESS = "Less"
    LESS_EQ = "LessEq"


class MetricModel:
    def __init__(self, model, threshold, comparator):
        self.model = model
        self.threshold = threshold
        self.comparator = comparator


class MetricSpecConfig:
    def __init__(self, model_version_id: int, threshold: Union[int, float], threshold_op: TresholdCmpOp, servable=None):
        self.servable = servable
        self.threshold_op = threshold_op
        self.threshold = threshold
        self.model_version_id = model_version_id


class MetricSpec:
    BASE_URL = "/api/v2/monitoring/metricspec"
    GET_SPEC_URL = BASE_URL + "/"

    @staticmethod
    def __parse_json(cluster, json_dict):
        return MetricSpec(
            cluster=cluster,
            name=json_dict['name'],
            model_version_id=json_dict['modelVersionId'],
            config=MetricSpecConfig(
                model_version_id=json_dict['config']['modelVersionId'],
                threshold=json_dict['config']['threshold'],
                threshold_op=json_dict['config']['thresholdCmpOperator']['kind'],
                servable=json_dict['config']['servable']
            ),
            metric_spec_id=json_dict['id']
        )

    @staticmethod
    def create(cluster: Cluster, name: str, model_version_id: int, config: MetricSpecConfig):
        d = {
            'name': name,
            'modelVersionId': model_version_id,
            'config': {
                'modelVersionId': config.model_version_id,
                'threshold': config.threshold,
                'thresholdCmpOperator': {
                    'kind': config.threshold_op
                }
            }
        }
        resp = cluster.request("post", MetricSpec.BASE_URL, json=d)
        if resp.ok:
            return MetricSpec.__parse_json(cluster, resp.json())
        else:
            raise MetricSpecException(
                f"Failed to create a MetricSpec. Name={name}, model_version_id={model_version_id}. {resp.status_code} {resp.text}")

    @staticmethod
    def list_all(cluster: Cluster):
        resp = cluster.request("get", MetricSpec.BASE_URL)
        if resp.ok:
            return [MetricSpec.__parse_json(cluster, x) for x in resp.json()]
        else:
            raise MetricSpecException(f"Failed to list MetricSpecs. {resp.status_code} {resp.text}")

    @staticmethod
    def list_for_model(cluster: Cluster, model_version_id: int):
        url = urljoin(MetricSpec.GET_SPEC_URL, f"modelversion/{model_version_id}")
        print(url)
        resp = cluster.request("get", url)
        if resp.ok:
            return [MetricSpec.__parse_json(cluster, x) for x in resp.json()]
        else:
            raise MetricSpecException(
                f"Failed to list MetricSpecs for model_version_id={model_version_id}. {resp.status_code} {resp.text}")

    @staticmethod
    def get(cluster: Cluster, metric_spec_id: int):
        url = urljoin(MetricSpec.BASE_URL, str(metric_spec_id))
        resp = cluster.request("get", url)
        if resp.ok:
            return MetricSpec.__parse_json(cluster, resp.json())
        elif resp.status_code == 404:
            return None
        else:
            raise MetricSpecException(
                f"Failed to list MetricSpecs for metric_spec_id={metric_spec_id}. {resp.status_code} {resp.text}")

    def __init__(self, cluster: Cluster, metric_spec_id: int, name: str, model_version_id: int,
                 config: MetricSpecConfig):
        self.cluster = cluster
        self.metric_spec_id = metric_spec_id
        self.config = config
        self.model_version_id = model_version_id
        self.name = name

    def delete(self):
        url = urljoin(MetricSpec.BASE_URL, str(self.metric_spec_id))
        resp = self.cluster.request("delete", url)
        if resp.ok:
            return True
        elif resp.status_code == 404:
            return False
        else:
            raise MetricSpecException(
                f"Failed to delete MetricSpecs for metric_spec_id={self.metric_spec_id}. {resp.status_code} {resp.text}")
