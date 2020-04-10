from enum import Enum
from urllib.parse import urljoin

import sseclient

from .exceptions import ServableException
from .model import Model


class ServableStatus(Enum):
    """
    # TODO add more information about statuses
    Servable can be in one of four states:
        1. STARTING -
        2. SERVING -
        3. NOT_SERVING -
        4. NOT_AVAILABLE -
    """
    STARTING = 0
    SERVING = 1
    NOT_SERVING = 2
    NOT_AVAILABLE = 3


class Servable:
    """
    Servable is an instance of a model version which could be used in application or by itself as it exposes various endpoints to your model version: HTTP, gRPC, and Kafka.
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable)
    """
    BASE_URL = "/api/v2/servable"

    @staticmethod
    def model_version_json_to_servable(mv_json: dict, cluster):
        """
        Deserializes model version json to servable object

        :param mv_json: model version json
        :param cluster: active cluster
        :return: servable object
        """
        model_data = mv_json['modelVersion']

        model = Model.from_json(cluster, model_data)
        return Servable(cluster=cluster,
                        model=model,
                        servable_name=mv_json['fullName'],
                        status=ServableStatus[mv_json['status']['status'].upper()],
                        status_message=mv_json['status']['msg'],
                        metadata=model_data['metadata'])

    @staticmethod
    def create(cluster, model_name, model_version, metadata=None):
        """
        Sends request to server and returns servable object

        :param cluster:
        :param model_name:
        :param model_version:
        :param metadata:
        :raises ServableException: If server returned not 200

        :return: servable
        """
        msg = {
            "modelName": model_name,
            "version": model_version,
            "metadata": metadata
        }

        res = cluster.request(method='POST', url='/api/v2/servable', json=msg)
        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(mv_json=json_res, cluster=cluster)
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    @staticmethod
    def get(cluster, servable_name):
        """
        Sends request to server and return servable object by name

        :param cluster: active cluster
        :param servable_name:
        :raises ServableException: If server returned not 200
        :return: servable
        """
        res = cluster.request("GET", Servable.BASE_URL + "/{}".format(servable_name))

        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(mv_json=json_res, cluster=cluster)
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    @staticmethod
    def list(cluster):
        """
        Sends request to server and returns list of all servables
        :param cluster: active cluster
        :return: json with request result
        """
        return cluster.request("GET", "/api/v2/servable").json()

    @staticmethod
    def delete(cluster, servable_name):
        """
        Sends request to delete servable by name

        :param cluster:
        :param servable_name:
        :raises ServableException: If server returned not 200
        :return: json response from server
        """
        res = cluster.request("DELETE", "/api/v2/servable/{}".format(servable_name))
        if res.ok:
            return res.json()
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    def __init__(self, cluster, model, servable_name, status, status_message, metadata=None):
        if metadata is None:
            metadata = {}
        self.model = model
        self.name = servable_name
        self.meta = metadata
        self.cluster = cluster
        self.status = status
        self.status_message = status_message

    # TODO: method not used
    def logs(self, follow=False):
        if follow:
            url_suffix = "{}/logs?follow=true".format(self.name)
        else:
            url_suffix = "{}/logs".format(self.name)

        url = urljoin(self.BASE_URL, url_suffix)
        res = self.cluster.request(method="GET", url=url)
        if res.ok:
            return sseclient.SSEClient(res).events()
        else:
            raise ServableException(f"{res.status_code} : {res.text}")
