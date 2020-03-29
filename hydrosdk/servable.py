from urllib.parse import urljoin

import sseclient

from .contract import contract_from_dict
from .exceptions import ServableException
from .image import DockerImage
from .model import Model


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
        model = Model(
            id=model_data['model']['id'],
            name=model_data['model']['name'],
            version=model_data['modelVersion'],
            contract=contract_from_dict(model_data.get('modelContract')),
            runtime=model_data['runtime'],
            image=DockerImage(model_data['image'].get('name'), model_data['image'].get('tag'),
                              model_data['image'].get('sha256')),
            cluster=cluster,
            metadata=model_data['metadata'],
            install_command=model_data.get('installCommand'))
        return Servable(cluster=cluster, model=model, servable_name=mv_json['fullName'],
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

    def __init__(self, cluster, model, servable_name, metadata=None):
        if metadata is None:
            metadata = {}
        self.model = model
        self.name = servable_name
        self.meta = metadata
        self.cluster = cluster

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

