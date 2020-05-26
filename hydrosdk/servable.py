"""This module contains all the code associated with Servables and their management at Hydrosphere platform.
You can learn more about Servables here https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable.
"""
import re
from enum import Enum
from typing import Dict, List
from urllib.parse import urljoin

import sseclient

from hydrosdk.cluster import Cluster
from hydrosdk.data.types import PredictorDT
from hydrosdk.exceptions import ServableException
from hydrosdk.modelversion import ModelVersion
from hydrosdk.predictor import PredictServiceClient, MonitorableImplementation, UnmonitorableImplementation


class ServableStatus(Enum):
    """
    Servable can be in one of four states
    """
    STARTING = 2
    SERVING = 3
    NOT_SERVING = 0
    NOT_AVAILABLE = 1

    @staticmethod
    def from_camel_case(camel_case_servable_status: str) -> 'ServableStatus':
        """
        Get value of ServableStatus enum from CamelCase style string returned from the manager.
        :param camel_case_servable_status:
        :return: ServableStatus

        :Example:

        ServableStatus.from_camel_case("NotServing")
        >> ServableStatus.NOT_SERVING
        """
        return ServableStatus[re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case_servable_status).upper()]


class Servable:
    """
    Servable is an instance of a model version which could be used in application or by itself as it exposes various endpoints to your model
     version: HTTP, gRPC, and Kafka.
    You can find more about Servables in the documentation https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable
    """
    BASE_URL = "/api/v2/servable"

    @staticmethod
    def model_version_json_to_servable(model_version_json: dict, cluster: Cluster) -> 'Servable':
        """
        Deserializes servable json description into servable object

        :param model_version_json: Servable description in json format
        :param cluster: Cluster connected to Hydrosphere
        :return: Servable object
        """
        modelversion_data = model_version_json['modelVersion']

        modelversion = ModelVersion.from_json(cluster, modelversion_data)
        return Servable(cluster=cluster,
                        modelversion=modelversion,
                        servable_name=model_version_json['fullName'],
                        status=ServableStatus.from_camel_case(model_version_json['status']['status']),
                        status_message=model_version_json['status']['msg'],
                        metadata=modelversion_data['metadata'])

    @staticmethod
    def create(cluster: Cluster,
               model_name: str,
               version: str,
               metadata: Dict[str, str] = None) -> 'Servable':
        """
        Deploy an instance of uploaded model version at your cluster.

        :param cluster: Cluster connected to Hydrosphere
        :param model_name: Name of uploaded model
        :param version: Version of uploaded model
        :param metadata: Information which you can attach to your servable in form of Dict[str, str]
        :raises ServableException:

        :return: servable
        """
        msg = {
            "modelName": model_name,
            "version": version,
            "metadata": metadata
        }

        res = cluster.request(method='POST', url='/api/v2/servable', json=msg)
        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(model_version_json=json_res, cluster=cluster)
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    @staticmethod
    def find(cluster: Cluster, servable_name: str) -> 'Servable':
        """
        Connects to servable at your cluster.

        :param cluster: Cluster connected to Hydrosphere
        :param servable_name:
        :raises ServableException:
        :return: Servable
        """
        res = cluster.request("GET", Servable.BASE_URL + "/{}".format(servable_name))

        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(model_version_json=json_res, cluster=cluster)
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    @staticmethod
    def list(cluster: Cluster) -> List['Servable']:
        """
        Retrieve list of all servables available at your cluster
        :param cluster: Cluster connected to Hydrosphere
        :return: List of all Servables available at your cluster
        """
        res = cluster.request("GET", "/api/v2/servable")

        if res.ok:
            json_res = res.json()
            servables = [Servable.model_version_json_to_servable(model_version_json=servable_json, cluster=cluster) for
                         servable_json in json_res]
            return servables
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    @staticmethod
    def delete(cluster: Cluster, servable_name: str) -> Dict:
        """
        Shut down and delete servable instance.

        :param cluster:
        :param servable_name:
        :raises ServableException: If server returned not 200
        :return: json response from server

        .. warnings also: Use with caution. Predictors previously associated with this servable will not be able to connect to it.
        """
        res = cluster.request("DELETE", "/api/v2/servable/{}".format(servable_name))
        if res.ok:
            return res.json()
        else:
            raise ServableException(f"{res.status_code} : {res.text}")

    def __init__(self, cluster, modelversion, servable_name, status, status_message, metadata=None):
        if metadata is None:
            metadata = {}
        self.modelversion = modelversion
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

    def __str__(self) -> str:
        return f"Servable '{self.name}' for modelversion '{self.modelversion.name}'v{self.modelversion.version}"

    def predictor(self, monitorable=True, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        if monitorable:
            self.impl = MonitorableImplementation(channel=self.cluster.channel, target=self.name)
        else:
            self.impl = UnmonitorableImplementation(channel=self.cluster.channel, target=self.name)

        self.predictor_return_type = return_type

        return PredictServiceClient(impl=self.impl, signature=self.modelversion.contract.predict,
                                    return_type=self.predictor_return_type)
