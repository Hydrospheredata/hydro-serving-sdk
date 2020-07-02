"""This module contains all the code associated with Servables and their management at Hydrosphere platform.
You can learn more about Servables here https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable.
"""
import re
from enum import Enum
from typing import Dict, List, Optional, Iterable
from urllib.parse import urljoin, url

import sseclient
from sseclient import Event

from hydrosdk.cluster import Cluster
from hydrosdk.data.types import PredictorDT
from hydrosdk.exceptions import ServableException, handle_request_error
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
        >>> ServableStatus.from_camel_case("NotServing")
        ServableStatus.NOT_SERVING
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
    def modelversion_json_to_servable(cluster: Cluster, modelversion_json: dict) -> 'Servable':
        """
        Deserializes servable json description into servable object

        :param cluster: Cluster connected to Hydrosphere
        :param model_version_json: Servable description in json format
        :return: Servable object
        """
        modelversion_data = modelversion_json['modelVersion']

        modelversion = ModelVersion.from_json(cluster, modelversion_data)
        return Servable(cluster=cluster,
                        modelversion=modelversion,
                        servable_name=modelversion_json['fullName'],
                        status=ServableStatus.from_camel_case(modelversion_json['status']['status']),
                        status_message=modelversion_json['status']['msg'],
                        metadata=modelversion_data['metadata'])

    @staticmethod
    def create(cluster: Cluster, model_name: str, version: str,
               metadata: dict = None) -> 'Servable':
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

        resp = cluster.request(method='POST', url='/api/v2/servable', json=msg)
        handle_request_error(
            resp, f"Failed to create a servable. {resp.status_code} {resp.text}")
        return Servable.modelversion_json_to_servable(cluster, res.json())

    @staticmethod
    def find(cluster: Cluster, servable_name: str) -> 'Servable':
        """
        Connects to servable at your cluster.

        :param cluster: Cluster connected to Hydrosphere
        :param servable_name:
        :raises ServableException:
        :return: Servable
        """
        resp = cluster.request("GET", Servable.BASE_URL + "/{}".format(servable_name))
        handle_request_error(
            resp, f"Failed to find servable for name={servable_name}. {resp.status_code} {resp.text}")
        return Servable.modelversion_json_to_servable(cluster, res.json())

    @staticmethod
    def list_all(cluster: Cluster) -> List['Servable']:
        """
        Retrieve list of all servables available at your cluster
        :param cluster: Cluster connected to Hydrosphere
        :return: List of all Servables available at your cluster
        """
        resp = cluster.request("GET", "/api/v2/servable")
        handle_request_error(
            resp, f"Failed to list servables. {resp.status_code} {resp.text}")
        return [Servable.modelversion_json_to_servable(cluster, servable_json) for
                servable_json in res.json()]

    @staticmethod
    def delete(cluster: Cluster, servable_name: str) -> dict:
        """
        Shut down and delete servable instance.

        :param cluster:
        :param servable_name:
        :raises ServableException: If server returned not 200
        :return: json response from server

        .. warnings also: Use with caution. Predictors previously associated with this servable will not be able to connect to it.
        """
        resp = cluster.request("DELETE", "/api/v2/servable/{}".format(servable_name))
        handle_request_error(
            resp, f"Failed to delete servable with name={servable_name}. {resp.status_code} {resp.text}")
        return res.json()

    def __init__(self, cluster: Cluster, modelversion: ModelVersion, servable_name: str, 
                 status: str, status_message: str, metadata: Optional[dict] = None) -> 'Servable':
        self.modelversion = modelversion
        self.name = servable_name
        self.meta = metadata or {}
        self.cluster = cluster
        self.status = status
        self.status_message = status_message

    def logs(self, follow=False) -> Iterable[Event]:
        if follow:
            url = urljoin(self.BASE_URL, f"{self.name}/logs?follow=true")
            resp = self.cluster.request("GET", url, stream=True)
        else:
            url = urljoin(self.BASE_URL, f"{self.name}/logs")
            resp = self.cluster.request("GET", url)
        self.handle_request_error(
            resp, f"Failed to retrieve logs for {self}. {resp.status_code} {resp.text}")
        return sseclient.SSEClient(resp).events()

    def __str__(self) -> str:
        return f"Servable '{self.name}' for modelversion {self.modelversion.name}:{self.modelversion.version}"

    def __repr__(self) -> str:
        return f"Servable {self.name}"

    def predictor(self, monitorable=True, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        if monitorable:
            self.impl = MonitorableImplementation(channel=self.cluster.channel, target=self.name)
        else:
            self.impl = UnmonitorableImplementation(channel=self.cluster.channel, target=self.name)

        self.predictor_return_type = return_type
        return PredictServiceClient(impl=self.impl, signature=self.modelversion.contract.predict,
                                    return_type=self.predictor_return_type)
