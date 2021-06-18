"""
This module contains all the code associated with Servables and their management at Hydrosphere platform.
You can learn more about Servables here https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable.
"""
import re, json, datetime
from enum import Enum
from typing import Dict, List, Optional, Iterable

import sseclient
from sseclient import Event

from hydrosdk import DeploymentConfiguration
from hydrosdk.cluster import Cluster
from hydrosdk.data.types import PredictorDT
from hydrosdk.modelversion import ModelVersion
from hydrosdk.predictor import PredictServiceClient, MonitorableServablePredictionService, UnmonitorableServablePredictionService
from hydrosdk.utils import handle_request_error
from hydrosdk.exceptions import TimeoutException


class ServableStatus(Enum):
    """
    Servable can be in one of four states
    """
    STARTING = 2
    SERVING = 3
    NOT_SERVING = 0
    NOT_AVAILABLE = 1
    UNKNOWN = 4

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
    Servable is an instance of a model version which is used within applications. Intendend for
    internal usage only.
    """
    _BASE_URL = "/api/v2/servable"

    @staticmethod
    def list(cluster: Cluster) -> List['Servable']:
        """
        Retrieve a list of all servables available at your cluster

        :param cluster: Hydrosphere cluster
        :return: List of all Servables available at your cluster
        """
        resp = cluster.request("GET", Servable._BASE_URL)
        handle_request_error(
            resp, f"Failed to list servables. {resp.status_code} {resp.text}")
        return [Servable._from_json(cluster, servable_json) for
                servable_json in resp.json()]

    @staticmethod
    def find_by_name(cluster: Cluster, servable_name: str) -> 'Servable':
        """
        Finds a serving servable in a cluster

        :param cluster: active cluster
        :param servable_name: a name of the servable
        :raises ServableException:
        :return: Servable
        """

        resp = cluster.request("GET", f"{Servable._BASE_URL}/{servable_name}")
        handle_request_error(
            resp, f"Failed to find servable for name={servable_name}. {resp.status_code} {resp.text}")
        return Servable._from_json(cluster, resp.json())

    @staticmethod
    def create(cluster: Cluster, model_name: str, version: int,
               metadata: Optional[Dict[str, str]] = None,
               deployment_configuration: Optional[DeploymentConfiguration] = None) -> 'Servable':
        """
        Deploy an instance of uploaded model version at your cluster.

        :param deployment_configuration: k8s configurations used to run this servable
        :param cluster: Cluster connected to Hydrosphere
        :param model_name: Name of uploaded model
        :param version: Version of uploaded model
        :param metadata: Information which you can attach to your servable in a form of Dict[str, str]
        :raises ServableException:
        :return: servable
        """
        msg = {
            "modelName": model_name,
            "version": version,
        }
        if metadata:
            msg['metadata'] = metadata

        if deployment_configuration:
            msg['deploymentConfigName'] = deployment_configuration.name

        resp = cluster.request('POST', Servable._BASE_URL, json=msg)
        handle_request_error(resp, f"Failed to create a servable. {resp.status_code} {resp.text}")
        return Servable._from_json(cluster, resp.json())

    @staticmethod
    def delete(cluster: Cluster, servable_name: str) -> dict:
        """
        Shut down and delete servable instance.

        :param cluster: active cluster
        :param servable_name: name of the servable
        :return: json response from serve

        .. warnings also::
          Use with caution. Predictors previously associated with this servable will not be able to connect to it.

        """
        resp = cluster.request("DELETE", f"{Servable._BASE_URL}/{servable_name}")
        handle_request_error(
            resp, f"Failed to delete the servable with name={servable_name}. {resp.status_code} {resp.text}")
        return resp.json()

    @staticmethod
    def _from_json(cluster: Cluster, servable_json: dict) -> 'Servable':
        """
        Deserializes Servable from JSON into a Servable object

        :param cluster: active cluster
        :param servable_json: Servable description in json format
        :return: Servable object
        """
        model_version_id = servable_json['modelVersionId']
        deployment_configuration_name = servable_json.get("deploymentConfigurationName", None)
        status_message = servable_json.get('statusMessage', None)

        return Servable(cluster=cluster,
                        model_version_id=model_version_id,
                        servable_name=servable_json.get('fullName', 'unknown'),
                        status=ServableStatus.from_camel_case(servable_json.get('status', "Unknown")),
                        status_message=status_message,
                        metadata=servable_json.get('metadata'),
                        deployment_configuration_name=deployment_configuration_name)
    
    def __init__(self, cluster: Cluster,
                 model_version_id: int,
                 servable_name: str,
                 status: ServableStatus,
                 status_message: Optional[str],
                 deployment_configuration_name: Optional[str],
                 metadata: Optional[dict] = None) -> 'Servable':
        self.model_version_id = model_version_id
        self.name = servable_name
        self.meta = metadata or {}
        self.cluster = cluster
        self.status = status
        self.status_message = status_message
        self.deployment_configuration_name = deployment_configuration_name
        self.model_version = ModelVersion.find_by_id(cluster, self.model_version_id)

    def logs(self, follow=False) -> Iterable[Event]:
        if follow:
            url = f"{self._BASE_URL}/{self.name}/logs?follow=true"
            resp = self.cluster.request("GET", url, stream=True)
        else:
            url = f"{self._BASE_URL}/{self.name}/logs"
            resp = self.cluster.request("GET", url)
            handle_request_error(
                resp, f"Failed to retrieve logs for {self}. {resp.status_code} {resp.text}")
        return sseclient.SSEClient(resp).events()

    def lock_while_starting(self, timeout: int = 120) -> 'Servable':
        """ Wait for a servable to become ready. """
        events_stream = self.cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        self.status = Servable.find_by_name(self.cluster, self.name).status
        if self.status is ServableStatus.STARTING: 
            return self
        if self.status in (ServableStatus.NOT_AVAILABLE, ServableStatus.NOT_SERVING):
            raise ValueError('Servable initialization failed')
        try:
            deadline_at = datetime.datetime.now().timestamp() + timeout
            for event in events_client.events():
                if datetime.datetime.now().timestamp() > deadline_at:
                    raise TimeoutException('Time out waiting for a servable to become ready')
                if event.event == "ServableUpdate":
                    data = json.loads(event.data)
                    print(data)
                    if data.get("fullName") == self.name:
                        self.status = ServableStatus.from_camel_case(data.get("status"))
                        if self.status is ServableStatus.SERVING:
                            return self
                        raise ValueError('Servable initialization failed')
        finally:
            events_client.close()

    def __str__(self) -> str:
        return f"Servable '{self.name}' for model_version_id {self.model_version_id}"

    def __repr__(self) -> str:
        return f"Servable {self.name}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.name,
            "status_message": self.status_message,
            "meta": self.meta,
            "model_version": self.model_version.to_dict(),
            "deployment_configuration": self.deployment_configuration.to_dict(),
        }

    def predictor(self, monitorable=True, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """
        Returns a predictor object which is used to pass data into the deployed Servable

        :param monitorable: If True, the data will be shadowed to the monitoring service
        :param return_type: Specifies into which data format should predictor return Servable outputs
        :return:
        """
        if monitorable:
            impl = MonitorableServablePredictionService(channel=self.cluster.channel, target=self.name)
        else:
            impl = UnmonitorableServablePredictionService(channel=self.cluster.channel, target=self.name)

        return PredictServiceClient(impl=impl, signature=self.model_version.signature,
                                    return_type=return_type)
