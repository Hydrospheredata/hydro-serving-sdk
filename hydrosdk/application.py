from collections import namedtuple
from enum import Enum
from typing import List, Optional

from hydro_serving_grpc.contract import ModelSignature

from hydrosdk.cluster import Cluster
from hydrosdk.contract import _signature_dict_to_ModelSignature
from hydrosdk.data.types import PredictorDT
from hydrosdk.predictor import PredictServiceClient, MonitorableImplementation
from hydrosdk.exceptions import handle_request_error


class ApplicationStatus(Enum):
    FAILED = 0
    ASSEMBLING = 1
    READY = 2


class Application:
    """
    An application is an endpoint to reach your models 
    (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications).

    :Example:

    List all applications created on the cluster.
    >>> apps = Application.list_all(cluster)
    >>> for app in apps: 
    >>>     print(app)

    Find an application by name and perfrom a prediction from it. 
    >>> app = Application.find_by_name("my-application")
    >>> pred = app.predictor()
    >>> resp = pred.predict({"my-input": 1})
    """

    @staticmethod
    def list_all(cluster: Cluster) -> List['Application']:
        """
        List all available applications from server.

        :param cluster: active cluster
        :return: deserialized list of application objects
        """
        resp = cluster.request("GET", "/api/v2/application")
        handle_request_error(
            resp, f"Failed to list all applications. {resp.status_code} {resp.text}")
        applications = [Application._app_json_to_app_obj(cluster, app_json) 
                        for app_json in resp.json()]
        return applications

    @staticmethod
    def find_by_name(cluster: Cluster, application_name: str) -> 'Application':
        """
        Search for an application by name. 

        :param cluster: active cluster
        :param application_name: application name
        :return: deserialized application object
        """
        resp = cluster.request("GET", f"/api/v2/application/{application_name}")
        handle_request_error(
            resp, f"Failed to find application by name={application_name}. {resp.status_code} {resp.text}")
        return Application._app_json_to_app_obj(cluster, resp.json())

    @staticmethod
    def delete(cluster: Cluster, application_name: str) -> dict:
        """
        Delete an application by name.

        :param cluster: active cluster
        :param application_name: application name
        :return: response from the server
        """
        resp = cluster.request("DELETE", f"/api/v2/application/{application_name}")
        handle_request_error(
            resp, f"Failed to delete application for name={application_name}. {resp.status_code} {res.text}")
        return resp.json()

    def delete(self) -> dict:
        """
        Delete an application by name.
        
        :return: response from the server
        """
        return Application.delete(self.cluster, self.name)

    def lock_till_ready(self):
        """
        Lock till the application completes deployment.

        :raises Application.DeployFailed: if application failed to be deployed
        :return: None
        """
        events_steam = cluster.request("GET", "/api/v2/events", stream=True)
        events_client = sseclient.SSEClient(events_stream)

        if not self._is_assembling() and self._is_ready(self.status): 
            return None
        try:
            for event in events_client.events():
                if event.event == "ApplicationUpdate":
                    data = json.loads(event.data)
                    if data.get("name") == self.name:
                        if not self._is_ready():
                            raise Application.DeployFailed(f"{self} failed to be deployed.")
                        return None
        finally:
            events_client.close()

    def predictor(self, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        impl = MonitorableImplementation(channel=self.cluster.channel, target=self.name)
        return PredictServiceClient(impl=impl, signature=self.signature, return_type=return_type)

    @staticmethod
    def _app_json_to_app_obj(cluster: Cluster, application_json: dict) -> 'Application':
        """
        Deserialize json into application object. 

        :param cluster: active cluster
        :param application_json: input json with application object fields
        :return: application object
        """
        app_name = application_json.get("name")
        app_execution_graph = application_json.get("executionGraph")
        app_kafka_streaming = application_json.get("kafkaStreaming")
        app_metadata = application_json.get("metadata")
        app_signature = _signature_dict_to_ModelSignature(data=application_json.get("signature"))
        app_status = ApplicationStatus[application_json.get("status").upper()]
        app = Application(cluster=cluster, name=app_name, execution_graph=app_execution_graph, 
                          status=app_status, signature=app_signature, kafka_streaming=app_kafka_streaming, 
                          metadata=app_metadata)
        return app

    def _update_status(self):
        self.status = self.find_by_name(self.cluster, self.name).status
    
    def _is_assembling(self) -> bool:
        self._update_status()
        if self.status == ApplicationStatus.ASSEMBLING.value:
            return True
        return False

    def _is_ready(self) -> bool:
        self._update_status()
        if self.status == ApplicationStatus.READY.value:
            return True
        return False

    def __init__(self, cluster: Cluster, name: str, execution_graph: dict, status: int, 
                 signature: ModelSignature, kafka_streaming: Optional[dict] = None, 
                 metadata: Optional[dict] = None) -> 'Application':
        self.name = name
        self.execution_graph = execution_graph
        self.kafka_streaming = kafka_streaming
        self.metadata = metadata
        self.status = status
        self.signature = signature
        self.cluster = cluster

    def __str__(self):
        return f"Application {self.name}"

    class DeployFailed(BaseException):
        pass
