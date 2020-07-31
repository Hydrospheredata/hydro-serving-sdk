import json

import sseclient

from hydrosdk.servable import Servable, ServableStatus
from hydrosdk.application import Application, ApplicationStatus
from hydrosdk.cluster import Cluster


def servable_lock_till_serving(cluster: Cluster, servable_name: str, timeout_messages: int = 30) -> bool:
    """ Wait for a servable to become SERVING """
    events_stream = cluster.request("GET", "/api/v2/events", stream=True)
    events_client = sseclient.SSEClient(events_stream)
    status = Servable.find_by_name(cluster, servable_name).status
    if not status is ServableStatus.STARTING and \
            ServableStatus.SERVING: 
        return None
    try:
        for event in events_client.events():
            timeout_messages -= 1
            if timeout_messages < 0:
                raise ValueError
            if event.event == "ServableUpdate":
                data = json.loads(event.data)
                if data.get("fullName") == servable_name:
                    status = ServableStatus.from_camel_case(data.get("status", {}).get("status"))
                    if status is ServableStatus.SERVING:
                        return None
                    raise ValueError
    finally:
        events_client.close()


def application_lock_till_ready(cluster: Cluster, application_name: str, timeout_messages: int = 30) -> bool:
    """ Wait for a servable to become SERVING """
    events_stream = cluster.request("GET", "/api/v2/events", stream=True)
    events_client = sseclient.SSEClient(events_stream)
    status = Application.find(cluster, application_name).status
    if not status is ApplicationStatus.ASSEMBLING and \
            ApplicationStatus.READY: 
        return None
    try:
        for event in events_client.events():
            timeout_messages -= 1
            if timeout_messages < 0:
                raise ValueError
            if event.event == "ApplicationUpdate":
                data = json.loads(event.data)
                if data.get("name") == application_name:
                    status = ApplicationStatus[data.get("status").upper()]
                    if status is ApplicationStatus.READY:
                        return None
                    raise ValueError
    finally:
        events_client.close()
