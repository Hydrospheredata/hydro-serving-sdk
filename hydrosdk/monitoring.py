import logging
from enum import Enum

import sseclient


class Monitoring:
    def __init__(self, model, threshold, comparator, name=None):
        self.name = name
        self.model = model
        self.threshold = threshold
        self.comparator = comparator


class MetricSpec(Enum):
    EQ = "Eq"
    NOT_EQ = "NotEq"
    GREATER = "Gt"
    GREATER_EQ = "GtEq"
    LESS = "Less"
    LESS_EQ = "LessEq"


class BuildStatus(Enum):
    BUILDING = "BUILDING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class UploadResponse:
    def __init__(self, model, version_id):
        self.cluster = model.cluster
        self.model = model
        self.model_version_id = version_id
        self.cluster = self.model.cluster
        self._logs_iterator = self.logs()
        self.last_log = ""
        self._status = ""

    def logs(self):
        logger = logging.getLogger("ModelDeploy")
        try:
            url = "/api/v2/model/version/{}/logs".format(self.model_version_id)
            logs_response = self.model.cluster.request("GET", url, stream=True)
            self._logs_iterator = sseclient.SSEClient(logs_response).events()
        except RuntimeError:
            logger.exception("Unable to get build logs")
            self._logs_iterator = None
        return self._logs_iterator

    def set_status(self) -> None:
        try:
            if self.last_log.startswith("Successfully tagged"):
                self._status = BuildStatus.FINISHED
            else:
                self.last_log = next(self._logs_iterator).data
                self._status = BuildStatus.BUILDING
        except StopIteration:
            if not self._status == BuildStatus.FINISHED:
                self._status = BuildStatus.FAILED

    def get_status(self):
        return self._status

    def not_ok(self) -> bool:
        self.set_status()
        return self.get_status() == BuildStatus.FAILED

    def ok(self):
        self.set_status()
        return self.get_status() == BuildStatus.FINISHED

    def building(self):
        self.set_status()
        return self.get_status() == BuildStatus.BUILDING

    def request_model(self):
        return self.cluster.request("GET", f"api/v2/model/{self.model.id}")
