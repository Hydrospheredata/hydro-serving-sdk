from enum import Enum


class Monitoring:
    def __init__(self, model, threshold, comparator, name=None):
        self.name = name
        self.model = model
        self.threshold = threshold
        self.comparator = comparator


class CustomModelMetricSpec(Enum):
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
    def __init__(self, model, logs_iterator):
        self.cluster = model.cluster
        self.model = model
        self.model_version_id = model.id
        self.cluster = self.model.cluster
        self._logs_iterator = logs_iterator
        self.last_log = None
        self._status = ""

    def logs(self):
        return self._logs_iterator

    def set_status(self) -> None:
        try:
            self.last_log = next(self.logs()).data
            if self.last_log.startswith("Successfully tagged"):
                self._status = BuildStatus.FINISHED
            else:
                self._status = BuildStatus.BUILDING
        except StopIteration:
            if self.last_log == "":
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
