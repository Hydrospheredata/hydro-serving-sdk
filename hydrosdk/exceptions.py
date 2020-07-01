import requests
from hydrosdk.cluster import Cluster


class RequestsErrorHandler:

    @classmethod
    def handle_request_error(cls, resp: requests.Response, error_message: str):
        if resp.ok:
            return
        if 400 <= resp.status_code < 500:
            raise cls.BadRequest(error_message)
        if 500 <= resp.status_code < 600:
            raise Cluster.BadResponse(error_message)
        else:
            raise Cluster.UnknownException(error_message)

    class BadRequest(Exception):
        """
        Fallback BadRequest exception.
        """
        pass


class ServableException(BaseException):
    """
    Servable class base exception
    """
    pass


class MetricSpecException(BaseException):
    """
    Metric Spec base exception
    """
    pass