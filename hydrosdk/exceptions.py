import requests


def handle_request_error(resp: requests.Response, error_message: str):
    if resp.ok:
        return None
    if 400 <= resp.status_code < 500:
        raise BadRequest(error_message)
    if 500 <= resp.status_code < 600:
        raise BadResponse(error_message)
    else:
        raise UnknownException(error_message)


class BadRequest(BaseException):
    """
    Exception for client-side errors.
    """
    pass


class BadResponse(BaseException):
    """
    Exception for server-side errors.
    """
    pass


class UnknownException(BaseException):
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
