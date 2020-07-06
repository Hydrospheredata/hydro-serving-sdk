class ServableException(Exception):
    pass


class MetricSpecException(Exception):
    pass


class ApplicationException(Exception):
    pass


class ApplicationNotFoundError(ApplicationException):
    pass


class ApplicationDeletionError(ApplicationException):
    pass


class ApplicationCreationError(ApplicationException):
    pass


class BadResponse(Exception):
    pass


class MetricSpecException(BaseException):
    """
    Metric Spec base exception
    """
    pass
