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


class ContractViolationException(Exception):
    """
    Exception raised when contract is violated
    """
    pass

