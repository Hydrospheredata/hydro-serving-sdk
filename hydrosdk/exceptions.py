class BadRequest(Exception):
    """
    Exception for client-side errors.
    """
    pass


class BadResponse(Exception):
    """
    Exception for server-side errors.
    """
    pass


class UnknownException(Exception):
    pass


class ServableException(Exception):
    """
    Servable class base exception
    """
    pass


class MetricSpecException(Exception):
    pass


class ContractViolationException(Exception):
    """
    Exception raised when contract is violated
    """
    pass
