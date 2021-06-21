class HydrosphereException(Exception):
    """ 
    Base exception for SDK library.
    """
    pass


class BadRequestException(HydrosphereException):
    """
    Exception for client-side errors.
    """
    pass


class BadResponseException(HydrosphereException):
    """
`    Exception for server-side errors.
    """
    pass


class TimeoutException(HydrosphereException):
    """
    Exception for timeouts
    """
    pass


class UnknownException(HydrosphereException):
    pass


class ServableException(HydrosphereException):
    """
    Servable class base exception
    """
    pass


class MetricSpecException(HydrosphereException):
    pass


class SignatureViolationException(HydrosphereException):
    """
    Exception raised when signature is violated
    """
    pass
