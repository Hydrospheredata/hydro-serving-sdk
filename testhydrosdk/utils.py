import re
from dataclasses import is_dataclass
from typing import Generator, Dict

import grpc
import requests

from hydrosdk.exceptions import BadRequestException, BadResponseException, UnknownException


def grpc_server_on(channel: grpc.Channel, timeout: int) -> bool:
    """
    The channel_ready_future function allows the client to wait for a specified timeout 
    duration (in seconds) for the server to be ready. If our client times out, it raises.

    :param channel:
    :param timeout: timeout value to check for gRPC connection
    :raises grpc.FutureTimeoutError: if server is off
    :return: status bool
    """
    try:
        grpc.channel_ready_future(channel).result(timeout)
        return True
    except grpc.FutureTimeoutError:
        return False


def handle_request_error(resp: requests.Response, error_message: str):
    """
    Handle http response errors.

    :param resp: response from the cluster
    :param error_message: message of the error to be raised
    :raises BadRequestException: for client-side errors
    :raises BadResponseException: for server-side errors
    """
    if resp.ok:
        return None
    if 400 <= resp.status_code < 500:
        raise BadRequestException(error_message)
    if 500 <= resp.status_code < 600:
        raise BadResponseException(error_message)
    else:
        raise UnknownException(error_message)


def read_in_chunks(filename: str, chunk_size: int) -> Generator[bytes, None, None]:
    """
    Generator to read a file peace by peace.
    
    :param filename: name of the file to read
    :param chunk_size: amount of data to read at once
    """
    with open(filename, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            yield data


def enable_camel_case(cls):
    """
    Decorator used for translating Python dataclasses with snake_named
     attributes to same-named backend entities with camelCase named attributes.
     If attributes are dataclasses, then they are converted to\from camelCase recursively

     e.g.
     @enable_camel_case
     @dataclass
     class A:
        super_variable: int = 0

    >> A().to_camel_case_dict()
    {"superVariable" : 0}

    >> A.from_camel_case_dict({"superVariable": 23})
    A(super_variable=23)

    :param cls:
    :return:
    """

    def to_camel_case(s):
        components = s.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

    def to_snake_case(s):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

    def from_camel_case_dict(d) -> cls:
        if d is None: return None

        keys = list(d.keys())
        camel_to_snake_keys = dict(((k, to_snake_case(k)) for k in keys))

        snake_case_dict = dict()

        for k, v in d.items():
            snake_case_key = camel_to_snake_keys[k]
            item_type = cls.__annotations__[snake_case_key]
            if is_dataclass(item_type):
                snake_case_dict[snake_case_key] = item_type.from_camel_case_dict(v)
            else:
                try:
                    if item_type._name == 'List' and len(item_type.__args__) == 1 and is_dataclass(item_type.__args__[0]):
                        snake_case_dict[snake_case_key] = [item_type.__args__[0].from_camel_case_dict(x) for x in v]
                        continue
                    else:
                        snake_case_dict[snake_case_key] = v
                except AttributeError:
                    snake_case_dict[snake_case_key] = v
                snake_case_dict[snake_case_key] = v

        return cls(**snake_case_dict)

    def to_camel_case_dict(self) -> Dict:
        camel_cased_dict = dict()
        for k, v in self.__dict__.items():
            if v is None or k.startswith("_"): continue
            camel_case_key = to_camel_case(k)
            if is_dataclass(v):
                camel_cased_dict[camel_case_key] = v.to_camel_case_dict()
            elif isinstance(v, list):
                if all(map(is_dataclass, v)):
                    camel_cased_dict[camel_case_key] = [x.to_camel_case_dict() for x in v]
                else:
                    camel_cased_dict[camel_case_key] = v
            else:
                camel_cased_dict[camel_case_key] = v
        return camel_cased_dict

    setattr(cls, "from_camel_case_dict", from_camel_case_dict)
    setattr(cls, "to_camel_case_dict", to_camel_case_dict)
    return cls
