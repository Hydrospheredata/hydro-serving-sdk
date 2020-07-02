from typing import Generator

import grpc
import requests
from hydrosdk.exceptions import BadRequest, BadResponse, UnknownException


def grpc_server_on(channel: grpc.Channel) -> bool:
    """
    The channel_ready_future function allows the client to wait for a specified timeout 
    duration (in seconds) for the server to be ready. If our client times out, it raises.

    :param channel:
    :raises grpc.FutureTimeoutError: if server is off
    :return: status bool
    """
    try:
        grpc.channel_ready_future(channel).result()
        return True
    except grpc.FutureTimeoutError:
        return False


def handle_request_error(resp: requests.Response, error_message: str):
    """
    Handle http response errors.

    :param resp: response from the cluster
    :param error_message: message of the error to be raised
    :raises BadRequest: for client-side errors
    :raises BadResponse: for server-side errors
    """
    if resp.ok:
        return None
    if 400 <= resp.status_code < 500:
        raise BadRequest(error_message)
    if 500 <= resp.status_code < 600:
        raise BadResponse(error_message)
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