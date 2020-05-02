import grpc


def grpc_server_on(channel: grpc.Channel) -> bool:
    """
    The channel_ready_future function allows the client to wait for a specified timeout duration (in seconds) for the server to be ready. If our client times out, it raises.
    :param channel:
    :raises grpc.FutureTimeoutError: if server is off
    :return: status bool
    """
    try:
        grpc.channel_ready_future(channel).result()
        return True
    except grpc.FutureTimeoutError:
        return False
