import json
import logging
from typing import Dict
from typing import Optional
from urllib import parse

import grpc
import requests

from hydrosdk.utils import grpc_server_on, handle_request_error
from hydrosdk.exceptions import BadResponseException


class Cluster:
    """
    Cluster responsible for interactions with the server.

    :Example:

    Create a cluster instance only with HTTP connection.

    >>> cluster = Cluster("http-cluster-endpoint")
    >>> print(cluster.build_info())

    Create a cluster instance with both HTTP and gRPC connection.

    >>> from grpc import ssl_channel_credentials
    >>> grpc_credentials = ssl_channel_credentials()
    >>> cluster = Cluster("http-cluster-endpoint", "grpc-cluster-endpoint", ssl=True, grpc_credentials=grpc_credentials)
    >>> print(cluster.build_info())
    """

    def __init__(self, http_address: str, grpc_address: Optional[str] = None,
                 grpc_credentials: Optional[grpc.ChannelCredentials] = None,
                 grpc_options: Optional[list] = None, grpc_compression: Optional[grpc.Compression] = None,
                 timeout: int = 5, **kwarg) -> 'Cluster':
        """
        A cluster object which hides networking details and provides a connection to a 
        deployed Hydrosphere cluster.

        :param http_address: HTTP endpoint of the cluster
        :param grpc_address: gRPC endpoint of the cluster
        :param grpc_credentials: an optional instance of ChannelCredentials to use for gRPC endpoint
        :param grpc_options: an optional list of key-value pairs to configure the channel
        :param grpc_compression: an optional value indicating the compression method to be
                                 used over the lifetime of the channel
        :param timeout: timeout value to check for gRPC connection
        :returns: Cluster instance
        """

        # TODO: add better url validation (but not python validators lib!)
        parse.urlsplit(http_address)  # check if address is ok
        self.http_address = http_address

        if grpc_address:
            parse.urlsplit(grpc_address)
            self.grpc_address = grpc_address

            if grpc_credentials is not None:
                self.channel = grpc.secure_channel(target=self.grpc_address, credentials=grpc_credentials,
                                                   options=grpc_options,
                                                   compression=grpc_compression)
            else:
                self.channel = grpc.insecure_channel(target=self.grpc_address, options=grpc_options,
                                                     compression=grpc_compression)

            if not grpc_server_on(self.channel, timeout):
                raise ConnectionError(
                    f"Couldn't establish connection with grpc {self.grpc_address}. No connection")
            logging.info(f"Connected to the grpc - {self.grpc_address}")

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Formats address and url, send request

        :param method: type of request
        :param url: url for a request to be sent to
        :param kwargs: additional args
        :return: request res
        """
        res = parse.urlsplit(self.http_address)
        res = res._replace(path=f"{res.path.rstrip('/')}/{url.lstrip('/')}")
        return requests.request(method, parse.urlunsplit(res), **kwargs)

    def build_info(self) -> Dict[str, str]:
        """
        Returns Manager, Gateway and Sonar services builds information containing version, release commit, etc.

        :return: Dictionary with build information
        """
        manager_bl = self.safe_buildinfo("/api/buildinfo")
        gateway_bl = self.safe_buildinfo("/gateway/buildinfo")
        monitoring_bl = self.safe_buildinfo("/monitoring/buildinfo")
        rootcause_bl = self.safe_buildinfo("/rootcause/buildinfo")
        visualization_bl = self.safe_buildinfo("/visualization/buildinfo")
        stat_bl = self.safe_buildinfo("/stat/buildinfo")
        return {
            "manager": manager_bl,
            "gateway": gateway_bl,
            "monitoring": monitoring_bl,
            "rootcase": rootcause_bl,
            "visualization": visualization_bl,
            "stat": stat_bl,
        }

    def safe_buildinfo(self, url: str) -> Dict[str, str]:
        try:
            resp = self.request("GET", url)
            try:
                handle_request_error(
                    resp, f"Can't fetch buildinfo for {url}. {resp.status_code} {resp.text}")
            except BadResponseException as e:
                return {"status": "Unavailable", "reason": str(e)}
            resp_json = resp.json()
            if 'status' not in resp_json:
                resp_json['status'] = "Ok"
            return resp_json
        except requests.exceptions.ConnectionError as ex:
            return {"status": "Unavailable", "reason": f"Can't establish connection. {ex}"}
        except json.decoder.JSONDecodeError as ex:
            return {"status": "Unknown", "reason": f"Can't parse JSON response. {ex}"}
