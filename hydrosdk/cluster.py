import json
import logging
from typing import Optional, Dict
from urllib import parse

import grpc
import requests

from hydrosdk.utils import grpc_server_on


class Cluster:
    """
    Cluster responsible for server interactions
    """

    @staticmethod
    def connect(http_address: str, grpc_address: str = None) -> 'Cluster':
        """
        :param http_address: HTTP connection address of a Hydrosphere cluster
        :param grpc_address: gRPC connection address of a Hydrosphere cluster
        :return: Cluster

        Checks the address, the connectivity, and creates an instance of Cluster.
        """
        cl = Cluster(http_address=http_address, grpc_address=grpc_address)

        logging.info("Connecting to {} cluster".format(cl.http_address))
        info = cl.build_info()
        if info['manager']['status'] != 'Ok':
            raise ConnectionError(
                "Couldn't establish connection with cluster {}. {}".format(http_address, info['manager'].get('reason')))
        logging.info("Connected to the {} cluster".format(info))
        return cl

    def __init__(self, http_address: str,
                 grpc_address: Optional[str] = None, ssl: bool = False,
                 grpc_credentials: Optional[grpc.ChannelCredentials] = None,
                 grpc_options: Optional[Dict] = None, grpc_compression: Optional[grpc.Compression] = None):
        """
        Creates a Cluster object which hides networking details and provides a connection to a deployed Hydrosphere cluster.

        :param http_address: HTTP connection address of a Hydrosphere cluster
        :param grpc_address: gRPC connection address of a Hydrosphere cluster
        :param ssl: Whether to use an SSL-enabled channel for a gRPC connection
        :param grpc_credentials: An optional ChannelCredentials instance
        :param grpc_options: An optional list of key-value pairs (channel args in gRPC Core runtime) to configure the channel
        :param grpc_compression: An optional value indicating the compression method to be used over the lifetime of the channel

        Examples:
            Example of creating a Cluster::

                # Cluster with only an HTTP connection
                Cluster("your-cluster-url")

                # Example of creating a Cluster with an HTTP and a secured gRPC connection::
                from grpc import ssl_channel_credentials
                Cluster("http-cluster-address", grpc_address="grpc-cluster-address", ssl=True, grpc_credentials=ssl_channel_credentials())
        """
        # TODO: add better url validation (but not python validators lib!)
        parse.urlsplit(http_address)  # check if address is ok
        self.http_address = http_address

        if grpc_address:
            parse.urlsplit(grpc_address)
            self.grpc_address = grpc_address

            if ssl:
                if not grpc_credentials:
                    raise ValueError("Missing grpc credentials")

                self.channel = grpc.secure_channel(target=self.grpc_address, credentials=grpc_credentials,
                                                   options=grpc_options,
                                                   compression=grpc_compression)
            else:
                self.channel = grpc.insecure_channel(target=self.grpc_address, options=grpc_options,
                                                     compression=grpc_compression)

            # with grpc.secure_channel it takes a lot of time to check the grpc-connection
            if not grpc_server_on(self.channel):
                raise ConnectionError(
                    "Couldn't establish connection with grpc {}. No connection".format(self.grpc_address))

            logging.info("Connected to the grpc - {}".format(self.grpc_address))

    def request(self, method, url, **kwargs):
        """
        Formats address and url, send request

        :param method: type of request
        :param url: url for a request to be sent to
        :param kwargs: additional args
        :return: request res
        """
        url = parse.urljoin(self.http_address, url)
        return requests.request(method, url, **kwargs)

    def host_selectors(self):
        return []

    def models(self):
        return []

    def servables(self):
        return []

    def applications(self):
        return []

    def build_info(self) -> Dict:
        """
        Returns Manager, Gateway and Sonar services builds information containing version, release commit, etc.

        :return: Dictionary with build information
        """
        manager_bl = self.safe_buildinfo("/api/buildinfo")
        gateway_bl = self.safe_buildinfo("/gateway/buildinfo")
        sonar_bl = self.safe_buildinfo("/monitoring/buildinfo")
        return {
            "manager": manager_bl,
            "gateway": gateway_bl,
            "sonar": sonar_bl
        }

    def safe_buildinfo(self, url):
        try:
            result = self.request("GET", url).json()
            if 'status' not in result:
                result['status'] = "Ok"
            return result
        except requests.exceptions.ConnectionError as ex:
            return {"status": "Unavailable", "reason": "Can't establish connection"}
        except json.decoder.JSONDecodeError as ex:
            return {"status": "Unknown", "reason": "Can't parse JSON response. {}".format(ex)}
