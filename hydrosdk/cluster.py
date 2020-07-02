import json
import logging
from urllib import parse
from typing import Optional

import grpc
import requests

from hydrosdk.utils import grpc_server_on


class Cluster:
    """
    Cluster responsible for interactions with the server.
    """
    def __init__(self, http_address: str, grpc_address: Optional[str] = None, ssl: bool = False,
                 grpc_credentials: Optional[grpc.ChannelCredentials] = None, 
                 grpc_options: Optional[list] = None, grpc_compression=None) -> 'Cluster':
        """
        :param http_address: HTTP endpoint of the cluster
        :param grpc_address: gRPC endpoint of the cluster
        :param ssl: whether to use SSL connection for gRPC endpoint
        :param grpc_credentials: an optional instance of ChannelCredentials to use for gRPC endpoint
        :param grpc_options: an optional list of key-value pairs to configure the channel
        :param grpc_compression: an optional value indicating the compression method to be
        used over the lifetime of the channel
        :returns: Cluster instance
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

    def build_info(self):
        """
        Returns manager, gateway, sonar builds info
        :return: manager, gateway, sonar build infos
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
        """

        :param url:
        :raises ConnectionError: if no connection
        :raises JSONDecodeError: if json is not properly formatted
        :return: result request json
        """
        try:
            result = self.request("GET", url).json()
            if 'status' not in result:
                result['status'] = "Ok"
            return result
        except requests.exceptions.ConnectionError as ex:
            return {"status": "Unavailable", "reason": "Can't establish connection"}
        except json.decoder.JSONDecodeError as ex:
            return {"status": "Unknown", "reason": "Can't parse JSON response. {}".format(ex)}
