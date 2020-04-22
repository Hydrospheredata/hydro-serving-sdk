import json
import logging
from urllib import parse

import grpc
import requests


class Cluster:
    """
    Cluster responsible for server interactions
    """

    @staticmethod
    def connect(http_address: str, grpc_address: str = None) -> 'Cluster':
        """
        The preferable factory method for Cluster creation. Use it.

        :param http_address: http connection address
        :param grpc_address: optional grpc connection address
        :return: cluster object

        Checks the address, the connectivity, and creates an instance of Cluster.
        """
        cl = Cluster(http_address=http_address, grpc_address=grpc_address)

        Cluster.check_connection(cluster=cl, address=http_address, address_type="http")
        if grpc_address:
            Cluster.check_connection(cluster=cl, address=grpc_address, address_type="grpc")

        return cl

    @staticmethod
    def check_connection(cluster: 'Cluster', address: str, address_type: str) -> None:
        """
        Checks cluster connection
        :param cluster: active cluster
        :param address: address to check
        :param address_type: grpc or http
        :raises ConnectionError: if no connection with cluster
        :return: None
        """
        logging.info("Connecting to {} cluster".format(address))

        if address_type == "http":
            info = cluster.build_info(manager=True, sonar=True, gateway=True)
        elif address_type == "grpc":
            info = cluster.build_info(manager=True, gateway=True)

        if info['manager']['status'] != 'Ok':
            raise ConnectionError(
                "Couldn't establish connection with cluster {}. {}".format(address,
                                                                           info['manager'].get('reason')))
        logging.info("Connected to the {} cluster".format(info))

    def __init__(self, http_address, grpc_address=None, ssl=True,
                 grpc_credentials=None, grpc_options=None, grpc_compression=None):
        """
        Cluster ctor. Don't use it unless you understand what you are doing.
        :param http_address:
        :param grpc_address:
        """
        if ssl:
            self.channel = self.cluster.grpc_secure(credentials=grpc_credentials, options=grpc_options,
                                                    compression=grpc_compression)
        else:
            self.channel = self.cluster.grpc_insecure(options=grpc_options, compression=grpc_compression)

        # TODO: add better validation (but not python validators lib!)
        parse.urlsplit(http_address)  # check if address is ok
        if grpc_address:
            parse.urlsplit(grpc_address)

        self.http_address = http_address
        self.grpc_address = grpc_address

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

    def grpc_secure(self, credentials=None, options=None, compression=None):
        """
        Validates credentials and returns grpc secure channel

        :param credentials:
        :param options:
        :param compression:
        :return: grpc secure channel
        """
        if credentials is None:
            credentials = grpc.ssl_channel_credentials()
        if not self.grpc_address:
            raise ValueError("Grpc address is not set")
        return grpc.secure_channel(self.grpc_address, credentials, options=options, compression=compression)

    def grpc_insecure(self, options=None, compression=None):
        """
        Returns grpc insecure channel

        :param options:
        :param compression:
        :return: grpc insecure channel
        """
        if not self.grpc_address:
            raise ValueError("Grpc address is not set")

        return grpc.insecure_channel(self.grpc_address, options=options, compression=compression)

    def host_selectors(self):
        return []

    def models(self):
        return []

    def servables(self):
        return []

    def applications(self):
        return []

    def build_info(self, manager=True, gateway=True, sonar=True) -> dict:
        """
        Returns manager, gateway, sonar builds info
        :param manager: return manager status or not
        :param sonar: return sonar status or not
        :param gateway: return gateway status or not
        :return: select build infos
        """
        build_info = {}
        if manager:
            manager_bl = self.safe_buildinfo("/api/buildinfo")
            build_info["manager"] = manager_bl
        if gateway:
            gateway_bl = self.safe_buildinfo("/gateway/buildinfo")
            build_info["gateway"] = gateway_bl
        if sonar:
            sonar_bl = self.safe_buildinfo("/monitoring/buildinfo")
            build_info["sonar"] = sonar_bl

        return build_info

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
