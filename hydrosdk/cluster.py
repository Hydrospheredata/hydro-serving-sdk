import logging
from urllib import parse
import requests
import json
import grpc
import functools


class Cluster:
    """
    Cluster responsible for server interactions
    """
    @staticmethod
    def connect(http_address, grpc_address=None):
        """
        The preferable factory method for Cluster creation. Use it.

        :param http_address: connection address
        :raises ConnectionError: if no connection with cluster
        :return: cluster object

        Checks the address, the connectivity, and creates an instance of Cluster.
        """
        cl = Cluster(http_address=http_address, grpc_address=grpc_address)
        logging.info("Connecting to {} cluster".format(cl.http_address))
        info = cl.build_info()
        if info['manager']['status'] != 'Ok':
            raise ConnectionError("Couldn't establish connection with cluster {}. {}".format(http_address, info['manager'].get('reason')))
        logging.info("Connected to the {} cluster".format(info))
        return cl

    def __init__(self, http_address, grpc_address=None):
        """
        Cluster ctor. Don't use it unless you understand what you are doing.
        :param http_address:
        :param grpc_address:
        """
        parse.urlsplit(http_address)  # check if address is ok
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