import json
import logging
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
        The preferable factory method for Cluster creation. Use it.

        :param http_address: http connection address
        :param grpc_address: optional grpc connection address
        :return: cluster object

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

    def __init__(self, http_address, grpc_address=None, ssl=False,
                 grpc_credentials=None, grpc_options=None, grpc_compression=None):
        """
        Cluster ctor. Don't use it unless you understand what you are doing.
        :param http_address:
        :param grpc_address:
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
