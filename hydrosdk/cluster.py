import logging
from urllib import parse
import requests
import json
import grpc
import functools


class Cluster:
    @staticmethod
    def connect(address):
        """
        The preferrable factory method for Cluster creation. Use it.

        Checks the address, the connectivity, and creates an instance of Cluster.
        """
        cl = Cluster(address)
        logging.info("Connecting to {} cluster".format(cl.address))
        info = cl.build_info()
        if info['manager']['status'] != 'Ok':
            raise ConnectionError("Couldn't establish connection with cluster {}. {}".format(address, info['manager'].get('reason')))
        logging.info("Connected to the {} cluster".format(info))
        return cl

    def __init__(self, address):
        """
        Cluster ctor. Don't use it unless you understand what you are doing.
        """
        parse.urlsplit(address)  # check if address is ok
        self.address = address

    def request(self, method, url, **kwargs):
        url = parse.urljoin(self.address, url)
        return requests.request(method, url, **kwargs)

    def grpc_secure(self, credentials=None, options=None, compression=None):
        if credentials is None:
            credentials = grpc.ChannelCredentials(None)
        return grpc.secure_channel(self.address, credentials, options=options, compression=compression)

    def grpc_insecure(self, options=None, compression=None):
        return grpc.insecure_channel(self.address, options=options, compression=compression)

    def host_selectors(self):
        return []

    def models(self):
        return []

    def servables(self):
        return []

    def applications(self):
        return []

    def build_info(self):
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