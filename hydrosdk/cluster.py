import logging
from urllib import parse
import requests

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
        logging.info("Connected to the {} cluster".format(info))

    def __init__(self, address):
        """
        Cluster ctor. Don't use it unless you understand what you are doing.
        """
        parse.urlsplit(address)  # check if address is ok
        self.address = address
        self.grpc_channel = None
        self.manager_stub = None
        self.gateway_stub = None
        self.predcition_stub = None

    def request(self, method, url, **kwargs):
        url = parse.urljoin(self.address, url)
        return requests.request(method, url, **kwargs)

    def host_selectors():
        pass

    def models(self, ):
        pass

    def servables(self, ):
        pass

    def applications(self, ):
        pass

    def build_info(self):
        pass
