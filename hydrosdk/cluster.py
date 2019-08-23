from urllib import parse
import logging

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
        parse.urlsplit(address) # check if address is ok
        self.address = address

    def host_selectors():
        pass

    def models():
        pass

    def servables():
        pass

    def applications():
        pass

    def build_info(self):
        pass