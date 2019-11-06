from urllib.parse import urljoin
import sseclient
from .predictor import GRPCPredictor, ShadowlessGRPCPredictor

class ServableException(BaseException):
    pass


class Servable:
    BASE_URL = "/api/v2/servable"

    @staticmethod
    def find(servable_name):
        pass

    @staticmethod
    def list_for_model(model_name, model_version):
        pass

    def __init__(self, cluster, model, servable_name, host, port,  metadata=None):
        if metadata is None:
            metadata = {}
        self.model = model
        self.name = servable_name
        self.meta = metadata
        self.cluster = cluster
        self.host = host
        self.port = port

    def remove(self):
        url = urljoin(self.BASE_URL, self.name)
        resp = self.cluster.request(method="DELETE", url=url)
        if resp.ok:
            return resp.json()
        else:
            raise ServableException(resp)

    def predictor(self, secure=False, shadowed=True):
        if secure:
            channel = self.cluster.grpc_secure()
        else:
            channel = self.cluster.grpc_insecure()

        if shadowed:
            predictor = GRPCPredictor(channel, self.name, self.model.contract.predict)
        else:
            predictor = ShadowlessGRPCPredictor(channel, self.name, self.model.contract.predict)

        return predictor

    def logs(self, follow=False):
        if follow:
            url_suffix = "{}/logs?follow=true".format(self.name)
        else:
            url_suffix = "{}/logs".format(self.name)

        url = urljoin(self.BASE_URL, url_suffix)
        resp = self.cluster.request(method="GET", url=url)
        if resp.ok:
            return sseclient.SSEClient(resp).events()
        else:
            raise ServableException(resp)

