from urllib.parse import urljoin

import sseclient

from .predictor import GRPCPredictor, ShadowlessGRPCPredictor


class ServableException(BaseException):
    pass


class Servable:
    BASE_URL = "/api/v2/servable"

    def get(self, servable_name):
        res = self.cluster.request("GET", self.BASE_URL + "/{}".format(servable_name))
        print(res)
        if res.ok:
            return res.json()
        else:
            return None

    # def list_for_model(self, model_name, model_version):
    #     res = self.cluster.request("GET", self.BASE_URL)
    #     if res.ok:
    #         return res.json()
    #     else:
    #         return None

    def create(self, model_name, model_version):
        msg = {
            "modelName": model_name,
            "version": model_version
        }
        res = self.cluster.request(method='POST', url='/api/v2/servable', json=msg)
        if res.ok:
            return res.json()
        else:
            raise Exception(res.content)

    def __init__(self, cluster, model, servable_name, metadata=None):
        if metadata is None:
            metadata = {}
        self.model = model
        self.name = servable_name
        self.meta = metadata
        self.cluster = cluster

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

    def list(self):
        return self.cluster.request("GET", "/api/v2/servable").json()

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

    def delete(self, servable_name):
        res = self.cluster.request("DELETE", "/api/v2/servable/{}".format(servable_name))
        if res.ok:
            return res.json()
        else:
            raise ServableException(res)
