from urllib.parse import urljoin

import sseclient

from .contract import contract_from_dict
from .image import DockerImage
from .model import Model


class ServableException(BaseException):
    pass


class Servable:
    BASE_URL = "/api/v2/servable"

    @staticmethod
    def model_version_json_to_servable(mv_json: dict, cluster):
        model_data = mv_json['modelVersion']
        model = Model(
            id=model_data['model']['id'],
            name=model_data['model']['name'],
            version=model_data['modelVersion'],
            contract=contract_from_dict(model_data.get('modelContract')),
            runtime=model_data['runtime'],
            image=DockerImage(model_data['image'].get('name'), model_data['image'].get('tag'),
                              model_data['image'].get('sha256')),
            cluster=cluster,
            metadata=model_data['metadata'],
            install_command=model_data.get('installCommand'))
        return Servable(cluster=cluster, model=model, servable_name=mv_json['fullName'],
                        metadata=model_data['metadata'])

    @staticmethod
    def create(cluster, model_name, model_version):
        msg = {
            "modelName": model_name,
            "version": model_version
        }
        res = cluster.request(method='POST', url='/api/v2/servable', json=msg)
        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(mv_json=json_res, cluster=cluster)
        else:
            raise ServableException(res)

    @staticmethod
    def get(cluster, servable_name):
        res = cluster.request("GET", Servable.BASE_URL + "/{}".format(servable_name))

        if res.ok:
            json_res = res.json()
            return Servable.model_version_json_to_servable(mv_json=json_res, cluster=cluster)
        else:
            raise ServableException(res)

    @staticmethod
    def list(cluster):
        return cluster.request("GET", "/api/v2/servable").json()

    @staticmethod
    def delete(cluster, servable_name):
        res = cluster.request("DELETE", "/api/v2/servable/{}".format(servable_name))
        if res.ok:
            return res.json()
        else:
            raise ServableException(res)

    def __init__(self, cluster, model, servable_name, metadata=None):
        if metadata is None:
            metadata = {}
        self.model = model
        self.name = servable_name
        self.meta = metadata
        self.cluster = cluster

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

