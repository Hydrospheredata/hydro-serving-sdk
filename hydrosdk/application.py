import os
from typing import List, Union, Tuple
from urllib.parse import urljoin

import yaml
from hydro_serving_grpc.contract import ModelContract

from hydrosdk.cluster import Cluster
from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.model import Model, LocalModel


class PipelineUnit:
    def __init__(self, model: Union[Model, LocalModel], weight: int, servable=None):
        self.model = model
        self.weight = weight
        self.servable = servable


class PipelineStage:
    @staticmethod
    def singular(model: Union[Model, LocalModel]):
        return PipelineStage([PipelineUnit(model, 100)])

    @staticmethod
    def weighted(model_with_weights: List[Tuple[Union[Model, LocalModel], int]]):
        units = [PipelineUnit(m, w) for m, w in model_with_weights]
        return PipelineStage(units)

    def __init__(self, units: List[PipelineUnit]):
        self.units = units


class Pipeline:
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages


class StreamingConfig:
    def __init__(self, source_topic, destination_topic, error_topic, consumer_id):
        self.consumer_id = consumer_id
        self.error_topic = error_topic
        self.destination_topic = destination_topic
        self.source_topic = source_topic


class LocalApplication:
    @staticmethod
    def from_file(path: str, cluster: Cluster):
        with open(path, 'r') as f:
            app_docs = [x for x in yaml.safe_load_all(f) if x.get("kind").lower() == "application"]
        if not app_docs:
            raise InvalidYAMLFile(path, "Couldn't find proper documents (kind: application)")
        app_doc = app_docs[0]
        name = app_doc.get('name')
        if not name:
            raise InvalidYAMLFile(path, "name is not defined")
        if app_doc.get('singular'):
            singular = app_doc['singular']
            if singular.get('model'):
                pass
            elif singular.get('path'):
                pass
            else:
                raise Exception("Invalid singular Application definition")
        elif app_doc.get('pipeline'):
            pipeline = app_doc['pipeline']
            for stage in pipeline:
                if isinstance(stage, list):
                    for unit in stage:
                        if unit.get('model'):
                            name, version = unit['model'].split(':')
                            model = Model.find(cluster, name, version)
                        elif unit.get('path'):
                            folder = os.path.dirname(path)  # resolve path to model?
                            model = LocalModel.from_file(unit['path'])
                        else:
                            raise Exception("Invalid pipeline Application definition")
                        weight = unit['weight']
                        PipelineUnit(model, weight)
                else:
                    raise Exception("Invalid pipeline Application definition")
        else:
            raise Exception("Invalid Application definition")

    def __init__(self, name: str, graph: Pipeline, streaming_config: List[StreamingConfig]):
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config

    def create(self, cluster: Cluster):
        return Application.create(cluster, self.name, self.graph, self.streaming_config)


class Application:
    BASE_URL = '/api/v2/application'

    @staticmethod
    def __parse_dict(d: dict):
        return Application()

    @staticmethod
    def create(cluster: Cluster, name: str, graph: Pipeline, streaming_config: List[StreamingConfig]):
        json_dict = {}
        resp = cluster.request('post', Application.BASE_URL, json=json_dict)
        if resp.ok:
            return Application.__parse_dict(resp.json())
        else:
            raise Exception(f"Can't create an Application. Code: {resp.status_code} Reason: {resp.text}")

    @staticmethod
    def list(cluster: Cluster) -> List['Application']:
        resp = cluster.request('get', Application.BASE_URL)
        if resp.ok:
            return [Application.__parse_dict(x) for x in resp.json()]
        else:
            raise Exception(f"Can't get the list of Applications. Code: {resp.status_code} Reason: {resp.text}")

    @staticmethod
    def find_by_id(cluster: Cluster, id: int):
        apps_with_id = [x for x in Application.list(cluster) if x.id == id]
        if apps_with_id:
            return apps_with_id[0]
        else:
            raise Exception(f"Can't find an Application with id {id}")

    @staticmethod
    def get_by_name(cluster: Cluster, name: str):
        url = urljoin(Application.BASE_URL, name)
        resp = cluster.request('get', url)
        if resp.ok:
            return Application.__parse_dict(resp.json())
        else:
            raise Exception(f"Can't find an Application with name {name}. Code: {resp.status_code} Reason: {resp.text}")

    def __init__(self, cluster: Cluster, id: int, name: str, graph: Pipeline, contract: ModelContract, streaming_config: List[StreamingConfig], status: str):
        self.cluster = cluster
        self.id = id
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config
        self.contract = contract
        self.status = status

    def update(self, graph: Pipeline, streaming_config: StreamingConfig):
        pass

    def delete(self):
        url = urljoin(Application.BASE_URL, self.name)
        resp = self.cluster.request('DELETE', url)
        if resp.ok:
            return self
        else:
            raise Exception(f"Can't delete an Application with id {self.name}. Code: {resp.status_code} Reason: {resp.text}")