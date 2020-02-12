import os

import yaml

from hydrosdk.errors import InvalidYAMLFile
from hydrosdk.model import Model, LocalModel


class PipelineUnit:
    def __init__(self, model, weight, servable=None):
        self.model = model
        self.weight = weight
        self.servable = servable


class PipelineStage:
    @staticmethod
    def singular(model):
        return PipelineStage([PipelineUnit(model, 100)])

    @staticmethod
    def weighted(model_with_weights):
        units = [PipelineUnit(m, w) for m, w in model_with_weights]
        return PipelineStage(units)

    def __init__(self, units):
        self.units = units


class LocalApplication:
    @staticmethod
    def from_file(path, cluster):
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

    def __init__(self, name, graph, streaming_config):
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config



class Application:
    @staticmethod
    def create(cluster, local_application):
        pass

    def __init__(self, cluster, id, name, graph, contract, streaming_config, status):
        self.cluster = cluster
        self.id = id
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config
        self.contract = contract
        self.status = status