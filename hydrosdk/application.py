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
    def __init__(self, name, graph, streaming_config):
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config


class Application:
    @staticmethod
    def create(cluster, local_application):
        pass

    def __init__(self, cluster, id, name, graph, contract, streaming_config):
        self.cluster = cluster
        self.id = id
        self.name = name
        self.graph = graph
        self.streaming_config = streaming_config
        self.contract = contract
