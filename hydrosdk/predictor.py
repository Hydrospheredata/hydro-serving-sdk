import abc
import requests
import grpc
import hydro_serving_grpc as hsg
import hydro_serving_grpc.gateway as hsgateway

class AbstractPredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, data):
        pass

class JSONPredictor(AbstractPredictor):
    def __init__(self, url):
        self.url = url

    def predict(self, data):
        return requests.post(url=self.url, json=data)

class GRPCPredictor(AbstractPredictor):

    def secure(url, credentials=None, options=None, compression=None):
        channel = grpc.secure_channel(url, credentials, options=options, compression=compression)
        return GRPCPredictor(channel)

    def insecure(url, options=None, compression=None):
        channel = grpc.insecure_channel(url, options=options, compression=compression)
        return GRPCPredictor(channel)

    def __init__(self, channel):
        self.url = url
        self.channel = channel
        self.stub = hsg.PredictionServiceStub(self.channel)

    def predict(self, data):
        return self.stub.Predict(data)

class ShadowlessGRPCPredictor(GRPCPredictor):
    def __init__(self, channel):
        super().__init__(channel)
        self.stub = hsgateway.GatewayServiceStub(self.channel)

    def predict(self, data):
        return self.stub.ShadowlessPredictServable(data)