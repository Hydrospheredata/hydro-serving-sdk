import abc
import requests
import hydro_serving_grpc as hsg
import hydro_serving_grpc.gateway as hsgateway


class AbstractPredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, data):
        pass


class JSONPredictor(AbstractPredictor):
    def __init__(self, url, target_name, signature):
        self.url = url

    def predict(self, data):
        return requests.post(url=self.url, json=data)


class GRPCPredictor(AbstractPredictor):

    def __init__(self, channel, target_name, signature):
        self.channel = channel
        self.target_name = target_name
        self.contract = signature
        self.stub = hsg.PredictionServiceStub(self.channel)

    def predict(self, data):
        return self.stub.Predict(data)


class ShadowlessGRPCPredictor(GRPCPredictor):
    def __init__(self, channel, target_name, signature):
        super().__init__(channel, target_name, signature)
        self.stub = hsgateway.GatewayServiceStub(self.channel)

    def predict(self, data):
        return self.stub.ShadowlessPredictServable(data)

    # def __call__(self, profile=True, *args, **kwargs):
    #     input_tensors = []
    #
    #     for arg in args:
    #         input_tensors.extend(decompose_arg_to_tensors(arg))
    #
    #     for key, arg in kwargs.items():
    #         input_tensors.append(decompose_kwarg_to_tensor(key, arg))
    #
    #     is_valid, error_msg = self.model.contract.signature.validate_input(input_tensors)
    #     if not is_valid:
    #         return ContractViolationException(error_msg)
    #
    #     input_proto_dict = dict((map(lambda x: (x.name, x.proto), input_tensors)))
    #     predict_request = ServablePredictRequest(servable_name=self.name, data=input_proto_dict)
    #
    #     if profile:
    #         result = self.model.cluster.gateway_stub.PredictServable(predict_request)
    #     else:
    #         result = self.model.cluster.gateway_stub.ShadowlessPredictServable(predict_request)
    #
    #     output_tensors = []
    #     for tensor_name, tensor_proto in result.outputs:
    #         output_tensors.append(Tensor.from_proto(tensor_name, tensor_proto))
    #
    #     is_valid, error_msg = self.model.contract.signature.validate_output(output_tensors)
    #     if not is_valid:
    #         warnings.warn("Output is not valid.\n" + error_msg)
    #
    #     return output_tensors
