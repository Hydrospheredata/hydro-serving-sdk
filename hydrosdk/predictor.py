from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2
from hydro_serving_grpc.contract import ModelSignature
from hydro_serving_grpc.gateway import GatewayServiceStub

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto


class PredictImplementation(ABC):
    @abstractmethod
    def send_data(self, model_spec, inputs):
        pass


class MetricableImplementation(PredictImplementation):
    def __init__(self, channel):
        self.stub = PredictionServiceStub(channel)

    def send_data(self, model_spec, inputs):
        return self.stub.Predict(model_spec, inputs)


class UnmetricableImplementation(PredictImplementation):
    def __init__(self, channel):
        self.stub = GatewayServiceStub(channel)

    def send_data(self, model_spec, inputs):
        return self.stub.ShadowlessPredictServable(model_spec, inputs)


def is_dataframe(obj):
    return isinstance(obj, pd.DataFrame)


def is_dict(obj):
    return isinstance(obj, dict)


def is_pdSeries(obj):
    return isinstance(obj, pd.Series)


def is_npArray(obj):
    return isinstance(obj, np.ndarray)


class PredictServiceClient:
    """Client to use with Predict. Have to be created in order to do predict"""

    def __init__(self, impl: PredictImplementation, target: str, signature: ModelSignature):
        self.impl = impl
        self.model_spec = ModelSpec(model_name=target)
        self.signature = signature

    def predict(self, inputs: Union[pd.DataFrame, dict]) -> Union[pd.DataFrame, dict]:
        """
        It forms a PredictRequest. PredictRequest specifies which TensorFlow model to run, as well as
        how inputs are mapped to tensors and how outputs are filtered before returning to user.
        + int, str, float, list, (later support of dict)
        when непонятное говно object, bytestring - raise ValueError()
        :param inputs: dict in the format of {string: Union[python_primitive_types, numpy_primitive_type]} with contract info
        :param model_spec: model specification created for associated servable
        :return: PredictResponse with outputs in protobuf format
        """
        # проверка inputs на соответсиве сигнатуре

        inputs_frmt, inputs_dtype = convert_inputs_to_tensor_proto(inputs, self.signature)

        request = predict_pb2.PredictRequest(model_spec=self.model_spec, inputs=inputs_frmt)

        try:
            response = self.impl.send_data(model_spec=self.model_spec, inputs=inputs)

            # TODO: add reverse conversion

            if is_dataframe(inputs_dtype):
                # convert
                pass

            if is_npArray(inputs_dtype):
                # convert
                pass
            if is_dict(inputs_dtype):
                # convert
                pass

            return response
            # проверка response на соответствие сигнатуре
        except Exception as err:
            raise Exception(f"Failed to predict.{str(err)} ")


class Predictable:
    """Adds Predictor functionality"""

    def predictor(self, shadowless=False, ssl=False) -> PredictServiceClient:
        if ssl:
            self.channel = self.cluster.grpc_secure()
        else:
            self.channel = self.cluster.grpc_insecure()
        if shadowless:
            self.impl = UnmetricableImplementation(self.channel)
        else:
            self.impl = MetricableImplementation(self.channel)

        return PredictServiceClient(self.impl, self.name, self.contract.predict)

# d = {'x': [10]}
# df = pd.DataFrame(data=d)
# result = PredictServiceClient.predict(None, df)  # returns df with additional columns sum
# assert isinstance(result, pd.DataFrame)
# assert result['sum'] == result['col1'] + result['col2']
# AbstractPredictService -> ShadowlessPredict, SecurePredict
# Factory -> PredictServiceClient(ssl+shadowless) -> PredictServiceClient.predict(x: DataFrame): DataFrame
# predict(x)
# x -> map(string, TensorProto)
# call stub
# result -> type(x)
# return result


# TODO: delete if not needed
# import abc
# import requests
# import hydro_serving_grpc as hsg
# import hydro_serving_grpc.gateway as hsgateway
#
#
# class AbstractPredictor(abc.ABC):
#     @abc.abstractmethod
#     def predict(self, data):
#         pass
#
#
# class JSONPredictor(AbstractPredictor):
#     def __init__(self, url, target_name, signature):
#         self.url = url
#
#     def predict(self, data):
#         return requests.post(url=self.url, json=data)
#
#
# class GRPCPredictor(AbstractPredictor):
#
#     def __init__(self, channel, target_name, signature):
#         self.channel = channel
#         self.target_name = target_name
#         self.contract = signature
#         self.stub = hsg.PredictionServiceStub(self.channel)
#
#     def predict(self, data):
#         return self.stub.Predict(data)
#
#
# class ShadowlessGRPCPredictor(GRPCPredictor):
#     def __init__(self, channel, target_name, signature):
#         super().__init__(channel, target_name, signature)
#         self.stub = hsgateway.GatewayServiceStub(self.channel)
#
#     def predict(self, data):
#         return self.stub.ShadowlessPredictServable(data)

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
