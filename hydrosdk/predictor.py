from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2
from hydro_serving_grpc.contract import ModelSignature
from hydro_serving_grpc.gateway import GatewayServiceStub

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto
from hydrosdk.data.types import PredictorDT, proto2np_dtype


class PredictImplementation(ABC):
    @abstractmethod
    def send_data(self, model_spec, inputs):
        pass

    def data_to_predictRequest(self, model_spec, inputs) -> predict_pb2.PredictRequest:
        return predict_pb2.PredictRequest(model_spec=model_spec, inputs=inputs)


class MetricableImplementation(PredictImplementation):
    def __init__(self, channel):
        self.stub = PredictionServiceStub(channel)

    def send_data(self, model_spec, inputs):
        request = self.data_to_predictRequest(model_spec=model_spec, inputs=inputs)
        return self.stub.Predict(request)


class UnmetricableImplementation(PredictImplementation):
    def __init__(self, channel):
        self.stub = GatewayServiceStub(channel)

    def send_data(self, model_spec, inputs):
        request = self.data_to_predictRequest(model_spec=model_spec, inputs=inputs)
        return self.stub.ShadowlessPredictServable(request)


def is_dataframe(obj):
    return obj == PredictorDT.PD_DF


def is_dict(obj):
    return obj == PredictorDT.DICT

# TODO: do we need series?
# def is_pdSeries(obj):
#     return obj == PredictorDT.PD_SERIES


def is_npArray(obj):
    return obj == PredictorDT.NP_ARRAY


class PredictServiceClient:
    """Client to use with Predict. Have to be created in order to do predict"""

    def __init__(self, impl: PredictImplementation, target: str, signature: ModelSignature):
        self.impl = impl
        self.model_spec = ModelSpec(name=target)
        self.signature = signature

    # TODO: fix doc
    def predict(self, inputs: Union[pd.DataFrame, dict, pd.Series]) -> Union[pd.DataFrame, dict, pd.Series]:
        """
        It forms a PredictRequest. PredictRequest specifies which TensorFlow model to run, as well as
        how inputs are mapped to tensors and how outputs are filtered before returning to user.
        :param inputs: dict in the format of {string: Union[python_primitive_types, numpy_primitive_type]} with contract info
        :param model_spec: model specification created for associated servable
        :return: PredictResponse with outputs in protobuf format
        """
        # TODO: add проверка inputs на соответсиве сигнатуре

        inputs_as_proto, inputs_dtype = convert_inputs_to_tensor_proto(inputs, self.signature)

        try:
            response = self.impl.send_data(model_spec=self.model_spec, inputs=inputs_as_proto)

            if is_dataframe(inputs_dtype):
                df = pd.DataFrame()
                for key, value in response.outputs.items():
                    current_index = key
                    tensor_shape = value.tensor_shape

                    dims = pd.Series([dim.size for dim in tensor_shape.dim])
                    df[current_index] = dims.values
                return df

            elif is_npArray(inputs_dtype):
                output_tensors_dict = {}
                for key, value in response.outputs.items():
                    key_str = key

                    tensor_shape = value.tensor_shape

                    dims = [dim.size for dim in tensor_shape.dim]

                    dtype = proto2np_dtype(value.dtype)
                    output_tensors_dict[key_str] = np.asarray(dims, dtype=dtype)

                return output_tensors_dict

            elif is_dict(inputs_dtype):
                output_tensors_dict = {}
                for key, value in response.outputs.items():
                    key_str = key

                    tensor_shape = value.tensor_shape

                    dims = [dim.size for dim in tensor_shape.dim]

                    output_tensors_dict[key_str] = dims
                return output_tensors_dict
            # TODO: do we need series?
            # elif is_pdSeries(inputs_dtype):
            #     indices = []
            #     data = []
            #     for key, value in response.outputs.items():
            #         current_index = key
            #         tensor_shape = value.tensor_shape
            #
            #         indices.append(current_index)
            #         dims = [dim.size for dim in tensor_shape.dim]
            #         data.append(dims)
            #
            #     series = pd.Series(data, index=indices)
            #     return series

            else:
                raise ValueError(
                    "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
                        inputs_dtype))

            # проверка response на соответствие сигнатуре
        except Exception as err:
            raise Exception(f"Failed to predict.{str(err)} ")


class Predictable:
    """Adds Predictor functionality"""

    def predictor(self, shadowless=False, ssl=False) -> PredictServiceClient:
        if ssl:
            self.channel = self.grpc_cluster.grpc_secure()
        else:
            self.channel = self.grpc_cluster.grpc_insecure()
        if shadowless:
            self.impl = UnmetricableImplementation(self.channel)
        else:
            self.impl = MetricableImplementation(self.channel)

        return PredictServiceClient(self.impl, self.name, self.model.contract.predict)

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
