from abc import abstractmethod, ABC
from typing import Union, Dict

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2, PredictResponse
from hydro_serving_grpc.contract import ModelSignature
from hydro_serving_grpc.gateway import GatewayServiceStub

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto
from hydrosdk.data.types import PredictorDT, proto2np_dtype, DTYPE_TO_FIELDNAME


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


class PredictServiceClient:
    """Client to use with Predict. Have to be created in order to do predict"""

    def __init__(self, impl: PredictImplementation, target: str, signature: ModelSignature, return_type: PredictorDT):
        self.impl = impl
        self.model_spec = ModelSpec(name=target)
        self.signature = signature
        self.return_type = return_type

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
        inputs_as_proto = convert_inputs_to_tensor_proto(inputs, self.signature)

        try:
            response = self.impl.send_data(model_spec=self.model_spec, inputs=inputs_as_proto)

            if self.return_type == PredictorDT.DF:
                return self.predict_resp_to_df(response=response)

            elif self.return_type == PredictorDT.DICT_NP_ARRAY:
                return self.predict_resp_to_dict_nparray(response=response)

            elif self.return_type == PredictorDT.DICT_PYTHON:
                return self.predict_resp_to_dict_pydtype(response=response)
        except Exception as err:
            raise Exception(f"Failed to predict.{str(err)} ")

    @staticmethod
    def predict_resp_to_dict_pydtype(response: PredictResponse) -> dict:
        output_tensors_dict = {}
        for tensor_name, tensor_proto in response.outputs.items():
            dims = [dim.size for dim in tensor_proto.tensor_shape.dim]
            value = getattr(tensor_proto, DTYPE_TO_FIELDNAME[tensor_proto.dtype])
            value = np.reshape(value, dims).tolist()
            output_tensors_dict[tensor_name] = value
        return output_tensors_dict

    @staticmethod
    def predict_resp_to_dict_nparray(response: PredictResponse) -> dict:
        output_tensors_dict = {}
        for tensor_name, tensor_proto in response.outputs.items():
            array_shape = [dim.size for dim in tensor_proto.tensor_shape.dim]
            np_dtype = proto2np_dtype(tensor_proto.dtype)
            value = getattr(tensor_proto, DTYPE_TO_FIELDNAME[tensor_proto.dtype])
            output_tensors_dict[tensor_name] = np.asarray(value, dtype=np_dtype).reshape(*array_shape if array_shape else (-1,))

        return output_tensors_dict

    @staticmethod
    def predict_resp_to_df(response: PredictResponse) -> pd.DataFrame:
        response_dict: Dict[str, np.array] = PredictServiceClient.predict_resp_to_dict_nparray(response)
        return pd.DataFrame(response_dict)


class Predictable(ABC):
    @abstractmethod
    def predictor(self):
        pass


class PredictableApplication(Predictable):
    """Adds Predictor functionality"""

    def predictor(self, ssl=False, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """

        :param ssl:
        :param return_type:
        :return:
        """
        if ssl:
            self.channel = self.grpc_cluster.grpc_secure()
        else:
            self.channel = self.grpc_cluster.grpc_insecure()

        self.impl = MetricableImplementation(self.channel)
        self.predictor_return_type = return_type

        return PredictServiceClient(self.impl, self.name, self.model.contract.predict,
                                    return_type=self.predictor_return_type)


class PredictableServable(Predictable):
    """Adds Predictor functionality"""

    def predictor(self, shadowless=False, ssl=False, return_type=PredictorDT.DICT_NP_ARRAY) -> PredictServiceClient:
        """

        :param shadowless:
        :param ssl:
        :param return_type:
        :return:
        """
        if ssl:
            self.channel = self.grpc_cluster.grpc_secure()
        else:
            self.channel = self.grpc_cluster.grpc_insecure()
        if shadowless:
            self.impl = UnmetricableImplementation(self.channel)
        else:
            self.impl = MetricableImplementation(self.channel)
        self.predictor_return_type = return_type

        return PredictServiceClient(self.impl, self.name, self.model.contract.predict,
                                    return_type=self.predictor_return_type)
