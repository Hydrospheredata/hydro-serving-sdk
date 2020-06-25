from abc import abstractmethod, ABC
from typing import Union, Dict

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2, PredictResponse
from hydro_serving_grpc.contract import ModelSignature
from hydro_serving_grpc.gateway import GatewayServiceStub, api_pb2

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto, tensor_proto_to_np, tensor_proto_to_py
from hydrosdk.data.types import PredictorDT


class PredictImplementation(ABC):
    @abstractmethod
    def send_data(self, inputs: dict):
        pass

    @abstractmethod
    def get_monitoring_spec_params(self, target: str):
        pass


class MonitorableImplementation(PredictImplementation):
    def __init__(self, channel, target: str):
        """

        :param channel:
        :param target: name of application/servable it's been created to
        """
        self.stub = PredictionServiceStub(channel)
        self.send_params = self.get_monitoring_spec_params(target=target)

    def send_data(self, inputs: dict):
        """

        :param inputs:
        :return:
        """
        model_spec = self.send_params

        request = predict_pb2.PredictRequest(model_spec=model_spec, inputs=inputs)
        return self.stub.Predict(request)

    def get_monitoring_spec_params(self, target: str) -> ModelSpec:
        return ModelSpec(name=target)


class UnmonitorableImplementation(PredictImplementation):
    def __init__(self, channel, target: str):
        """

        :param channel:
        :param target: servable name
        """
        self.stub = GatewayServiceStub(channel)
        self.send_params = self.get_monitoring_spec_params(target=target)

    def send_data(self, inputs: dict):
        """

        :param inputs:
        :return:
        """
        servable_name = self.send_params

        request = api_pb2.ServablePredictRequest(servable_name=servable_name, data=inputs)
        return self.stub.ShadowlessPredictServable(request)

    def get_monitoring_spec_params(self, target: str) -> str:
        return target


class PredictServiceClient:
    """Client to use with Predict. Have to be created in order to do predict"""

    def __init__(self, impl: PredictImplementation, signature: ModelSignature, return_type: PredictorDT):
        self.impl = impl

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
        inputs_as_proto = convert_inputs_to_tensor_proto(inputs, self.signature)

        try:
            response = self.impl.send_data(inputs=inputs_as_proto)
            if self.return_type == PredictorDT.DF:
                return self.predict_resp_to_df(response=response)

            elif self.return_type == PredictorDT.DICT_NP_ARRAY:
                return self.predict_resp_to_dict_np(response=response)

            elif self.return_type == PredictorDT.DICT_PYTHON:
                return self.predict_resp_to_dict_pydtype(response=response)
        except Exception as err:
            raise Exception(f"Failed to predict.{str(err)} ")

    @staticmethod
    def predict_resp_to_dict_pydtype(response: PredictResponse) -> Dict:
        output_tensors_dict = {}
        for tensor_name, tensor_proto in response.outputs.items():
            output_tensors_dict[tensor_name] = tensor_proto_to_py(tensor_proto)
        return output_tensors_dict

    @staticmethod
    def predict_resp_to_dict_np(response: PredictResponse) -> Dict[str, np.array]:
        """
        Transform tensors insider PredictResponse into np.arrays to create Dict[str, np.array]
        :param response:
        :return:
        """
        output_tensors_dict = dict()
        for tensor_name, tensor_proto in response.outputs.items():
            output_tensors_dict[tensor_name] = tensor_proto_to_np(tensor_proto)
        return output_tensors_dict

    @staticmethod
    def predict_resp_to_df(response: PredictResponse) -> pd.DataFrame:
        """
        Transform PredictResponse into pandas.DataFrame by using intermediate representation of Ditt[str, np.array]
        :param response:
        :return:
        """
        response_dict: Dict[str, np.array] = PredictServiceClient.predict_resp_to_dict_np(response)
        return pd.DataFrame(response_dict)
