from abc import abstractmethod, ABC
from typing import Union, Dict

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2, PredictResponse, TensorProto
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
        Sends data to an application/servable and shadows it to monitoring services
        :param channel:
        :param target: Name of an application/servable which will receive data
        """
        self.stub = PredictionServiceStub(channel)
        self.send_params = self.get_monitoring_spec_params(target=target)

    def send_data(self, inputs: dict):
        model_spec = self.send_params
        request = predict_pb2.PredictRequest(model_spec=model_spec, inputs=inputs)
        return self.stub.Predict(request)

    def get_monitoring_spec_params(self, target: str) -> ModelSpec:
        return ModelSpec(name=target)


class UnmonitorableImplementation(PredictImplementation):
    """ This implementation sends data to a servable without shadowing it to monitoring services"""

    def __init__(self, channel, target: str):
        self.stub = GatewayServiceStub(channel)
        self.send_params = self.get_monitoring_spec_params(target=target)

    def send_data(self, inputs: Dict[str, TensorProto]):
        servable_name = self.send_params
        request = api_pb2.ServablePredictRequest(servable_name=servable_name, data=inputs)
        return self.stub.ShadowlessPredictServable(request)

    def get_monitoring_spec_params(self, target: str) -> str:
        return target


class PredictServiceClient:
    """PredictServiceClient is the main way of passing your data to a deployed ModelVersion"""

    def __init__(self, impl: PredictImplementation, signature: ModelSignature, return_type: PredictorDT):
        """
        Creates a client through which you could send your data.
        :param impl: implementation - either Monitorable (which shadows data to monitoring services), either not
        :param signature: ModelVersion signature which is used to encode/decode your data into proto messages
        :param return_type: One of 3 ways (Pandas, Numpy, Python) to represent a ModelVersion output
        """
        self.impl = impl
        self.signature = signature
        self.return_type = return_type

    def predict(self, inputs: Union[pd.DataFrame, dict, pd.Series]) -> Union[pd.DataFrame, Dict]:
        """
        Sends data to the model version deployed in the cluster and returns the response from it.

        This methods hides the conversion of Python/Numpy objects into a TensorProto objects.

        :param inputs: Input data in either Pandas DataFrame, Numpy or Python lists/scalars.
        :return: Output data in a format specified by `return_type`
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
            raise Exception(f"Failed to predict. {str(err)} ")

    @staticmethod
    def predict_resp_to_dict_pydtype(response: PredictResponse) -> Dict:
        """
        Transform PredictResponse into a Dictionary with Python lists/scalars
        :param response: PredictResponse proto message returned from the runtime
        :return: Dictionary with Python list/scalars
        """
        output_tensors_dict = {}
        for tensor_name, tensor_proto in response.outputs.items():
            output_tensors_dict[tensor_name] = tensor_proto_to_py(tensor_proto)
        return output_tensors_dict

    @staticmethod
    def predict_resp_to_dict_np(response: PredictResponse) -> Dict[str, np.array]:
        """
        Transform PredictResponse into a Dictionary with Numpy arrays/scalars
        :param response: PredictResponse proto message returned from the runtime
        :return: Dictionary with Numpy arrays
        """
        output_tensors_dict = dict()
        for tensor_name, tensor_proto in response.outputs.items():
            output_tensors_dict[tensor_name] = tensor_proto_to_np(tensor_proto)
        return output_tensors_dict

    @staticmethod
    def predict_resp_to_df(response: PredictResponse) -> pd.DataFrame:
        """
        Transform PredictResponse into a pandas.DataFrame by using intermediate representation of Dict[str, np.array]
        :param response: PredictResponse proto message returned from the runtime
        :return: pandas DataFrame
        """
        response_dict: Dict[str, np.array] = PredictServiceClient.predict_resp_to_dict_np(response)
        return pd.DataFrame(response_dict)
