from abc import abstractmethod, ABC
from typing import Union, Dict

import numpy as np
import pandas as pd
from hydro_serving_grpc.serving.runtime.api_pb2 import PredictRequest, PredictResponse
from hydro_serving_grpc.serving.contract.tensor_pb2 import Tensor
from hydro_serving_grpc.serving.gateway.api_pb2 import GatewayPredictRequest
from hydro_serving_grpc.serving.gateway.api_pb2_grpc import GatewayServiceStub
from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto, tensor_proto_to_np, tensor_proto_to_py
from hydrosdk.data.types import PredictorDT


class PredictImplementation(ABC):
    @abstractmethod
    def send_data(self, inputs: dict):
        pass


class MonitorableApplicationPredictionService(PredictImplementation):
    """Sends data to an application and shadows it to the monitoring services."""
    def __init__(self, channel, target: str):
        """
        :param channel:
        :param target: Name of an application, which will receive data.
        """
        self.stub = GatewayServiceStub(channel)
        self.target = target

    def send_data(self, inputs: Dict[str, Tensor]) -> PredictResponse:
        request = GatewayPredictRequest(name=self.target, data=inputs)
        return self.stub.PredictApplication(request)


class MonitorableServablePredictionService(PredictImplementation):
    """Sends data to a servable and shadows it to the monitoring services."""
    def __init__(self, channel, target: str):
        """
        :param channel:
        :param target: Name of a servable, which will receive the data.
        """
        self.stub = GatewayServiceStub(channel)
        self.target = target

    def send_data(self, inputs: Dict[str, Tensor]) -> PredictResponse:
        request = GatewayPredictRequest(name=self.target, data=inputs)
        return self.stub.PredictServable(request)


class UnmonitorableServablePredictionService(PredictImplementation):
    """Sends data to a servable without shadowing it to the monitoring services."""
    def __init__(self, channel, target: str):
        """
        :param channel:
        :param target: Name of a servable, which will receive the data.
        """
        self.stub = GatewayServiceStub(channel)
        self.target = target

    def send_data(self, inputs: Dict[str, Tensor]) -> PredictResponse:
        request = GatewayPredictRequest(name=self.target, data=inputs)
        return self.stub.ShadowlessPredictServable(request)


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
