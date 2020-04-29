from abc import abstractmethod, ABC
from typing import Union, Dict

import numpy as np
import pandas as pd
from hydro_serving_grpc import PredictionServiceStub, ModelSpec, predict_pb2, PredictResponse
from hydro_serving_grpc.contract import ModelSignature
from hydro_serving_grpc.gateway import GatewayServiceStub, api_pb2

from hydrosdk.data.conversions import convert_inputs_to_tensor_proto
from hydrosdk.data.types import PredictorDT, proto2np_dtype, DTYPE_TO_FIELDNAME


class PredictImplementation(ABC):
    @abstractmethod
    def send_data(self, *args, **kwargs):
        pass


class MonitorableImplementation(PredictImplementation):
    def __init__(self, channel):
        self.stub = PredictionServiceStub(channel)

    def send_data(self, *args, **kwargs):
        """

        :param args:
        :param kwargs: expects ModelSpec and list of inputs
        :return:
        """
        model_spec = kwargs["send_params"]["model_spec"]
        inputs = kwargs["inputs"]

        request = predict_pb2.PredictRequest(model_spec=model_spec, inputs=inputs)
        return self.stub.Predict(request)


class UnmonitorableImplementation(PredictImplementation):
    def __init__(self, channel, servable_name: str):
        """

        :param channel:
        :param servable_name:
        """
        self.stub = GatewayServiceStub(channel)
        self.servable_name = servable_name

    def send_data(self, *args, **kwargs):
        """

        :param args:
        :param kwargs: expects servable_name and list of inputs
        :return:
        """
        servable_name = kwargs["send_params"]["servable_name"]
        inputs = kwargs["inputs"]

        request = api_pb2.ServablePredictRequest(servable_name=servable_name, data=data)
        return self.stub.ShadowlessPredictServable(request)


class PredictServiceClient:
    """Client to use with Predict. Have to be created in order to do predict"""

    def __init__(self, impl: PredictImplementation, target: str, signature: ModelSignature, return_type: PredictorDT):
        self.impl = impl

        if isinstance(impl, MonitorableImplementation):
            self.send_params = {"model_spec": ModelSpec(name=target)}
        elif isinstance(impl, UnmonitorableImplementation):
            self.send_params = {"servable_name": target}
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
            response = self.impl.send_data(send_params=self.send_params, inputs=inputs_as_proto)
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

            # If no dims specified in TensorShapeProto, then it is scalar
            if dims:
                value = np.reshape(value, dims).tolist()
                output_tensors_dict[tensor_name] = value
            else:
                output_tensors_dict[tensor_name] = value[0]

        return output_tensors_dict

    @staticmethod
    def predict_resp_to_dict_nparray(response: PredictResponse) -> dict:
        output_tensors_dict = {}
        for tensor_name, tensor_proto in response.outputs.items():
            array_shape = [dim.size for dim in tensor_proto.tensor_shape.dim]
            np_dtype = proto2np_dtype(tensor_proto.dtype)
            value = getattr(tensor_proto, DTYPE_TO_FIELDNAME[tensor_proto.dtype])
            np_array_value = np.array(value, dtype=np_dtype)

            # If no dims specified in TensorShapeProto, then it is scalar
            if array_shape:
                output_tensors_dict[tensor_name] = np_array_value.reshape(*array_shape)
            else:
                output_tensors_dict[tensor_name] = np.asscalar(np_array_value)

        return output_tensors_dict

    @staticmethod
    def predict_resp_to_df(response: PredictResponse) -> pd.DataFrame:
        response_dict: Dict[str, np.array] = PredictServiceClient.predict_resp_to_dict_nparray(response)
        return pd.DataFrame(response_dict)
