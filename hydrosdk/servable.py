import warnings

import numpy as np
import pandas as pd
from hydro_serving_grpc.gateway import ServablePredictRequest

from hydrosdk.contract import ContractViolationException, Tensor


def decompose_arg_to_tensors(x):
    tensors = []
    if type(x) is dict:
        for k, v in x.items():
            tensors.append(Tensor(k, v))
    elif type(x) is pd.DataFrame:
        for k, v in dict(x).items():
            tensors.append(Tensor(k, v))
    elif type(x) is pd.Series:
        if x.name is None:
            raise ValueError("Provided pandas.Series should have names")
        else:
            tensors.append(Tensor(x.name, np.array(x)))
    elif type(x) is np.ndarray:
        raise NotImplementedError("Conversion of nameless np.array is not supported")
    else:
        raise ValueError("Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(type(x)))
    return tensors


def decompose_kwarg_to_tensor(key, x):
    if type(x) is dict:
        raise ValueError("Conversion of dict as kwatg is not supported/")
    elif type(x) is pd.DataFrame:
        tensor = Tensor(key, np.array(x))
    elif type(x) is pd.Series:
        tensor = Tensor(key, np.array(x))
    elif type(x) is np.ndarray:
        tensor = Tensor(key, x)
    elif np.isscalar(x):
        if x in (0, 1):
            # Minimum scalar dtype for 0 or 1 is `uint8`, but it
            # cannot be casted into `bool` safely. So, we detect
            # for bool scalars by hand.
            min_input_dtype = np.bool
        else:
            min_input_dtype = np.min_scalar_type(x)

        tensor = Tensor(key, np.array(x, dtype=min_input_dtype))
    else:
        raise ValueError("Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(type(x)))
    return tensor


class Servable:

    def __init__(self, model, servable_name, meta=None):
        if meta is None:
            meta = dict()
        self.model = model
        self.name = servable_name
        self.meta = meta

    def __call__(self, profile=True, *args, **kwargs):
        input_tensors = []

        for arg in args:
            input_tensors.extend(decompose_arg_to_tensors(arg))

        for key, arg in kwargs.items():
            input_tensors.append(decompose_kwarg_to_tensor(key, arg))

        is_valid, error_msg = self.model.contract.signature.validate_input(input_tensors)
        if not is_valid:
            return ContractViolationException(error_msg)

        input_proto_dict = dict((map(lambda x: (x.name, x.proto), input_tensors)))
        predict_request = ServablePredictRequest(servable_name=self.name, data=input_proto_dict)

        if profile:
            result = self.model.cluster.gateway_stub.PredictServable(predict_request)
        else:
            result = self.model.cluster.gateway_stub.ShadowlessPredictServable(predict_request)

        output_tensors = []
        for tensor_name, tensor_proto in result.outputs:
            output_tensors.append(Tensor.from_proto(tensor_name, tensor_proto))

        is_valid, error_msg = self.model.contract.signature.validate_output(output_tensors)
        if not is_valid:
            warnings.warn("Output is not valid.\n" + error_msg)

        return output_tensors

    def remove(self, ):
        """
        Kill this servable
        :return:
        """
        return self.model.cluster.remove_servable(self.name)
