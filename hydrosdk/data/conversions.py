import numpy as np
import pandas as pd
from hydro_serving_grpc import TensorProto, DataType, TensorShapeProto

from hydrosdk.data.types import NP_TO_HS_DTYPE, DTYPE_TO_FIELDNAME, np2proto_shape, PY_TO_DTYPE, PredictorDT, \
    find_in_list_by_name


def numpy_data_to_tensor_proto(data, dtype, shape):
    proto_dtype = NP_TO_HS_DTYPE[dtype.type]
    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: data.flatten(),
        "dtype": proto_dtype,
        "tensor_shape": np2proto_shape(shape)
    }
    return TensorProto(**kwargs)


def dtype_to_tensor_proto(data: int, dtype: str, shape: TensorShapeProto):
    proto_dtype = DataType.Value(DataType.Name(dtype))
    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: [data],
        "dtype": proto_dtype,
        "tensor_shape": shape
    }
    return TensorProto(**kwargs)


def python_dtype_to_proto(value, key, signature) -> TensorProto:
    return numpy_data_to_tensor_proto(value, signature.inputs[key].dtype, signature.inputs[key].shape)


def convert_inputs_to_tensor_proto(inputs, signature) -> dict:
    """

    :param inputs:
    :param signature:
    :return:
    """
    tensors = {}
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            if type(value) in PY_TO_DTYPE:
                # if we got a single val, to have the same logic in the next step(creating prot) we do this
                value = [value]

            if isinstance(value, list):  # x: [1,2,3,4]
                for list_el in value:
                    current_signature = find_in_list_by_name(some_list=signature.inputs, name=key)
                    tensors[key] = dtype_to_tensor_proto(list_el, current_signature.dtype,
                                                         current_signature.shape)
            elif isinstance(value, np.ndarray) or isinstance(value, np.ScalarType):
                # Support both np.ndarray and np.scalar since they support same operations on them
                tensors[key] = numpy_data_to_tensor_proto(value, value.dtype, value.shape)
            else:
                raise TypeError("Unsupported objects in dict values {}".format(type(value)))
    elif isinstance(inputs, pd.DataFrame):
        for key, value in dict(inputs).items():
            tensors[key] = numpy_data_to_tensor_proto(value.ravel(), value.dtype, value.shape)
    else:
        raise ValueError(
            "Conversion failed. Expected [pandas.DataFrame, dict[str, numpy.ndarray], dict[str, list], dict[str, python_primitive]], got {}".format(
                type(inputs)))

    return tensors
