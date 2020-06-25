from typing import Union, Dict, List

import numpy as np
import pandas as pd
from hydro_serving_grpc import TensorProto, DataType, TensorShapeProto
from hydro_serving_grpc.contract import ModelSignature

from hydrosdk.data.types import to_proto_dtype, DTYPE_TO_FIELDNAME, tensor_shape_proto_from_tuple, find_in_list_by_name, from_proto_dtype


def tensor_proto_to_py(t: TensorProto):
    """
    Converts tensor proto into corresponding python object
    :param t:
    :return:
    """
    dims = [dim.size for dim in t.tensor_shape.dim]
    value = getattr(t, DTYPE_TO_FIELDNAME[t.dtype])

    # If no dims specified in TensorShapeProto, then it is scalar
    if dims:
        value = np.reshape(value, dims).tolist()
        return value
    else:
        return value[0]


def tensor_proto_to_np(t: TensorProto):  # -> Union[np.array, np.ScalarType]:
    """
    Creates either np.array or scalar with Numpy dtype based on
     data type, shape and values from TensorProto object
    :param t:
    :return:
    """
    array_shape = [dim.size for dim in t.tensor_shape.dim]
    np_dtype = from_proto_dtype(t.dtype)
    value = getattr(t, DTYPE_TO_FIELDNAME[t.dtype])

    if np_dtype == np.float16:
        nparray = np.fromiter(value, dtype=np.uint16).view(np.float16)
    else:
        nparray = np.array(value, dtype=np_dtype)
    # If no dims specified in TensorShapeProto, then it is scalar
    if array_shape:
        return nparray.reshape(*array_shape)
    else:
        return nparray.flatten()[0]


# : Union[np.array, np.ScalarType]
def np_to_tensor_proto(x) -> TensorProto:
    if isinstance(x, np.ScalarType):
        return scalar_to_tensor_proto(x)
    elif isinstance(x, np.ndarray):
        return nparray_to_tensor_proto(x)
    else:
        raise TypeError(f"Unsupported object {x}")


def nparray_to_tensor_proto(x: np.array) -> TensorProto:
    """
    Creates TensorProto object with specified dtype, shape and values under respective fieldname from np.array
    :param x:
    :return:
    """
    proto_dtype = to_proto_dtype(x.dtype.type)

    if x.dtype == np.float16:
        x = x.view(np.uint16)

    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: x.flatten(),
        "dtype": proto_dtype,
        "tensor_shape": tensor_shape_proto_from_tuple(x.shape)
    }

    return TensorProto(**kwargs)


def scalar_to_tensor_proto(x: np.ScalarType) -> TensorProto:
    proto_dtype = to_proto_dtype(type(x))

    if type(x) == np.float16:
        x = np.array(x, dtype=np.float16).view(np.uint16).item()

    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: [x],
        "dtype": proto_dtype,
        "tensor_shape": TensorShapeProto()
    }
    return TensorProto(**kwargs)


def list_to_tensor_proto(data: List, dtype: str, shape: TensorShapeProto):
    proto_dtype = DataType.Value(DataType.Name(dtype))
    tensor_proto_parameters = {
        DTYPE_TO_FIELDNAME[proto_dtype]: data,
        "dtype": proto_dtype,
        "tensor_shape": shape
    }
    return TensorProto(**tensor_proto_parameters)


def convert_inputs_to_tensor_proto(inputs: Union[Dict, pd.DataFrame], signature: ModelSignature) -> dict:
    """

    :param inputs:
    :param signature:
    :return:
    """
    tensors = {}
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            if isinstance(value, list):  # x: [1,2,3,4]
                signature_field = find_in_list_by_name(some_list=signature.inputs, name=key)
                tensors[key] = list_to_tensor_proto(value, signature_field.dtype, signature_field.shape)
            elif isinstance(value, np.ScalarType):
                # This works for all scalars, including python int, float, etc.
                tensors[key] = scalar_to_tensor_proto(value)
            elif isinstance(value, np.ndarray):
                tensors[key] = nparray_to_tensor_proto(value)
            else:
                raise TypeError("Unsupported objects in dict values {}".format(type(value)))

    elif isinstance(inputs, pd.DataFrame):
        for key, value in dict(inputs).items():
            tensors[key] = nparray_to_tensor_proto(value.ravel())
    else:
        raise ValueError(f"Conversion failed. Expected [pandas.DataFrame, dict[str, numpy.ndarray],\
                           dict[str, list], dict[str, np.ScalarType]], got {type(inputs)}")

    return tensors
