from typing import Union, Dict, List

import numpy as np
import pandas as pd
from hydro_serving_grpc import TensorProto, DataType, TensorShapeProto
from hydro_serving_grpc.contract import ModelSignature

from hydrosdk.data.types import NP_TO_HS_DTYPE, DTYPE_TO_FIELDNAME, np2proto_shape, PY_TO_DTYPE, find_in_list_by_name, proto2np_dtype


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


def tensor_proto_to_nparray(t: TensorProto):
    """
    Creates Numpy array given dtype, shape and values from TensorProto object
    :param t:
    :return:
    """
    array_shape = [dim.size for dim in t.tensor_shape.dim]
    np_dtype = proto2np_dtype(t.dtype)
    value = getattr(t, DTYPE_TO_FIELDNAME[t.dtype])

    nparray = np.array(value, dtype=np_dtype)

    # If no dims specified in TensorShapeProto, then it is scalar
    if array_shape:
        return nparray.reshape(*array_shape)
    else:
        return np.asscalar(nparray)


def nparray_to_tensor_proto(x: np.array):
    """
    Creates TensorProto object with specified dtype, shape and values under respective fieldname from np.array
    :param x:
    :return:
    """
    proto_dtype = NP_TO_HS_DTYPE.get(x.dtype.type)
    if proto_dtype is None:
        raise ValueError(f"Couldn't convert numpy dtype {x.dtype.type} to one of available TensorProto dtypes")

    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: x.flatten(),
        "dtype": proto_dtype,
        "tensor_shape": np2proto_shape(x.shape)
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

            if type(value) in PY_TO_DTYPE:
                # If we got a single val, we can perform the same logic in the next steps if we create List[value] from it
                value = [value]

            if isinstance(value, list):  # x: [1,2,3,4]
                signature_field = find_in_list_by_name(some_list=signature.inputs, name=key)
                tensors[key] = list_to_tensor_proto(value, signature_field.dtype, signature_field.shape)

            elif isinstance(value, np.ndarray) or isinstance(value, np.ScalarType):
                # Support both np.ndarray and np.scalar since they support same operations on them
                tensors[key] = nparray_to_tensor_proto(value)
            else:
                raise TypeError("Unsupported objects in dict values {}".format(type(value)))

    elif isinstance(inputs, pd.DataFrame):
        for key, value in dict(inputs).items():
            tensors[key] = nparray_to_tensor_proto(value.ravel())
    else:
        raise ValueError(
            "Conversion failed. Expected [pandas.DataFrame, dict[str, numpy.ndarray], dict[str, list], dict[str, python_primitive]], got {}".format(
                type(inputs)))

    return tensors
