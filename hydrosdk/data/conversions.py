from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
from hydro_serving_grpc import TensorProto, DataType, TensorShapeProto, DT_STRING, DT_HALF, DT_COMPLEX64, DT_COMPLEX128
from hydro_serving_grpc.contract import ModelSignature
from pandas.core.common import flatten

from hydrosdk.data.types import np_to_proto_dtype, DTYPE_TO_FIELDNAME, find_in_list_by_name, proto_to_np_dtype


def tensor_proto_to_py(t: TensorProto):
    """
    Converts tensor proto into a corresponding python object - list or scalar
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


def list_to_tensor_proto(data: List, proto_dtype: DataType, proto_shape: TensorShapeProto) -> TensorProto:
    """
    Converts data in a form of a Python List into a TensorProto object
    :param data: List with data
    :param proto_dtype: DataType of a future TensorProto
    :param proto_shape: TensorShapeProto of a future TensorProto
    :return: Same data but in a TensorProto object
    """
    # We can pack only flattened lists into TensorProto, so we need to flatten the list
    flattened_list = flatten(data)
    tensor_proto_parameters = {
        DTYPE_TO_FIELDNAME[proto_dtype]: flattened_list,
        "dtype": proto_dtype,
        "tensor_shape": proto_shape
    }
    return TensorProto(**tensor_proto_parameters)


def tensor_proto_to_np(t: TensorProto):
    """
    Creates either np.array or scalar with Numpy dtype based on
     data type, shape and values from TensorProto object
    :param t:
    :return:
    """
    array_shape = [dim.size for dim in t.tensor_shape.dim]
    np_dtype = proto_to_np_dtype(t.dtype)
    proto_values = getattr(t, DTYPE_TO_FIELDNAME[t.dtype])

    if t.dtype == DT_HALF:
        x = np.fromiter(proto_values, dtype=np.uint16).view(np.float16)
    elif t.dtype == DT_STRING:
        x = np.array([s.decode("utf-8") for s in proto_values])
    elif t.dtype == DT_COMPLEX64 or t.dtype == DT_COMPLEX128:
        it = iter(proto_values)
        x = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=np_dtype)
    else:
        x = np.array(proto_values, dtype=np_dtype)

    # If no dims specified in TensorShapeProto, then it is scalar
    if array_shape:
        return x.reshape(*array_shape)
    else:
        return x.flatten()[0]


def np_to_tensor_proto(x) -> TensorProto:
    """
    Creates TensorProto object from Numpy ndarray or scalar with inferred TensorProtoShape and DataType
    :param x: Union[np.array, np.ScalarType]
    :return:
    """
    if isinstance(x, np.ScalarType):
        return scalar_to_tensor_proto(x)
    elif isinstance(x, np.ndarray):
        return nparray_to_tensor_proto(x)
    else:
        raise TypeError(f"Unsupported object {x}")


def nparray_to_tensor_proto(x: np.array) -> TensorProto:
    """
    Creates TensorProto object from Numpy ndarray
    with TensorProtoShape and DataType inferred from the latter
    :param x: Data in form ofa numpy ndarray
    :return: Same data packed into a TensorProto object
    """

    if x.dtype.isbuiltin != 1 and x.dtype.type != np.str_:
        raise ValueError(f"{x.dtype} is not supported."
                         f" Dtypes not compiled into numpy are not supported, except for np.str.")

    proto_dtype = np_to_proto_dtype(x.dtype.type)
    proto_shape = tensor_shape_proto_from_tuple(x.shape)

    if proto_dtype == DT_HALF:
        proto_values = x.view(np.uint16).flatten()
    elif proto_dtype == DT_STRING:
        proto_values = [s.encode("utf-8") for s in x.flatten()]
    elif proto_dtype == DT_COMPLEX64 or proto_dtype == DT_COMPLEX128:
        proto_values = [v.item() for c_number in x.flatten() for v in [c_number.real, c_number.imag]]
    else:
        proto_values = x.flatten()

    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: proto_values,
        "dtype": proto_dtype,
        "tensor_shape": proto_shape
    }

    return TensorProto(**kwargs)


def scalar_to_tensor_proto(x: np.ScalarType) -> TensorProto:
    """
    Creates TensorProto object from a scalar with a Numpy dtype
      with TensorProtoShape and DataType inferred from the latter
    :param x: Scalar value with a Numpy dtype
    :return: Same value but packed into a TensorProto object
    """
    proto_dtype = np_to_proto_dtype(type(x))

    if proto_dtype == DT_HALF:
        proto_values = [np.array(x, dtype=np.float16).view(np.uint16)]
    elif proto_dtype == DT_STRING:
        proto_values = [x.encode("utf-8")]
    elif proto_dtype == DT_COMPLEX64 or proto_dtype == DT_COMPLEX128:
        proto_values = [x.real, x.imag]
    else:
        proto_values = [x]

    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: proto_values,
        "dtype": proto_dtype,
        "tensor_shape": TensorShapeProto()
    }
    return TensorProto(**kwargs)


def tensor_shape_proto_from_tuple(shape: Iterable[int]) -> TensorShapeProto:
    """
    Helper function to transform shape in the form of a tuple (Numpy shape representation) into a TensorProtoShape
    :param shape: Shape in a tuple form
    :return: same shape but in a TensorShapeProto object
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=s) for s in shape])


def convert_inputs_to_tensor_proto(inputs: Dict, signature: ModelSignature) -> Dict[str, TensorProto]:
    """
    Generate Dict[str, TensorProto] from pd.DataFrame or Dict[str, Union[np.array, np.ScalarType]]

    Converts inputs into a representation of data where each field
     of a signature is represented by a valid TensorProto object.
    :param inputs: Dict, where keys are names of signature fields and
     values are data in either Numpy or Python form, or alternatively,
     pd.DataFrame, where columns are names of fields and column values are data.
    :param signature: ModelVersion signature with names, shapes and dtypes
     of fields into which `inputs` are converted into
    :return: Dictionary with TensorProtos to be used in forming a PredictRequest
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
