import numbers
from enum import Enum

import numpy as np
from hydro_serving_grpc.serving.contract.types_pb2 import (
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
    DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX64, DT_COMPLEX128, DT_BOOL, DT_STRING, 
    DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, DT_QUINT16, DT_INVALID, DataType
)
from hydro_serving_grpc.serving.contract.tensor_pb2 import TensorShape

DTYPE_TO_FIELDNAME = {
    DT_HALF: "half_val",
    DT_FLOAT: "float_val",
    DT_DOUBLE: "double_val",

    DT_INT8: "int_val",
    DT_INT16: "int_val",
    DT_INT32: "int_val",
    DT_INT64: "int64_val",
    DT_UINT8: "int_val",
    DT_UINT16: "int_val",
    DT_UINT32: "uint32_val",
    DT_UINT64: "uint64_val",
    DT_COMPLEX64: "scomplex_val",
    DT_COMPLEX128: "dcomplex_val",
    DT_BOOL: "bool_val",
    DT_STRING: "string_val",
}

PY_TO_DTYPE = {
    int: DT_INT64,
    str: DT_STRING,
    bool: DT_BOOL,
    float: DT_DOUBLE,
    complex: DT_COMPLEX128
}

DTYPE_TO_PY = {v: k for k, v in PY_TO_DTYPE.items()}

ALIAS_TO_DTYPE = {
    "string": DT_STRING,
    "str": DT_STRING,
    "bool": DT_BOOL,

    "float16": DT_HALF,
    "half": DT_HALF,
    "float32": DT_FLOAT,
    "single": DT_FLOAT,
    "float": DT_DOUBLE,  # We treat float the same way Numpy does, look at np.float_
    "float64": DT_DOUBLE,
    "double": DT_DOUBLE,

    "int8": DT_INT8,
    "byte": DT_INT8,
    "int16": DT_INT16,
    "short": DT_INT16,
    "int32": DT_INT32,
    "int64": DT_INT64,
    "int": DT_INT64,

    "uint8": DT_UINT8,
    "ubyte": DT_UINT8,
    "uint16": DT_UINT16,
    "ushort": DT_UINT16,
    "uint32": DT_UINT32,
    "uint64": DT_UINT64,

    "qint8": DT_QINT8,
    "qint16": DT_QINT16,
    "qint32": DT_QINT32,

    "quint8": DT_QUINT8,
    "quint16": DT_QUINT16,

    "complex64": DT_COMPLEX64,
    "complex128": DT_COMPLEX128,
    "complex": DT_COMPLEX128
}

scalar = "scalar"


def alias_to_proto_dtype(name):
    type_ = ALIAS_TO_DTYPE.get(name, DT_INVALID)
    if not type_:
        try:
            type_ = DataType.Value(name)
        except ValueError:
            type_ = DT_INVALID

    return type_


def shape_to_proto(user_shape):
    if isinstance(user_shape, dict):
        user_shape = user_shape.get("dims")

    if user_shape == "scalar":
        shape = TensorShape()
    elif user_shape is None:
        shape = None
    elif isinstance(user_shape, list) or isinstance(user_shape, tuple):
        if not all(isinstance(dim, numbers.Number) for dim in user_shape):
            raise TypeError("shape_list contains incorrect dim", user_shape)
        shape = TensorShape(dims=user_shape)
    else:
        raise ValueError("Invalid shape value", user_shape)
    return shape


# This dict also allows getting proper proto Dtypes for int, str and other builtin python types
NP_TO_HS_DTYPE = {
    np.int8: DT_INT8,
    np.int16: DT_INT16,
    np.int32: DT_INT32,
    np.int64: DT_INT64,
    np.int: DT_INT64,
    np.uint8: DT_UINT8,
    np.uint16: DT_UINT16,
    np.uint32: DT_UINT32,
    np.uint64: DT_UINT64,
    np.float16: DT_HALF,
    np.float32: DT_FLOAT,
    np.float64: DT_DOUBLE,
    np.float: DT_DOUBLE,
    np.float128: None,
    np.complex64: DT_COMPLEX64,
    np.complex128: DT_COMPLEX128,
    np.complex256: None,
    np.bool_: DT_BOOL,
    np.bool: DT_BOOL,
    np.str_: DT_STRING,
    np.unicode_: DT_STRING,
    str: DT_STRING,
}

HS_TO_NP_DTYPE = dict([(v, k) for k, v in NP_TO_HS_DTYPE.items()])


def proto_to_np_dtype(dt):
    if HS_TO_NP_DTYPE.get(dt) is not None:
        return HS_TO_NP_DTYPE[dt]
    else:
        raise TypeError("Datatype {}({}) is not supported in HydroSDK".format(DataType.Name(dt), dt))


def np_to_proto_dtype(dt):
    if NP_TO_HS_DTYPE.get(dt) is not None:
        return NP_TO_HS_DTYPE[dt]
    else:
        raise TypeError("Datatype {} is not supported in HydroSDK".format(dt))


def find_in_list_by_name(some_list: list, name: str):
    """
    Helper created to find ModelField by required name

    :param some_list: list of objects with name field
    :param name: name of object to be found
    :raises ValueError: not found
    :return: object with required name
    """
    for item in some_list:
        if item.name == name:
            return item

    raise ValueError(f"List: {some_list} doesn't have this name: {name}")


class PredictorDT(Enum):
    DICT_PYTHON = "dict"
    DICT_NP_ARRAY = "nparray"
    DF = "dataframe"
