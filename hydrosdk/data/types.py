import numbers
from enum import Enum

import numpy as np
from hydro_serving_grpc.tf import *

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

DTYPE_ALIASES = {
    DT_STRING: "string",
    DT_BOOL: "bool",
    DT_VARIANT: "variant",

    DT_HALF: "float16",
    DT_FLOAT: "float32",
    DT_DOUBLE: "float64",

    DT_INT8: "int8",
    DT_INT16: "int16",
    DT_INT32: "int32",
    DT_INT64: "int64",

    DT_UINT8: "uint8",
    DT_UINT16: "uint16",
    DT_UINT32: "uint32",
    DT_UINT64: "uint64",

    DT_QINT8: "qint8",
    DT_QINT16: "qint16",
    DT_QINT32: "qint32",

    DT_QUINT8: "quint8",
    DT_QUINT16: "quint16",

    DT_COMPLEX64: "complex64",
    DT_COMPLEX128: "complex128"
}

DTYPE_ALIASES_REVERSE = dict([(v, k) for k, v in DTYPE_ALIASES.items()])

scalar = "scalar"


def name2dtype(name):
    type_ = DTYPE_ALIASES_REVERSE.get(name, DT_INVALID)
    if not type_:
        try:
            type_ = DataType.Value(name)
        except ValueError:
            type_ = DT_INVALID

    return type_


def shape_to_proto(user_shape):
    if isinstance(user_shape, dict):
        user_shape = user_shape.get("dim")

    if user_shape == "scalar":
        shape = TensorShapeProto()
    elif user_shape is None:
        shape = None
    elif isinstance(user_shape, list) or isinstance(user_shape, tuple):
        dims = []
        for dim in user_shape:
            if not isinstance(dim, numbers.Number):
                if isinstance(dim, dict):
                    dim = dim.get("size")
                else:
                    raise TypeError("shape_list contains incorrect dim", user_shape, dim)
            converted = TensorShapeProto.Dim(size=dim)
            dims.append(converted)
        shape = TensorShapeProto(dim=dims)
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
HS_TO_NP_DTYPE[DT_BFLOAT16] = None


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
