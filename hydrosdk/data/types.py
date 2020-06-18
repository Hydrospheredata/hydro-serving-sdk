import numbers
from enum import Enum

from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, \
    DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, \
    DT_QUINT16, DT_VARIANT, DT_COMPLEX64, DT_COMPLEX128, DataType
from hydro_serving_grpc.contract import ModelSignature

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

DTYPE_ALIASES_REVERSE = {
    "string": DT_STRING,
    "bool": DT_BOOL,
    "variant": DT_VARIANT,

    "float16": DT_HALF,
    "half": DT_HALF,
    "float32": DT_FLOAT,
    "float64": DT_DOUBLE,
    "double": DT_DOUBLE,

    "int8": DT_INT8,
    "int16": DT_INT16,
    "int32": DT_INT32,
    "int64": DT_INT64,

    "uint8": DT_UINT8,
    "uint16": DT_UINT16,
    "uint32": DT_UINT32,
    "uint64": DT_UINT64,

    "qint8": DT_QINT8,
    "qint16": DT_QINT16,
    "qint32": DT_QINT32,

    "quint8": DT_QUINT8,
    "quint16": DT_QUINT16,

    "complex64": DT_COMPLEX64,
    "complex128": DT_COMPLEX128,
}

scalar = "scalar"


def name2dtype(name):
    type_ = DTYPE_ALIASES_REVERSE.get(name, DT_INVALID)
    if not type_:
        try:
            type_ = DataType.Value(name)
        except ValueError:
            type_ = DT_INVALID

    return type_


def dtype2name(dtype):
    return DTYPE_ALIASES.get(dtype)


def dtype_field(dtype):
    return DTYPE_TO_FIELDNAME.get(dtype)


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


import numpy as np
from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, \
    DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_COMPLEX64, DT_COMPLEX128, TensorShapeProto, DT_INVALID

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
    # np.float128: None,
    np.complex64: DT_COMPLEX64,
    np.complex128: DT_COMPLEX128,
    # np.complex256: None,
    np.bool: DT_BOOL,
    # np.object: None,
    np.str: DT_STRING,
    # np.void: None
}

HS_TO_NP_DTYPE = dict([(v, k) for k, v in NP_TO_HS_DTYPE.items()])


def proto2np_dtype(dt):
    if dt in HS_TO_NP_DTYPE:
        return HS_TO_NP_DTYPE[dt]
    else:
        raise KeyError("Datatype {}({}) is not supported in HydroSDK".format(DataType.Name(dt), dt))


def np2proto_dtype(dt):
    if dt in NP_TO_HS_DTYPE:
        return NP_TO_HS_DTYPE[dt]
    else:
        raise KeyError("Datatype {} is not supported in HydroSDK".format(dt))


# TODO: method not used
def proto2np_shape(tsp):
    if tsp is None or len(tsp.dim) == 0:
        return tuple()
    else:
        shape = tuple([int(s.size) for s in tsp.dim])
    return shape


# TODO: method not used
def np2proto_shape(np_shape):
    shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=x) for x in np_shape])
    return shape


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
