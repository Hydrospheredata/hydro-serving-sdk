import operator
from enum import Enum
from functools import reduce
import numbers

import numpy as np
from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField, DataProfileType

from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, \
    DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, \
    DT_QUINT16, DT_VARIANT, DT_COMPLEX64, DT_COMPLEX128, DT_INVALID, TensorShapeProto

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


def name2dtype(name):
    return DTYPE_ALIASES_REVERSE.get(name, DT_INVALID)


def dtype2name(dtype):
    return DTYPE_ALIASES.get(dtype)


def dtype_field(dtype):
    return DTYPE_TO_FIELDNAME.get(dtype)


class ContractViolationException(Exception):
    pass


class ProfilingType(Enum):
    NONE = 0
    CATEGORICAL = 1
    NOMINAL = 11
    ORDINAL = 12
    NUMERICAL = 2
    CONTINUOUS = 21
    INTERVAL = 22
    RATIO = 23
    IMAGE = 3
    VIDEO = 4
    AUDIO = 5
    TEXT = 6


def shape_to_proto(user_shape):
    if user_shape == "scalar":
        shape = TensorShapeProto()
    elif user_shape is None:
        shape = None
    elif isinstance(user_shape, list):
        dims = []
        for dim in user_shape:
            if not isinstance(dim, numbers.Number):
                raise TypeError("shape_list contains incorrect dim", user_shape, dim)
            converted = TensorShapeProto.Dim(size=dim)
            dims.append(converted)
        shape = TensorShapeProto(dim=dims)
    else:
        raise ValueError("Invalid shape value", user_shape)
    return shape


def field_from_dict(name, data_dict):
    shape = data_dict.get("shape")
    dtype = data_dict.get("type")
    subfields = data_dict.get("fields")
    raw_profile = data_dict.get("profile", "NONE")
    profile = raw_profile.upper()

    if profile not in DataProfileType.keys():
        profile = "NONE"

    result_dtype = None
    result_subfields = None
    if dtype is None:
        if subfields is None:
            raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name, data_dict)
        else:
            subfields_buffer = []
            for k, v in subfields.items():
                subfield = field_from_dict(k, v)
                subfields_buffer.append(subfield)
            result_subfields = subfields_buffer
    else:
        result_dtype = name2dtype(dtype)
        if result_dtype == DT_INVALID:
            raise ValueError("Invalid contract: {} field has invalid datatype {}".format(name, dtype))

    if result_dtype is not None:
        result_field = ModelField(
            name=name,
            shape=shape_to_proto(shape),
            dtype=result_dtype,
            profile=profile
        )
    elif result_subfields is not None:
        result_field = ModelField(
            name=name,
            shape=shape_to_proto(shape),
            subfields=ModelField.Subfield(data=result_subfields),
            profile=profile
        )
    else:
        raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name, data_dict)
    return result_field


def contract_from_dict(data_dict):
    if data_dict is None:
        return None
    name = data_dict.get("name", "Predict")
    inputs = []
    outputs = []
    for in_key, in_value in data_dict["inputs"].items():
        input = field_from_dict(in_key, in_value)
        inputs.append(input)
    for out_key, out_value in data_dict["outputs"].items():
        output = field_from_dict(out_key, out_value)
        outputs.append(output)
    signature = ModelSignature(
        signature_name=name,
        inputs=inputs,
        outputs=outputs
    )
    return ModelContract(model_name="model", predict=signature)


class SignatureBuilder:
    def __init__(self):
        self.name = []
        self.inputs = []
        self.outputs = []

    def with_input(self, name, dtype, shape, profile=ProfilingType.NONE):
        return self.__with_field(self.inputs, name, dtype, shape, profile)

    def with_output(self, name, dtype, shape, profile=ProfilingType.NONE):
        return self.__with_field(self.outputs, name, dtype, shape, profile)

    def __with_field(self, collection, name, dtype, shape, profile=ProfilingType.NONE):
        if profile not in DataProfileType.keys():
            profile = "NONE"

        if dtype is None:
            raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name)
        elif isinstance(dtype, dict):
            subfields_buffer = []
            for k, v in dtype.items():
                subfield = field_from_dict(k, v)
                subfields_buffer.append(subfield)
            result_subfields = subfields_buffer
            proto_field = ModelField(
                name=name,
                shape=shape_to_proto(shape),
                subfields=ModelField.Subfield(data=result_subfields),
                profile=profile
            )
        else:
            result_dtype = name2dtype(dtype)
            if result_dtype == DT_INVALID:
                raise ValueError("Invalid contract: {} field has invalid datatype {}".format(name, dtype))
            proto_field = ModelField(
                name=name,
                shape=shape_to_proto(shape),
                dtype=result_dtype,
                profile=profile
            )
        collection.append(proto_field)
        return self


class AnyDimSize(object):
    def __eq__(self, other):
        if isinstance(other, numbers.Number):
            return True
        else:
            raise TypeError("Unexpected other argument {}".format(other))


def are_shapes_compatible(a, b):
    if len(a) == 0:
        # scalar input can be used in following scenarios
        if b == tuple():
            return True
        else:
            if max(b) == 1:  # All dimensions are equal to 1
                return True
            else:
                return False
    if len(a) == len(b):
        possible_shape = tuple([AnyDimSize if s == -1 else s for s in a])
        is_valid = possible_shape == b
    else:
        is_valid = False
    return is_valid


def are_dtypes_compatible(a, b, strict=False):
    if strict:
        if a == b:
            return True, None
        elif a.kind == "U":
            # Numpy specify max string length in dtype, but HS has no such info in dtype, so we just check that it is the unicode-string
            return a.kind == b.kind
        else:
            return False
    else:
        if np.can_cast(b, a):
            return True
        else:
            return False


def validate(self, t, strict=False):
    """
    Return bool whether array is valid for this field and error message, if not valid.
    Error message is None if array is valid.
    :param strict: Strict comparison for dtypes.
    :param t: input Tensor
    :return: is_valid, error_message
    """
    is_shape_valid, shape_error_message = self.validate_shape(t.shape)
    is_dtype_valid, dtype_error_message = self.validate_dtype(t.dtype, strict=strict)
    error_message = ', '.join(filter(None, (shape_error_message, dtype_error_message)))
    return is_dtype_valid & is_dtype_valid, error_message if error_message else None


def check_tensor_fields(tensors, fields):
    is_valid = True
    error_messages = []

    tensors_dict = dict(zip(map(lambda x: x.name, tensors), tensors))
    field_dict = dict(zip(map(lambda x: x.name, fields), fields))

    extra_tensor_names = set(tensors_dict.keys()).difference(set(field_dict.keys()))
    missing_tensor_names = set(field_dict.keys()).difference(set(tensors_dict.keys()))
    common_tensor_names = set(tensors_dict.keys()).intersection(set(field_dict.keys()))

    if extra_tensor_names:
        is_valid = False
        error_messages.append("Extra tensors provided: {}".format(extra_tensor_names))

    if missing_tensor_names:
        is_valid = False
        error_messages.append("Missing tensors: {}".format(missing_tensor_names))

    for tensor_name in common_tensor_names:
        is_tensor_valid, error_message = field_dict[tensor_name].validate(tensors_dict[tensor_name])
        is_valid &= is_tensor_valid
        if error_message:
            error_messages.append(error_message)

    return is_valid, error_messages if error_messages else None


def mock_input_data(signature):
    input_tensors = []
    for field in signature.inputs:
        field_shape = tuple(np.abs(field.shape))  # TODO change -1 to random N, where N <=5
        field_shape = field_shape if field_shape else (1,)  # If field_shape is (), fix shape as (1,)
        size = reduce(operator.mul, field_shape)

        if field.dtype.kind == "b":
            x = (np.random.randn(*field_shape) >= 0).astype(np.bool)
        elif field.dtype.kind in ["f", "c"]:
            x = np.random.randn(*field_shape).astype(field.dtype)
        elif field.dtype.kind in ["i", "u"]:
            _min, _max = np.iinfo(field.dtype).min, np.iinfo(field.dtype).max
            x = np.random.randint(_min, _max, size, dtype=field.dtype).reshape(field_shape)
        elif field.dtype == "U":
            x = np.array(["foo"] * size).reshape(field_shape)
        else:
            raise Exception("{} does not support mock data generation yet.".format(field.dtype))
        input_tensors.append(Tensor(field.name, x))
    return input_tensors


def contract_from_df(example_df):
    """
    Suggest contract definition for model contract from dataframe
    :param example_df:
    :return:
    """
    signature_name = getattr(example_df, "name", "predict")
    inputs = []
    for name, dtype in zip(example_df.columns, example_df.dtypes):
        field = ModelField()
        inputs.append(Field(name, (-1, 1), dtype.type, profile=ProfilingType.NUMERICAL))
    signature = ModelSignature(signature_name, inputs, [])
    return ModelContract(predict=signature)
