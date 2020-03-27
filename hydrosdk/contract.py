import operator
from enum import Enum
from functools import reduce
import numbers

import numpy as np
from hydro_serving_grpc.tf.types_pb2 import *
from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField, DataProfileType
from hydro_serving_grpc import DataType

from hydrosdk.data.types import name2dtype, shape_to_proto, PY_TO_DTYPE, np2proto_dtype, proto2np_dtype


class ContractViolationException(Exception):
    """
    Exception raised when contract is violated
    """
    pass


class ProfilingType(Enum):
    """

    """
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


def field_from_dict(name, data_dict):
    """
    Old version of deserialization *data_dict* into ModelField. Should not be used or tested first
    """
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


def field_from_dict_new(name, data_dict):
    """
    Deserialization of *data_dict* into ModelField.

    :param name: name of passed data
    :param data_dict: data
    :raises ValueError: If data_dict is invalid
    :return: ModelField
    """
    shape = data_dict.get("shape")
    dtype = data_dict.get("dtype")
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
                # TODO: why is here field_from_dict old?
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


def signature_to_dict(signature: ModelSignature):
    """
    Serializes signature into signature_name, inputs, outputs

    :param signature: model signature obj
    :raises TypeError: If signature invalid
    :return: dict with signature_name, inputs, outputs
    """
    if not isinstance(signature, ModelSignature):
        raise TypeError("signature is not ModelSignature")
    inputs = []
    for i in signature.inputs:
        inputs.append(field_to_dict(i))
    outputs = []
    for o in signature.outputs:
        outputs.append(field_to_dict(o))
    result_dict = {
        "signatureName": signature.signature_name,
        "inputs": inputs,
        "outputs": outputs
    }
    return result_dict


def contract_to_dict(contract: ModelContract):
    """
    Serializes model contract into model_name, predict

    :param contract: model contract
    :return: dict with model_name, predict
    """
    if contract is None:
        return None
    if not isinstance(contract, ModelContract):
        raise TypeError("contract is not ModelContract")
    signature = signature_to_dict(contract.predict)
    result_dict = {
        "modelName": contract.model_name,
        "predict": signature
    }
    return result_dict


def field_to_dict(field: ModelField):
    """
    Serializes model field into name, profile and optional shape

    :param field: model field
    :raises TypeError: If field is invalid
    :return: dict with name and profile
    """
    if not isinstance(field, ModelField):
        raise TypeError("field is not ModelField")
    result_dict = {
        "name": field.name,
        "profile": DataProfileType.Name(field.profile)
    }
    if field.shape is not None:
        result_dict["shape"] = shape_to_dict(field.shape)

    attach_ds(result_dict, field)
    return result_dict


def attach_ds(result_dict, field):
    """
    Adds dtype or subfields

    :param result_dict:
    :param field:
    :raises ValueError: If field invalid
    :return: result_dict with dtype or subfields
    """
    if field.dtype is not None:
        result_dict["dtype"] = DataType.Name(field.dtype)
    elif field.subfields is not None:
        subfields = []
        for f in field.subfields:
            subfields.append(field_to_dict(f))
        result_dict["subfields"] = subfields
    else:
        raise ValueError("Invalid ModelField type")
    return result_dict


def shape_to_dict(shape):
    """
    Serializes model field's shape to dict

    :param shape: TensorShapeProto
    :return: dict with dim and unknown rank
    """
    dims = []
    for d in shape.dim:
        dims.append({"size": d.size, "name": d.name})
    result_dict = {
        "dim": dims,
        "unknownRank": shape.unknown_rank
    }
    return result_dict


def contract_from_dict_yaml(data_dict):
    """
    Old version of deserialization of yaml dict into model contract. Should not be used or tested first

    :param data_dict: contract from yaml
    :return: model contract
    """
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


def contract_from_dict(data_dict):
    """
    Deserialization of yaml dict into model contract

    :param data_dict: contract from yaml
    :return: model contract
    """

    if data_dict is None:
        return None
    name = data_dict.get("modelName", "Predict")
    inputs = []
    outputs = []
    for item in data_dict["predict"]["inputs"]:
        input_item = field_from_dict_new(item.get("name"), item)
        inputs.append(input_item)

    for item in data_dict["predict"]["outputs"]:
        output_item = field_from_dict_new(item.get("name"), item)
        outputs.append(output_item)

    signature = ModelSignature(
        signature_name=name,
        inputs=inputs,
        outputs=outputs
    )
    return ModelContract(model_name="model", predict=signature)


def parse_field(name, dtype, shape, profile=ProfilingType.NONE):
    """
    Deserializes into model field

    :param name: name of model field
    :param dtype: data type of model field
    :param shape: shape of model field
    :param profile: profile of model field
    :raises ValueError: If dtype is invalid
    :return: model field obj
    """
    if profile not in DataProfileType.keys():
        profile = "NONE"

    if dtype is None:
        raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", name)
    elif isinstance(dtype, dict):
        subfields_buffer = []
        for name, v in dtype.items():
            subfield = parse_field(name, v['dtype'], v['shape'], v['profile'])
            subfields_buffer.append(subfield)
        return ModelField(
            name=name,
            shape=shape_to_proto(shape),
            subfields=ModelField.Subfield(data=subfields_buffer),
            profile=profile
        )
    else:
        if dtype in DataType.keys():  # exact name e.g. DT_STRING
            result_dtype = dtype
        elif dtype in DataType.values():
            result_dtype = dtype
        elif isinstance(dtype, str):  # string alias
            result_dtype = name2dtype(dtype)
        elif isinstance(dtype, type):  # type. could be python or numpy type
            result_dtype = PY_TO_DTYPE.get(dtype)
            if not result_dtype:
                result_dtype = np2proto_dtype(dtype)
        else:
            result_dtype = DT_INVALID

        if result_dtype == DT_INVALID:
            raise ValueError("Invalid contract: {} field has invalid datatype {}".format(name, dtype))
        return ModelField(
            name=name,
            shape=shape_to_proto(shape),
            dtype=result_dtype,
            profile=profile
        )


class SignatureBuilder:
    """
    Build Model Signature
    """
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def with_input(self, name, dtype, shape, profile=ProfilingType.NONE):
        """
        Adds input to the SignatureBuilder

        :param name:
        :param dtype:
        :param shape:
        :param profile:
        :return: self SignatureBuilder
        """
        return self.__with_field(self.inputs, name, dtype, shape, profile)

    def with_output(self, name, dtype, shape, profile=ProfilingType.NONE):
        """
        Adds output to the SignatureBuilder

        :param name:
        :param dtype:
        :param shape:
        :param profile:
        :return: self SignatureBuilder
        """
        return self.__with_field(self.outputs, name, dtype, shape, profile)

    def build(self):
        """
        Creates Model Signature

        :return: ModelSignature obj
        """
        return ModelSignature(
            signature_name=self.name,
            inputs=self.inputs,
            outputs=self.outputs
        )

    def __with_field(self, collection, name, dtype, shape, profile=ProfilingType.NONE):
        """
        Adds fields to the SignatureBuilder

        :param collection: input or output
        :param name:
        :param dtype:
        :param shape:
        :param profile:
        :return: self SignatureBuilder obj
        """
        proto_field = parse_field(name, dtype, shape, profile)
        collection.append(proto_field)
        return self


class AnyDimSize(object):
    """
    Validation class for dimensions
    """
    def __eq__(self, other):
        """
        If dimension is of Number type than equal

        :param other: dimension
        :raises TypeError: If other not Number
        :return:
        """
        if isinstance(other, numbers.Number):
            return True
        else:
            raise TypeError("Unexpected other argument {}".format(other))

# TODO: method not used
def are_shapes_compatible(a, b):
    """
    Compares if shapes are compatible

    :param a:
    :param b:
    :return: result of comparision as bool
    """

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

# TODO: method not used
def are_dtypes_compatible(a, b, strict=False):
    """
    Compares if data types are compatible

    :param a:
    :param b:
    :param strict:
    :return: result of comparision as bool
    """
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

#TODO: method not used
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


def mock_input_data(signature: ModelSignature):
    """
    Creates dummy input data

    :param signature:
    :return: list of input tensors
    """
    input_tensors = []
    for field in signature.inputs:
        simple_shape = []
        if field.shape:
            simple_shape = [x.size if x.size > 0 else 1 for x in field.shape.dim] # TODO change -1 to random N, where N <=5
        if len(simple_shape) == 0:
            simple_shape = [1]
        field_shape = tuple(np.abs(simple_shape))
        size = reduce(operator.mul, field_shape)
        npdtype = proto2np_dtype(field.dtype)
        if field.dtype == DT_BOOL:
            x = (np.random.randn(*field_shape) >= 0).astype(np.bool)
        elif field.dtype in [DT_FLOAT, DT_HALF, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64]:
            x = np.random.randn(*field_shape).astype(npdtype)
        elif field.dtype in [DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64]:
            _min, _max = np.iinfo(npdtype).min, np.iinfo(npdtype).max
            x = np.random.randint(_min, _max, size, dtype=npdtype).reshape(field_shape)
        elif field.dtype == DT_STRING:
            x = np.array(["foo"] * size).reshape(field_shape)
        else:
            raise Exception("{} does not support mock data generation yet.".format(field.dtype))
        input_tensors.append(x)
    return input_tensors


# def contract_from_df(example_df):
#     """
#     Suggest contract definition for model contract from dataframe
#     :param example_df:
#     :return:
#     """
#     signature_name = getattr(example_df, "name", "predict")
#     inputs = []
#     for name, dtype in zip(example_df.columns, example_df.dtypes):
#         field = ModelField()
#         inputs.append(Field(name, (-1, 1), dtype.type, profile=ProfilingType.NUMERICAL))
#     signature = ModelSignature(signature_name, inputs, [])
#     return ModelContract(predict=signature)
