import numbers
import operator
from enum import Enum
from functools import reduce
from typing import Optional

import numpy as np
from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField, DataProfileType
from hydro_serving_grpc.tf.types_pb2 import *

from hydrosdk.data.types import alias_to_proto_dtype, shape_to_proto, PY_TO_DTYPE, np_to_proto_dtype, proto_to_np_dtype


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


def field_from_dict(field_name: str, field_dict: dict) -> ModelField:
    """
    Deserialization into ModelField.

    :param field_name:
    :param field_dict: data
    :raises ValueError: If data_dict is invalid
    :return: ModelField
    """

    shape = field_dict.get("shape")

    dtype = field_dict.get("dtype")
    if not dtype:
        dtype = field_dict.get("type")

    subfields = field_dict.get("subfields")
    if not subfields:
        subfields = field_dict.get("fields")

    raw_profile = field_dict.get("profile", "NONE")
    profile = raw_profile.upper()

    if profile not in DataProfileType.keys():
        profile = "NONE"

    result_dtype = None
    result_subfields = None
    if dtype is None:
        if subfields is None:
            raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", field_name, field_dict)
        else:
            subfields_buffer = []
            for k, v in subfields.items():
                subfield = field_from_dict(k, v)
                subfields_buffer.append(subfield)
            result_subfields = subfields_buffer
    else:
        result_dtype = alias_to_proto_dtype(dtype)

    if result_dtype is not None:
        result_field = ModelField(
            name=field_name,
            shape=shape_to_proto(shape),
            dtype=result_dtype,
            profile=profile
        )
    elif result_subfields is not None:
        result_field = ModelField(
            name=field_name,
            shape=shape_to_proto(shape),
            subfields=ModelField.Subfield(data=result_subfields),
            profile=profile
        )
    else:
        raise ValueError("Invalid field. Neither dtype nor subfields are present in dict", field_name, field_dict)
    return result_field


def ModelSignature_to_signature_dict(signature: ModelSignature) -> dict:
    """
    Serializes ModelSignature into signature dict

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


def ModelContract_to_contract_dict(contract: ModelContract) -> Optional[dict]:
    """
    Serializes ModelContract into contract dict

    :param contract: model contract
    :return: dict with model_name, predict
    """
    if contract is None:
        return None
    if not isinstance(contract, ModelContract):
        raise TypeError("contract is not ModelContract")
    signature = ModelSignature_to_signature_dict(contract.predict)
    result_dict = {
        "modelName": contract.model_name,
        "predict": signature
    }
    return result_dict


def field_to_dict(field: ModelField) -> dict:
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


def attach_ds(result_dict: dict, field) -> dict:
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


def shape_to_dict(shape) -> dict:
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


def _contract_yaml_to_contract_dict(model_name: str, yaml_contract: dict) -> dict:
    """
    Internal method.
    Yaml parsing methods create dict contracts with structure different to contracts we receive from servers, this method restructs yaml contract to the standart contract structure
    Yaml-dict:
    {'name': 'infer', 'inputs': {'input': {'shape': 'scalar', 'type': 'int64', 'profile': 'numerical'}}, 'outputs': {'output': {'shape': 'scalar', 'type': 'int64', 'profile': 'numerical'}}}
    Contract-dict:
    {'modelName': 'infer', 'predict': {'signatureName': 'infer', 'inputs': [{'input': {'shape': 'scalar', 'type': 'int64', 'profile': 'numerical'}}], 'outputs': [{'output': {'shape': 'scalar', 'type': 'int64', 'profile': 'numerical'}}]}}

    :param model_name:
    :param yaml_contract:
    :return:
    """

    # make list of dicts from dict of dicts
    inputs = [{field_key: field_def} for field_key, field_def in yaml_contract.get("inputs").items()]

    frmt_inputs = []
    for input_ in inputs:
        for field_name, field_dict in input_.items():
            frmt_inputs.append(field_from_dict(field_name=field_name, field_dict=field_dict))

    # make list of dicts from dict of dicts
    outputs = [{field_key: field_def} for field_key, field_def in yaml_contract.get("outputs").items()]

    frmt_outputs = []
    # TODO: make one general method for inputs/outputs -> frmt_inputs/frmt_outputs
    for output in outputs:
        for field_name, field_dict in output.items():
            frmt_outputs.append(field_from_dict(field_name=field_name, field_dict=field_dict))

    signature_name = yaml_contract.get("name")

    contract_dict = {
        "modelName": model_name,
        "predict": {
            "signatureName": signature_name,
            "inputs": inputs,
            "outputs": outputs
        }
    }

    return contract_dict


def _contract_dict_to_signature_dict(contract: dict) -> tuple:
    """
    Internal method.
    Makes a signature dict out of contract dict

    :param contract:
    :return:
    """

    name = contract.get("modelName")
    if not name:
        name = contract.get("name", "Predict")
    dict_signature = contract.get("predict")

    return name, dict_signature


def _signature_dict_to_ModelSignature(data: dict) -> ModelSignature:
    """
    Internal method.
    A method that makes ModelSignature out of signature dict
    :param data:
    :return:
    """
    """
    dict to ModelSignature
    :param data:
    :return:
    """
    signature_name = data.get("signatureName")
    inputs = data.get("inputs")  # list of dicts
    outputs = data.get("outputs")  # list of dicts

    frmt_inputs = []
    for input_ in inputs:

        field_name = input_.pop("name", None)
        if not field_name:
            [(field_name, field_dict)] = input_.items()
        else:
            field_dict = input_

        frmt_inputs.append(field_from_dict(field_name=field_name, field_dict=field_dict))

    frmt_outputs = []
    for output in outputs:
        field_name = output.pop("name", None)
        if not field_name:
            [(field_name, field_dict)] = output.items()
        else:
            field_dict = output

        frmt_outputs.append(field_from_dict(field_name=field_name, field_dict=field_dict))

    signature = ModelSignature(
        signature_name=signature_name,
        inputs=frmt_inputs,
        outputs=frmt_outputs
    )

    return signature


def contract_yaml_to_ModelContract(model_name: str, yaml_contract: dict) -> ModelContract:
    """
    Helper method that makes a ModelContract out of contract yaml

    :param model_name:
    :param yaml_contract:
    :return:
    """
    contract_dict = _contract_yaml_to_contract_dict(model_name=model_name, yaml_contract=yaml_contract)
    modelContract = contract_dict_to_ModelContract(contract=contract_dict)
    return modelContract


def contract_dict_to_ModelContract(contract: dict) -> ModelContract:
    """
    Helper method that makes a ModelContract out of contract dict

    :param contract:
    :return:
    """
    model_name, signature_dict = _contract_dict_to_signature_dict(contract=contract)
    modelSignature = _signature_dict_to_ModelSignature(data=signature_dict)
    modelContract = ModelContract(model_name=model_name, predict=modelSignature)
    return modelContract


def signature_dict_to_ModelContract(model_name: str, signature: dict) -> ModelContract:
    """
    Helper method that makes a ModelContract out of signature dict

    :param model_name:
    :param signature:
    :return:
    """
    modelSignature = _signature_dict_to_ModelSignature(data=signature)
    modelContract = ModelContract(model_name=model_name, predict=modelSignature)
    return modelContract


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
        elif dtype in DataType.values():  # int value of DataType
            result_dtype = dtype
        elif isinstance(dtype, str):  # string alias e.g. 'double'
            result_dtype = alias_to_proto_dtype(dtype)
        elif isinstance(dtype, type):  # type. could be python or numpy type
            result_dtype = PY_TO_DTYPE.get(dtype)
            if not result_dtype:
                result_dtype = np_to_proto_dtype(dtype)
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


# TODO: what is it doing here? should contract validation be moved out of LocalModel create?
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


# TODO: method not used
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
            simple_shape = [x.size if x.size > 0 else 1 for x in
                            field.shape.dim]  # TODO change -1 to random N, where N <=5
        if len(simple_shape) == 0:
            simple_shape = [1]
        field_shape = tuple(np.abs(simple_shape))
        size = reduce(operator.mul, field_shape)
        npdtype = proto_to_np_dtype(field.dtype)
        if field.dtype == DT_BOOL:
            x = (np.random.randn(*field_shape) >= 0).astype(np.bool)
        elif field.dtype in [DT_FLOAT, DT_HALF, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64]:
            x = np.random.randn(*field_shape).astype(npdtype)
        elif field.dtype in [DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64]:
            _min, _max = np.iinfo(npdtype).min, np.iinfo(npdtype).max
            x = np.random.randint(_min, _max, size, dtype=npdtype).reshape(field_shape)
        elif field.dtype == DT_STRING:
            x = np.array(["foo"] * size).reshape(field_shape)
        else:
            raise Exception("{} does not support mock data generation yet.".format(field.dtype))
        input_tensors.append(x)
    return input_tensors

# TODO: if commented, should be deleted?
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
