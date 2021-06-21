import numbers
import operator
from enum import Enum
from functools import reduce
from typing import Optional, Union, Iterable, List

import numpy as np
from hydro_serving_grpc.serving.contract.signature_pb2 import ModelSignature
from hydro_serving_grpc.serving.contract.field_pb2 import ModelField
from hydro_serving_grpc.serving.contract.types_pb2 import (
    DT_INVALID, DT_BOOL, DT_FLOAT, DT_HALF, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
    DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64,
    DT_STRING, DataProfileType, DataType,
)

from hydrosdk.data.types import (
    alias_to_proto_dtype, shape_to_proto, PY_TO_DTYPE, np_to_proto_dtype, proto_to_np_dtype,
)
from hydrosdk.exceptions import SignatureViolationException

class ProfilingType(Enum):
    """
    Profiling Types are used to tell monitoring services how to analyse signature fields.
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
    Serializes ModelSignature into a signature dict

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
        "profile": DataProfileType.Name(field.profile),
        "shape": shape_to_dict(field.shape)
    }

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


def shape_to_dict(shape: Optional['TensorShape']=None) -> dict:
    """
    Serializes model field's shape to dict

    :param shape: TensorShape
    :return: dict with dim
    """
    return None if shape is None else {"dims": list(shape.dims)}


def signature_dict_to_ModelSignature(data: dict) -> ModelSignature:
    """
    Internal method.
    A method that creates ModelSignature out of a signature dict.
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

    return ModelSignature(
        signature_name=signature_name,
        inputs=frmt_inputs,
        outputs=frmt_outputs
    )


def parse_field(name: str, dtype: Union[str, int, np.dtype],
                shape: Iterable[int], profile: ProfilingType = ProfilingType.NONE) -> ModelField:
    """
    Creates a proto ModelField object

    :param name: name of a model field
    :param dtype: data type of model field, either string dtype alias ("double", "int" .. ),
     proto DataType value or name, or a Numpy data type
    :param shape: shape of a model field
    :param profile: profile of a model field
    :raises ValueError: If dtype is invalid
    :return: ModelField proto object
    """
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
            profile=profile.value
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
            raise ValueError(f"Invalid contract: {name} field has invalid datatype {dtype}")
        return ModelField(
            name=name,
            shape=shape_to_proto(shape),
            dtype=result_dtype,
            profile=profile.value
        )


class SignatureBuilder:
    def __init__(self, name):
        """
        SignatureBuilder is used to help with the creation of a ModelSignature.

        Example:
            signature = SignatureBuilder('infer') \
                .with_input('x', 'double', 'scalar') \
                .with_output('y', 'double', 'scalar').build()
        """
        self.name = name
        self.inputs = []
        self.outputs = []

    def with_input(self, name: str, dtype: Union[np.dtype, str, type], 
                   shape: Union[str, Iterable[int]], 
                   profile: ProfilingType = ProfilingType.NONE) -> 'SignatureBuilder':
        """
        Adds an input field to the current ModelSignature

        :param name: string containing a name of the field
        :param dtype: type of the field (one of: DataType, numpy's dtype, string representing datatypes
            like 'double', 'int64', standard python types like `int`, `float`, `str`)
        :param shape: shape of the field (one of: 'scalar', iterable of ints)
        :param profile: one of the options from ProfilingType
        :return: SignatureBuilder object
        """
        return self.__with_field(self.inputs, name, dtype, shape, profile)

    def with_output(self, name: str, dtype: Union[np.dtype, str, type], 
                    shape: Union[str, Iterable[int]], 
                    profile: ProfilingType = ProfilingType.NONE) -> 'SignatureBuilder':
        """
        Adds an output field to the current ModelSignature

        :param name: string containing a name of the field
        :param dtype: type of the field (one of: DataType, numpy's dtype, string representing datatypes
            like 'double', 'int64', standard python types like `int`, `float`, `str`)
        :param shape: shape of the field (one of: 'scalar', iterable of ints)
        :param profile: one of the options from ProfilingType
        :return: SignatureBuilder object
        """
        return self.__with_field(self.outputs, name, dtype, shape, profile)

    def build(self) -> ModelSignature:
        """
        Creates ModelSignature
        :return: ModelSignature proto object
        """
        return ModelSignature(
            signature_name=self.name,
            inputs=self.inputs,
            outputs=self.outputs
        )
                    
    def __with_field(self, collection: List[ModelField], name: str, 
                     dtype: Union[np.dtype, str, type], shape: Union[str, Iterable[int]], 
                     profile: ProfilingType = ProfilingType.NONE) -> 'SignatureBuilder':
        """
        Adds fields to the SignatureBuilder

        :param collection: input or output
        :param name: string containing a name of the field
        :param dtype: type of the field (one of: DataType, numpy's dtype, string representing datatypes
            like 'double', 'int64', standard python types like `int`, `float`, `str`)
        :param shape: shape of the field (one of: 'scalar', iterable of ints)
        :param profile: one of the options from ProfilingType
        :return: SignatureBuilder object
        """
        if not isinstance(profile, ProfilingType):
            raise ValueError("`profile` should be of instance ProfilingType.")

        proto_field = parse_field(name, dtype, shape, profile)
        collection.append(proto_field)
        return self


class AnyDimSize(object):
    """
    Validation class for dimensions, used for -1 dims
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
            raise TypeError(f"Unexpected other argument {other}")


def validate_signature(signature: ModelSignature):
    if not signature.signature_name:
        raise SignatureViolationException("Creating model without signature_name is not allowed")
    if len(signature.inputs) == 0:
        raise SignatureViolationException("Creating model without inputs is not allowed")
    if len(signature.outputs) == 0:
        raise SignatureViolationException("Creating model without outputs is not allowed")
    for model_field in signature.inputs:
        if model_field.dtype == 0:
            raise SignatureViolationException("Creating model with invalid dtype in the signature inputs is not allowed")
    for model_field in signature.outputs:
        if model_field.dtype == 0:
            raise SignatureViolationException("Creating model with invalid dtype in the signature outputs is not allowed")


def mock_input_data(signature: ModelSignature):
    """
    Creates dummy input data

    :param signature:
    :return: list of input tensors
    """
    input_tensors = []
    for field in signature.inputs:
        simple_shape = field.shape.dims or [1]
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
            raise ValueError(f"{field.dtype} does not support mock data generation yet.")
        input_tensors.append(x)
    return input_tensors
