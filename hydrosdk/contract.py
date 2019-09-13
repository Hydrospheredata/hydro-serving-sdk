import operator
from enum import Enum
from functools import reduce
from typing import Tuple, List

import numpy as np
import pandas as pd
from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField
from hydro_serving_grpc.tf import TensorProto

<<<<<<< Updated upstream
from .proto_conversion_utils import np2proto_shape, np2proto_dtype, proto2np_shape, proto2np_dtype, NP_DTYPE_TO_ARG_NAME


class AlwaysTrueObj(object):
    def __eq__(self, other):
        return True


# Used for dimension comparisons
AnyDimSize = AlwaysTrueObj()


class ContractViolationException(Exception):
    """
    Exception raised after failed compliance check
    """
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


class Field:

    def __init__(self, name: str, shape: Tuple, dtype, profile=ProfilingType.NONE):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.profile = profile

    @property
    def proto(self):
        return ModelField(name=self.name,
                          shape=np2proto_shape(self.shape),
                          dtype=np2proto_dtype(self.dtype),
                          profile=self.profile)

    @classmethod
    def from_proto(cls, proto: ModelField):
        return cls(name=proto.name,
                   shape=proto2np_shape(proto.shape),
                   dtype=proto2np_dtype(proto.dtype),
                   profile=proto.profile)

    def validate_shape(self, other_shape: Tuple[int]):
        if len(self.shape) == 0:
            # scalar input can be used in following scenarios
            if other_shape == tuple():
                return True, None
            else:
                if max(other_shape) == 1:  # All dimensions are equal to 1
                    return True, None
                else:
                    return False, "Tensor {} has scalar shape. Received tensor is of shape {}".format(self.name, other_shape)

        if len(self.shape) == len(other_shape):
            possible_shape = tuple([AnyDimSize if s == -1 else s for s in self.shape])
            is_valid = possible_shape == other_shape
        else:
            is_valid = False

        if not is_valid:
            return False, "Tensor {} has invalid other_shape. Expected {}, received {}".format(self.name, self.shape, other_shape)
        else:
            return True, None

    def validate_dtype(self, other_dtype, strict=False):
        if strict:
            if self.dtype == other_dtype:
                return True, None
            elif self.dtype.kind == "U":
                # Numpy specify max string length in dtype, but HS has no such info in dtype, so we just check that it is the unicode-string
                return self.dtype.kind == other_dtype.kind, None
            else:
                return False, "Tensor {} has invalid dtype. Expected {}, received {}".format(self.name, self.dtype, other_dtype)
        else:
            if np.can_cast(other_dtype, self.dtype):
                return True, None
            else:
                return False, "Tensor {} has invalid dtype. Expected {}, received {}".format(self.name, self.dtype, other_dtype)

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


class Tensor:

    def __init__(self, name, x):
        self.name = name
        self.shape = x.shape
        self.dtype = x.dtype
        self.x = x

    @property
    def proto(self):
        kwargs = {NP_DTYPE_TO_ARG_NAME[self.dtype]: self.x.flatten(),
                  "dtype": np2proto_dtype(self.dtype),
                  "shape": np2proto_shape(self.shape)}
        return TensorProto(**kwargs)

    @classmethod
    def from_proto(cls, name, proto: ModelField):
        dtype = proto2np_dtype(proto.dtype)
        shape = proto2np_shape(proto.shape)
        x = np.array(getattr(proto, NP_DTYPE_TO_ARG_NAME[dtype]), dtype=dtype).reshape(shape)
        return cls(name, x)


class Signature:

    def __init__(self, name: str, inputs: List[Field], outputs: List[Field]):
        self.name = name

        for field_list in [inputs, outputs]:
            field_names = list(map(lambda x: x.name, field_list))
            unique_names, counts = np.unique(field_names, return_counts=True)
            if any(counts > 1):
                raise ValueError("Tensor names have to be unique. Repeating tensor names: {}".format(unique_names[counts > 1]))

        self.inputs = inputs
        self.outputs = outputs

    @property
    def proto(self):
        return ModelSignature(signature_name=self.name,
                              inputs=[tensor.proto for tensor in self.inputs],
                              outputs=[tensor.proto for tensor in self.outputs])

    @classmethod
    def from_proto(cls, proto: ModelSignature):
        return cls(name=proto.signature_name,
                   inputs=[Field.from_proto(p) for p in proto.inputs],
                   outputs=[Field.from_proto(p) for p in proto.outputs])
=======
class Field:
    def __init__(self, name, datatype, shape):
        self.name = name
        self.dtype = datatype
        self.shape = shape

    def to_proto(self):
        return ModelField()

    @staticmethod
    def from_proto(proto_obj):
        if not isinstance(proto_obj, ModelField):
            raise TypeError("{} is not supported as Field".format(proto_obj))
        

class Contract:
    @staticmethod    
    def from_proto(proto_obj):
        if isinstance(proto_obj, ModelContract):
            return Contract(proto_obj.predict)
        elif isinstance(proto_obj, ModelSignature):
            return Contract(proto_obj)
        else:
            raise TypeError("{} is not supported as Contract".format(proto_obj))
>>>>>>> Stashed changes

    @staticmethod
    def __validate_tensors(tensors, fields):
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

<<<<<<< Updated upstream
        for tensor_name in common_tensor_names:
            is_tensor_valid, error_message = field_dict[tensor_name].validate(tensors_dict[tensor_name])
            is_valid &= is_tensor_valid
            if error_message:
                error_messages.append(error_message)
=======
    def __init__(self, signature):
        if not isinstance(signature, ModelSignature):
            raise TypeError("Invalid signature type. Expected ModelSignature, got {}".format(type(signature)))
        self.signature = signature
>>>>>>> Stashed changes

        return is_valid, error_messages if error_messages else None

    def validate_input(self, input_tensors):
        return self.__validate_tensors(input_tensors, self.inputs)

    def validate_output(self, output_tensors):
        return self.__validate_tensors(output_tensors, self.outputs)

    def merge_sequential(self, other_signature):
        if set(self.outputs) != set(other_signature.inputs):
            raise ContractViolationException("Only strict direct implementation")

        return Signature("|".join([self.name, other_signature.name]),
                         self.inputs,
                         other_signature.outputs)

    def mock_input_data(self):
        input_tensors = []
        for field in self.inputs:
            field_shape = tuple(np.abs(field.shape))  # Fix -1 dimension to be equal to 1
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

    @classmethod
    def from_df(cls, example_df: pd.DataFrame):
        """
        Suggest contract definition for model contract from dataframe
        :param example_df:
        :return:
        """
        signature_name = getattr(example_df, "name", "predict")
        inputs = []
        for name, dtype in zip(example_df.columns, example_df.dtypes):
            inputs.append(Field(name, (-1, 1), dtype.type, profile=ProfilingType.NUMERICAL))
        return cls(signature_name, inputs, [])


class Contract:

    def __init__(self, model_name, signature: Signature):
        self.signature = signature
        self.model_name = model_name

    def merge_sequential(self, other_contract):
        return Contract(self.model_name, self.signature.merge_sequential(other_contract.signature))

    def mock_data(self):
        return self.signature.mock_input_data()

    @property
    def proto(self):
        return ModelContract(model_name=self.model_name, predict=self.signature.proto)

<<<<<<< Updated upstream
    @classmethod
    def from_proto(cls, proto):
        return cls(model_name=proto.model_name,
                   signature=Signature.from_proto(proto.predict))
=======
    def to_proto(self):
        return ModelContract(predict=self.signature)
>>>>>>> Stashed changes
