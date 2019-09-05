from enum import Enum
from typing import Tuple, List

import numpy as np
from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField

from .proto_conversion_utils import np2proto_shape, np2proto_dtype, proto2np_shape, proto2np_dtype


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
    TABULAR = 1
    IMAGE = 2
    TEXT = 3


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
                    return False, "Tensor {} has scalar shape. Recieved tensor is of shape {}".format(self.name, other_shape)

        if len(self.shape) == len(other_shape):
            possible_shape = tuple([AnyDimSize if s == -1 else s for s in self.shape])
            is_valid = possible_shape == other_shape
        else:
            is_valid = False

        if not is_valid:
            return False, "Tensor {} has invalid other_shape. Expected {}, received {}".format(self.name, self.shape, other_shape)
        else:
            return True, None

    def validate_dtype(self, other_dtype):
        # TODO Use np.can_cast rules? i.e. float32 can be casted to float16
        #  add strict = True for strict matching and strict = False for supporting subtypes?

        if self.dtype == other_dtype:
            return True, None
        else:
            return False, "Tensor {} has invalid dtype. Expected {}, recieved {}".format(self.name, self.dtype, other_dtype)

    def validate(self, x: np.array):
        """
        Return bool whether array is valid for this field and error message, if not valid.
        Error message is None if array is valid.
        :param x: input ndarray
        :return: is_valid, error_message
        """
        is_shape_valid, shape_error_message = self.validate_shape(x.shape)
        is_dtype_valid, dtype_error_message = self.validate_dtype(x.dtype)
        error_message = ', '.join(filter(None, (shape_error_message, dtype_error_message)))
        return is_dtype_valid & is_dtype_valid, error_message if error_message else None


class Signature:

    def __init__(self, name: str, inputs: List[Field], outputs: List[Field]):
        self.name = name
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

    @staticmethod
    def __validate_dict(tensor_dict, fields):
        is_valid = True
        error_messages = []

        extra_tensor_names = set(tensor_dict.keys()).difference(set(map(lambda x: x.name, fields)))
        missing_tensor_names = set(map(lambda x: x.name, fields)).difference(set(tensor_dict.keys()))
        common_tensor_names = set(tensor_dict.keys()).intersection(set(map(lambda x: x.name, fields)))

        common_tensor_names = list(common_tensor_names)
        common_fields = tensor_dict(
            zip(common_tensor_names, [next(filter(lambda x: x.name == name, fields)).val for name in common_tensor_names]))

        if extra_tensor_names:
            is_valid = False
        error_messages.append("Extra tensors provided: {}".format(extra_tensor_names))

        if missing_tensor_names:
            is_valid = False
        error_messages.append("Missing tensors: {}".format(missing_tensor_names))

        for tensor_name in common_tensor_names:
            is_tensor_valid, error_message = common_fields[tensor_name].validate(tensor_dict[tensor_name])
            is_valid &= is_tensor_valid
            if error_message:
                error_messages.append(error_message)

        return is_valid, error_messages

    def validate_input(self, input_dict):
        return self.__validate_dict(input_dict, self.inputs)

    def validate_output(self, output_dict):
        return self.__validate_dict(output_dict, self.outputs)

    def merge_sequential(self, other_signature):
        if set(self.outputs) != set(other_signature.inputs):
            raise ContractViolationException("Invalid blablalba")

        return Signature("|".join([self.name, other_signature.name]),
                         self.inputs,
                         other_signature.outputs)

    def mock_input_data(self):
        # work in progress
        input_dict = {}
        for field in self.inputs:
            x = (10 * np.random.randn(*field.shape)).astype(field.dtype)
            input_dict[field.name] = x

        return input_dict


class Contract:

    @staticmethod
    def from_fields(sig_name, inputs, outputs):
        pass

    def __init__(self, model_name, signature: Signature):
        self.signature = signature
        self.model_name = model_name  # zochem model name ???

    def validate(self):
        pass

    def merge_sequential(self, other_contract):
        return Contract(self.model_name, self.signature.merge_sequential(other_contract.signature))

    def merge_parallel(self, other_contract):
        pass

    def mock_data(self):
        return self.signature.mock_input_data()

    @property
    def proto(self):
        return ModelContract(model_name=self.model_name, predict=self.signature.proto)

    @classmethod
    def from_proto(cls, proto):
        return cls(model_name=proto.model_name,
                   signature=Signature.from_proto(proto.predict))
