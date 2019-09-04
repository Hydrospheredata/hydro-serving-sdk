from enum import Enum
from typing import Tuple, List

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
        pass

    def merge_parallel(self, other_contract):
        pass

    def mock_data(self):
        pass

    @property
    def proto(self):
        return ModelContract(model_name=self.model_name, predict=self.signature.proto)

    @classmethod
    def from_proto(cls, proto):
        return cls(model_name=proto.model_name,
                   signature=Signature.from_proto(proto.predict))
