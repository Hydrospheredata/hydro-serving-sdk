import numpy as np
import pytest
from hydro_serving_grpc.tf import *
from hydro_serving_grpc.tf import TensorProto, TensorShapeProto

from hydrosdk.data.conversions import np_to_tensor_proto, tensor_proto_to_np, from_proto_dtype
from hydrosdk.data.types import DTYPE_TO_FIELDNAME, np2proto_dtype, tensor_shape_proto_from_tuple

int_dtypes = [DT_INT64, DT_UINT16, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_UINT32, DT_UINT64]
float_types = [DT_DOUBLE, DT_FLOAT, ]
quantized_int_types = [DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, DT_QUINT16]
unsupported_dtypes = [DT_BFLOAT16, DT_INVALID, DT_MAP, DT_RESOURCE, DT_VARIANT]

supported_float_np_types = [np.single, np.double, np.float, np.float32, np.float64, ]
supported_int_np_types = [np.int, np.int64, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.uint64]

# TODO add more unsupported dtypes
unsupported_np_types = [np.float128, ]


class TestConversion:

    @pytest.mark.parametrize("dtype", int_dtypes + float_types + [DT_STRING, DT_BOOL, DT_HALF])
    def test_proto_dtype_to_np_to_proto(self, dtype):
        np_type = from_proto_dtype(dtype)
        restored_dtype = np2proto_dtype(np_type)
        assert dtype == restored_dtype

    @pytest.mark.parametrize("np_shape", [(100, 1), (-1, 100), (-1, 1), (1,), (10, 10, 10, 10,)])
    def test_np_shape_to_proto_and_back(self, np_shape):
        proto_shape = tensor_shape_proto_from_tuple(np_shape)
        restored_shape = tuple([dim.size for dim in proto_shape.dim])
        assert np_shape == restored_shape

    @pytest.mark.parametrize("dtype", int_dtypes + float_types)
    def test_tensor_to_np_array_to_tensor(self, dtype):
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=1)])

        tp_kwargs = {DTYPE_TO_FIELDNAME[dtype]: [1, 2, 3],
                     "dtype": dtype,
                     "tensor_shape": tensor_shape}

        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        restored_tensor_proto = np_to_tensor_proto(np_representation)
        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.parametrize("dtype", float_types)
    def test_tensor_to_np_array_to_tensor(self, dtype):
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=1)])

        tp_kwargs = {DTYPE_TO_FIELDNAME[dtype]: [1.10, 2.20, 3.30],
                     "dtype": dtype,
                     "tensor_shape": tensor_shape}

        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        restored_tensor_proto = np_to_tensor_proto(np_representation)
        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.parametrize("dtype", [DT_HALF])
    def test_half_dtype_conversion(self, dtype):
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=1)])
        tp_kwargs = {DTYPE_TO_FIELDNAME[dtype]: np.array([1.10, 2.20, 3.30], dtype=np.float16).view(np.uint16),
                     "dtype": dtype,
                     "tensor_shape": tensor_shape}

        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        restored_tensor_proto = np_to_tensor_proto(np_representation)
        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.parametrize("dtype", [DT_HALF])
    def test_half_dtype_scalar_conversion(self, dtype):
        tensor_shape = TensorShapeProto()
        tp_kwargs = {DTYPE_TO_FIELDNAME[dtype]: np.array([1.1], dtype=np.float16).view(np.uint16),
                     "dtype": dtype,
                     "tensor_shape": tensor_shape}

        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        print(type(np_representation))
        restored_tensor_proto = np_to_tensor_proto(np_representation)
        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.xfail()
    @pytest.mark.parametrize("dtype", quantized_int_types + unsupported_dtypes)
    def test_quantized_int_conversion(self, dtype):
        tensor_shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=3), TensorShapeProto.Dim(size=1)])

        tp_kwargs = {DTYPE_TO_FIELDNAME[dtype]: [1, 2, 3],
                     "dtype": dtype,
                     "tensor_shape": tensor_shape}

        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        restored_tensor_proto = np_to_tensor_proto(np_representation)
        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.parametrize("shape", [TensorShapeProto(), ])
    def test_tensor_to_np_scalar_to_tensor(self, shape):
        tp_kwargs = {"int64_val": [1], "dtype": DT_INT64, "tensor_shape": shape}
        original_tensor_proto = TensorProto(**tp_kwargs)
        np_representation = tensor_proto_to_np(original_tensor_proto)
        restored_tensor_proto = np_to_tensor_proto(np_representation)

        assert restored_tensor_proto == original_tensor_proto

    @pytest.mark.parametrize("np_dtype", supported_float_np_types + supported_int_np_types)
    def test_np_to_tensor_to_np(self, np_dtype):
        x = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)

    @pytest.mark.xfail
    @pytest.mark.parametrize("np_dtype", unsupported_np_types)
    def test_np_to_tensor_to_np(self, np_dtype):
        x = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)