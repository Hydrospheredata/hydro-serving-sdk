from collections import namedtuple
from random import random

import numpy as np
import pandas as pd
import pytest
from hydro_serving_grpc.tf import *
from hydro_serving_grpc.tf import TensorProto, TensorShapeProto

from hydrosdk.data.conversions import np_to_tensor_proto, tensor_proto_to_np, proto_to_np_dtype, \
    tensor_shape_proto_from_tuple, list_to_tensor_proto, tensor_proto_to_py, isinstance_namedtuple
from hydrosdk.data.types import DTYPE_TO_FIELDNAME, np_to_proto_dtype, PredictorDT, find_in_list_by_name
from hydrosdk.servable import Servable
from tests.common_fixtures import * 
from tests.utils import *

int_dtypes = [DT_INT64, DT_UINT16, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_UINT32, DT_UINT64]
float_types = [DT_DOUBLE, DT_FLOAT, ]
quantized_int_types = [DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, DT_QUINT16]
unsupported_dtypes = [DT_BFLOAT16, DT_INVALID, DT_MAP, DT_RESOURCE, DT_VARIANT]
complex_dtypes = [DT_COMPLEX64, DT_COMPLEX128, ]

supported_float_np_types = [np.single, np.double, np.float, np.float32, np.float64, np.float_]
supported_int_np_types = [np.int, np.int64, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.uint64]
supported_complex_np_types = [np.complex, np.complex128, np.complex64, np.csingle, np.cdouble, np.complex_]
unsupported_np_types = [np.float128, np.complex256, np.object, np.void,
                        np.longlong, np.ulonglong, np.clongdouble]


@pytest.fixture(scope="module")
def servable_tensor(cluster: Cluster, local_model: LocalModel):
    mv: ModelVersion = local_model.upload(cluster)
    mv.lock_till_released()
    sv: Servable = Servable.create(cluster, mv.name, mv.version)
    servable_lock_till_serving(cluster, sv.name)
    yield sv
    Servable.delete(cluster, sv.name)


class TestConversion:
    @pytest.mark.parametrize("dtype", int_dtypes + float_types + [DT_STRING, DT_BOOL, DT_HALF])
    def test_proto_dtype_to_np_to_proto(self, dtype):
        np_type = proto_to_np_dtype(dtype)
        restored_dtype = np_to_proto_dtype(np_type)
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

    @pytest.mark.parametrize("shape", [TensorShapeProto(), ])
    def test_int_scalar_tensor_to_np_scalar_back_to_tensor(self, shape):
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

    @pytest.mark.xfail(strict=True, raises=TypeError)
    @pytest.mark.parametrize("np_dtype", unsupported_np_types)
    def test_unsupported_np_to_tensor_to_np(self, np_dtype):
        x = np.array([1.0, 2.0, 3.0], dtype=np_dtype)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)

    def test_bool_np_to_tensor_to_np(self):
        x = np.array([True, False, True], dtype=np.bool)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)

    def test_bool_scalar_to_tensor_and_back(self):
        x = np.bool()
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert x == x_restored

    def test_str_np_to_tensor_to_np(self):
        x = np.array(["a", "b", "c"], dtype=np.str)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)

    def test_str_scalar_to_tensor_and_back(self):
        x = np.str("a")
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert x == x_restored

    @pytest.mark.parametrize("dt", supported_complex_np_types)
    def test_complex_np_to_tensor_to_np(self, dt):
        x = np.array([-1 - 1j, -1 + 1j, +1 - 1j, +1 + 1j], dtype=dt)
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert np.all(x == x_restored)

    @pytest.mark.parametrize("dt", supported_complex_np_types)
    def test_complex_scalar_to_tensor_to_np(self, dt):
        x = np.array([-1 - 1j], dtype=dt)[0]
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert x == x_restored

    @pytest.mark.parametrize("np_dtype", supported_float_np_types + supported_int_np_types)
    def test_np_scalar_to_tensor_to_np(self, np_dtype):
        x = np.array([1.0], dtype=np_dtype)[0]
        tensor_proto = np_to_tensor_proto(x)
        x_restored = tensor_proto_to_np(tensor_proto)
        assert x == x_restored

    def test_isinstance_namedtuple_namedtuple(self):
        Point = namedtuple('Point', ['x', 'y'])
        pt = Point(1.0, 5.0)
        assert isinstance_namedtuple(pt)

    def test_isinstance_namedtuple_tuple(self):
        pt = (1, 2, 3)
        assert not isinstance_namedtuple(pt)

    def test_isinstance_namedtuple_itertuples(self):
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        for row in df.itertuples():
            assert isinstance_namedtuple(row)

    def test_tensor_proto_to_py(self, servable_tensor):
        list_value = [int(random() * 1e5)]
        predictor = servable_tensor.predictor(return_type=PredictorDT.DICT_PYTHON)
        signature_field = find_in_list_by_name(some_list=predictor.signature.inputs, name="input")
        tensor_proto = list_to_tensor_proto(list_value, signature_field.dtype, signature_field.shape)
        value_again = tensor_proto_to_py(t=tensor_proto)
        assert list_value == value_again
        assert isinstance(value_again, list)
