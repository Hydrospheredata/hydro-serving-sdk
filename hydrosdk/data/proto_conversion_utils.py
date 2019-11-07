import numpy as np
from hydro_serving_grpc import DT_STRING, DT_BOOL, \
    DT_HALF, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_INT16, \
    DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, \
    DT_UINT64, DT_COMPLEX64, DT_COMPLEX128, DataType, TensorShapeProto, DT_INVALID

NP_TO_HS_DTYPE = {
    np.int8: DT_INT8,
    np.int16: DT_INT16,
    np.int32: DT_INT32,
    np.int64: DT_INT64,
    np.uint8: DT_UINT8,
    np.uint16: DT_UINT16,
    np.uint32: DT_UINT32,
    np.uint64: DT_UINT64,
    np.float16: DT_HALF,
    np.float32: DT_FLOAT,
    np.float64: DT_DOUBLE,
    np.float128: None,
    np.complex64: DT_COMPLEX64,
    np.complex128: DT_COMPLEX128,
    np.complex256: None,
    np.bool: DT_BOOL,
    np.object: None,
    np.str: DT_STRING,
    np.void: None
}

HS_TO_NP_DTYPE = dict([(v, k) for k, v in NP_TO_HS_DTYPE.items()])

def proto2np_dtype(dt):
    return HS_TO_NP_DTYPE[dt]


def np2proto_dtype(dt):
    return NP_TO_HS_DTYPE.get(dt, DT_INVALID)


def proto2np_shape(tsp):
    if tsp is None or len(tsp.dim) == 0:
        return tuple()
    else:
        shape = tuple([int(s.size) for s in tsp.dim])
    return shape


def np2proto_shape(np_shape):
    shape = TensorShapeProto(dim=[TensorShapeProto.Dim(size=x) for x in np_shape])
    return shape
