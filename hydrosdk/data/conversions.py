import numpy as np
import pandas as pd
from hydro_serving_grpc import TensorProto, DataType, TensorShapeProto

from hydrosdk.data.types import NP_TO_HS_DTYPE, DTYPE_TO_FIELDNAME, np2proto_shape, PY_TO_DTYPE, PredictorDT, \
    signature_get_item


def numpy_data_to_tensor_proto(data, dtype, shape):
    proto_dtype = NP_TO_HS_DTYPE[dtype.type]
    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: data.flatten(),
        "dtype": proto_dtype,
        "tensor_shape": np2proto_shape(shape)
    }
    return TensorProto(**kwargs)


def dtype_to_tensor_proto(data: int, dtype: str, shape: TensorShapeProto):
    proto_dtype = DataType.Value(DataType.Name(dtype))
    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: [data],
        "dtype": proto_dtype,
        "tensor_shape": shape
    }
    return TensorProto(**kwargs)


def python_dtype_to_proto(value, key, signature) -> TensorProto:
    return numpy_data_to_tensor_proto(value, signature.inputs[key].dtype, signature.inputs[key].shape)


def convert_inputs_to_tensor_proto(inputs, signature) -> tuple:
    """

    :param inputs:
    :param signature:
    :return:
    """
    tensors = {}
    if isinstance(inputs, dict):
        return_type = PredictorDT.DICT
        for key, value in inputs.items():
            # TODO: do we really need list an py_to_dtype ifs instead of one?
            if type(value) in PY_TO_DTYPE:  # x: 1
                current_signature = signature_get_item(signature=signature, item=key)
                tensors[key] = dtype_to_tensor_proto(value, current_signature.dtype,
                                                     current_signature.shape)
            elif isinstance(value, list):  # x: [1,2,3,4]
                for list_el in value:
                    current_signature = signature_get_item(signature=signature, item=key)
                    tensors[key] = dtype_to_tensor_proto(list_el, current_signature.dtype,
                                                         current_signature.shape)

            elif isinstance(value, np.ndarray):  # x: np.ndarray(1,2,3,4)
                return_type = PredictorDT.NP_ARRAY
                tensors[key] = numpy_data_to_tensor_proto(value, value.dtype, value.shape)
            else:
                raise TypeError("Unsupported objects in dict values {}".format(type(value)))
    # TODO: refactor df + pandas into one
    elif isinstance(inputs, pd.DataFrame):
        return_type = PredictorDT.PD_DF
        for key, value in dict(inputs).items():
            tensors[key] = numpy_data_to_tensor_proto(value.ravel(), value.dtype, value.shape)
    # TODO: do we need series?
    # elif isinstance(inputs, pd.Series):
    #     return_type = PredictorDT.PD_SERIES
    #     for key, value in inputs.items():
    #         tensors[key] = numpy_data_to_tensor_proto(value, value.dtypes, value.shape)
    else:
        raise ValueError(
            "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
                type(inputs)))

    return tensors, return_type
