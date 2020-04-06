from hydro_serving_grpc import TensorProto
from hydrosdk.data.types import NP_TO_HS_DTYPE, DTYPE_TO_FIELDNAME, np2proto_shape, PY_TO_DTYPE
import numpy as np
import pandas as pd


def numpy_data_to_tensor_proto(data, dtype, shape):
    proto_dtype = NP_TO_HS_DTYPE[dtype]
    kwargs = {
        DTYPE_TO_FIELDNAME[proto_dtype]: data.flatten(),
        "dtype": proto_dtype,
        "shape": np2proto_shape(shape)
    }
    return TensorProto(**kwargs)

def python_dtype_to_proto(value, key, signature) -> TensorProto:
    return numpy_data_to_tensor_proto(value, signature.inputs[key].dtype, signature.inputs[key].shape)


def convert_inputs_to_tensor_proto(x, signature) -> tuple:
    """

    :param x:
    :param signature:
    :return:
    """
    return_type = None
    tensors = {}
    if isinstance(x, dict):
        return_type = dict
        for key, value in x.items():
            if type(value) in PY_TO_DTYPE:  # x: 1
                tensors[key] = numpy_data_to_tensor_proto(value, signature.inputs[key].dtype, signature.inputs[key].shape) # получить данные о k,v
            elif isinstance(value, list):  # x: [1,2,3,4]
                for list_el in value:
                    tensors[key] = numpy_data_to_tensor_proto(list_el, signature.inputs[key].dtype, signature.inputs[key].shape) # получить данные о k,v
            elif isinstance(value, np.ndarray): # x: np.ndarray(1,2,3,4)
                return_type = np.ndarray
                tensors[key] = numpy_data_to_tensor_proto(value, value.dtype, value.shape) # получить данные о k,v
            else:
                raise TypeError("Unsupported objects in dict values {}".format(type(value)))
    elif isinstance(x, pd.DataFrame):
        return_type = pd.DataFrame
        for key, value in dict(x).items():
            tensors[key] = numpy_data_to_tensor_proto(value, value.type, value.shape)
    elif isinstance(x, pd.Series):
        return_type = pd.Series
        x_as_df = x.to_frame()
        for key, value in dict(x_as_df).items():
            tensors[key] = numpy_data_to_tensor_proto(value, value.type, value.shape)
    else:
        raise ValueError(
            "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
                type(x)))
    return tensors, return_type












# TODO: delete if not needed
#
# import logging
#
# from hydro_serving_grpc.gateway import ServablePredictRequest
#
# from hydrosdk.contract import ContractViolationException, Tensor
#
# logger = logging.getLogger("conversions")
#
#
# def decompose_arg_to_tensors(x):
#     tensors = []
#     if type(x) is dict:
#         for k, v in x.items():
#             tensors.append(Tensor(k, v))
#     else:
#         try:
#             import numpy as np
#             import pandas as pd
#         except ImportError as e:
#             logger.exception("Passed pandas or numpy object but encountered error while importing packages")
#             raise e
#         if type(x) is pd.DataFrame:
#             for k, v in dict(x).items():
#                 tensors.append(Tensor(k, v))
#         elif type(x) is pd.Series:
#             if x.name is None:
#                 raise ValueError("Provided pandas.Series should have names")
#             else:
#                 tensors.append(Tensor(x.name, np.array(x)))
#         elif type(x) is np.ndarray:
#             raise NotImplementedError("Conversion of nameless np.array is not supported")
#         else:
#             raise ValueError(
#                 "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
#                     type(x)))
#         return tensors
#
#
# def decompose_kwarg_to_tensor(key, x):
#     if type(x) is dict:
#         raise NotImplementedError("Conversion of dict as kwarg is not supported")
#     else:
#         try:
#             import numpy as np
#             import pandas as pd
#         except ImportError as e:
#             logger.exception("Passed pandas or numpy object but encountered error while importing packages")
#             raise e
#         if type(x) is pd.DataFrame:
#             tensor = Tensor(key, np.array(x))
#         elif type(x) is pd.Series:
#             tensor = Tensor(key, np.array(x))
#         elif type(x) is np.ndarray:
#             tensor = Tensor(key, x)
#         elif np.isscalar(x):
#             if x in (0, 1):
#                 # Minimum scalar dtype for 0 or 1 is `uint8`, but it
#                 # cannot be casted into `bool` safely. So, we detect
#                 # for bool scalars by hand.
#                 min_input_dtype = np.bool
#             else:
#                 min_input_dtype = np.min_scalar_type(x)
#
#             tensor = Tensor(key, np.array(x, dtype=min_input_dtype))
#         else:
#             raise ValueError(
#                 "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
#                     type(x)))
#         return tensor
#

