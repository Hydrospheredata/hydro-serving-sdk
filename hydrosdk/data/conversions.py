import logging

from hydro_serving_grpc.gateway import ServablePredictRequest

from hydrosdk.contract import ContractViolationException, Tensor

logger = logging.getLogger("conversions")


def decompose_arg_to_tensors(x):
    tensors = []
    if type(x) is dict:
        for k, v in x.items():
            tensors.append(Tensor(k, v))
    else:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as e:
            logger.exception("Passed pandas or numpy object but encountered error while importing packages")
            raise e
        if type(x) is pd.DataFrame:
            for k, v in dict(x).items():
                tensors.append(Tensor(k, v))
        elif type(x) is pd.Series:
            if x.name is None:
                raise ValueError("Provided pandas.Series should have names")
            else:
                tensors.append(Tensor(x.name, np.array(x)))
        elif type(x) is np.ndarray:
            raise NotImplementedError("Conversion of nameless np.array is not supported")
        else:
            raise ValueError(
                "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
                    type(x)))
        return tensors


def decompose_kwarg_to_tensor(key, x):
    if type(x) is dict:
        raise NotImplementedError("Conversion of dict as kwarg is not supported")
    else:
        try:
            import numpy as np
            import pandas as pd
        except ImportError as e:
            logger.exception("Passed pandas or numpy object but encountered error while importing packages")
            raise e
        if type(x) is pd.DataFrame:
            tensor = Tensor(key, np.array(x))
        elif type(x) is pd.Series:
            tensor = Tensor(key, np.array(x))
        elif type(x) is np.ndarray:
            tensor = Tensor(key, x)
        elif np.isscalar(x):
            if x in (0, 1):
                # Minimum scalar dtype for 0 or 1 is `uint8`, but it
                # cannot be casted into `bool` safely. So, we detect
                # for bool scalars by hand.
                min_input_dtype = np.bool
            else:
                min_input_dtype = np.min_scalar_type(x)

            tensor = Tensor(key, np.array(x, dtype=min_input_dtype))
        else:
            raise ValueError(
                "Conversion failed. Expected [pandas.DataFrame, pd.Series, dict[str, numpy.ndarray]], got {}".format(
                    type(x)))
        return tensor
