import pytest
import numpy as np

from hydrosdk.signature import SignatureBuilder, mock_input_data

SUPPORTED_NP_TYPES = [np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
                      np.float, np.float64, np.float32, np.float16,
                      np.complex, np.complex128, np.complex64,
                      np.uint, np.uint64, np.uint32, np.uint16, np.uint8, np.uint0,
                      np.bool,
                      np.str, np.unicode]

UNSUPPORTED_NP_TYPES = [np.float128, np.complex256]


@pytest.mark.parametrize('np_type', SUPPORTED_NP_TYPES)
def test_tensor_data(np_type):
    s = SignatureBuilder("change_state") \
        .with_input("tensor1", np_type, "scalar") \
        .with_input("tensor2", np_type, [-1, 10]) \
        .build()

    mock_data = mock_input_data(s)
    assert mock_data is not None


@pytest.mark.xfail(strict=True, raises=TypeError)
@pytest.mark.parametrize('np_type', UNSUPPORTED_NP_TYPES)
def test_unsupported(np_type):
    s = SignatureBuilder("change_state") \
        .with_input("tensor1", np_type, "scalar") \
        .with_input("tensor2", np_type, [-1, 10]) \
        .build()
    mock_data = mock_input_data(s)
    assert mock_data is not None
