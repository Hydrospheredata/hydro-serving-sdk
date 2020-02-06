import numpy as np
import pytest

import hydrosdk as hs
from hydrosdk.contract import SignatureBuilder, mock_input_data

SUPPORTED_NP_TYPES = [np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
                      np.float, np.float64, np.float32, np.float16,
                      np.complex, np.complex128, np.complex64,
                      np.uint, np.uint64, np.uint32, np.uint16, np.uint8, np.uint0,
                      np.bool,
                      np.str, np.unicode]

UNSUPPORTED_NP_TYPES = [np.float128, np.complex256]


def test_tensor_data():
    for t in SUPPORTED_NP_TYPES:
        dtype = np.dtype(t)
        print("Mocking {} dtype".format(dtype))
        s = SignatureBuilder("change_state") \
            .with_input("tensor1", t, hs.scalar) \
            .with_input("tensor2", t, [-1, 10]) \
            .build()

        mock_data = mock_input_data(s)
        assert mock_data is not None
        # is_valid, error_msg = s.validate_input(mock_data)
        # self.assertTrue(is_valid)
        # self.assertIsNone(error_msg)


def test_unsupported():
    for t in UNSUPPORTED_NP_TYPES:
        dtype = np.dtype(t)
        print("Trying to mock {} dtype".format(dtype))
        with pytest.raises(KeyError):
            s = SignatureBuilder("change_state") \
                .with_input("tensor1", t, hs.scalar) \
                .with_input("tensor2", t, [-1, 10]) \
                .build()
            mock_input_data(s)