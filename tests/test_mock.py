import unittest

import numpy as np

from hydrosdk.contract import Signature, Field

SUPPORTED_MOCK_DTYPES = [np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
                         np.float, np.float128, np.float64, np.float32, np.float16,
                         np.complex, np.complex256, np.complex128, np.complex64,
                         np.uint, np.uint64, np.uint32, np.uint16, np.uint8, np.uint0,
                         np.bool,
                         np.str, np.unicode]


class TestMockDataGeneration(unittest.TestCase):

    def test_scalar_shape(self):
        for t in SUPPORTED_MOCK_DTYPES:
            with self.subTest("mock {} dtype".format(t)):
                s = Signature(name="change_state",
                              inputs=[Field("kek1", tuple(), np.dtype(t)), Field("kek2", tuple(), np.dtype(t))],
                              outputs=[])

                mock_data = s.mock_input_data()
                is_valid, error_msg = s.validate_input(mock_data)
                self.assertTrue(is_valid)
                self.assertIsNone(error_msg)

    def test_columnar_data(self):
        for t in SUPPORTED_MOCK_DTYPES:
            with self.subTest("mock {} dtype".format(t)):
                s = Signature(name="change_state",
                              inputs=[Field("kek1", (-1, 1), np.dtype(t)), Field("kek2", (-1, 1), np.dtype(t))],
                              outputs=[])

                mock_data = s.mock_input_data()
                is_valid, error_msg = s.validate_input(mock_data)
                self.assertTrue(is_valid)
                self.assertIsNone(error_msg)

    def test_tensor_data(self):
        for t in SUPPORTED_MOCK_DTYPES:
            with self.subTest("mock {} dtype".format(t)):
                s = Signature(name="change_state",
                              inputs=[Field("kek1", tuple(), np.dtype(t)), Field("kek2", (-1, 10), np.dtype(t))],
                              outputs=[])

                mock_data = s.mock_input_data()
                is_valid, error_msg = s.validate_input(mock_data)
                self.assertTrue(is_valid)
                self.assertIsNone(error_msg)


if __name__ == '__main__':
    unittest.main()
