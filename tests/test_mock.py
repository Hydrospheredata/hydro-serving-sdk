# import numpy as np
#
# from hydrosdk.contract import SignatureBuilder, mock_input_data
# from hydrosdk.data.proto_conversion_utils import np2proto_dtype
#
# SUPPORTED_MOCK_DTYPES = [np.int, np.int64, np.int32, np.int16, np.int8, np.int0,
#                          np.float, np.float128, np.float64, np.float32, np.float16,
#                          np.complex, np.complex256, np.complex128, np.complex64,
#                          np.uint, np.uint64, np.uint32, np.uint16, np.uint8, np.uint0,
#                          np.bool,
#                          np.str, np.unicode]
#
#
# def test_scalar_shape():
#     for t in SUPPORTED_MOCK_DTYPES:
#         dtype = np.dtype(t)
#         print("Mocking {} dtype".format(dtype))
#         s = SignatureBuilder("change_state") \
#             .with_input("kek1", t, "scalar") \
#             .with_input("kek2", t, "scalar") \
#             .build()
#         mock_data = mock_input_data(s)
#         print(mock_data)
#         assert mock_data is not None
#         # is_valid, error_msg = s.validate_input(mock_data)
#         # self.assertTrue(is_valid)
#         # self.assertIsNone(error_msg)
#
#
# def test_columnar_data():
#     for t in SUPPORTED_MOCK_DTYPES:
#         dtype = np.dtype(t)
#         print("Mocking {} dtype".format(dtype))
#         s = SignatureBuilder("change_state") \
#             .with_input("kek1", np2proto_dtype(dtype), [-1, 1]) \
#             .with_input("kek2", np2proto_dtype(dtype), [-1, 1]) \
#             .build()
#
#         mock_data = mock_input_data(s)
#         print(mock_data)
#         assert mock_data is not None
#         # is_valid, error_msg = s.validate_input(mock_data)
#         # self.assertTrue(is_valid)
#         # self.assertIsNone(error_msg)
#
#
# def test_tensor_data():
#     for t in SUPPORTED_MOCK_DTYPES:
#         dtype = np.dtype(t)
#         print("Mocking {} dtype".format(dtype))
#         s = SignatureBuilder("change_state") \
#             .with_input("kek1", np2proto_dtype(dtype), "scalar") \
#             .with_input("kek2", np2proto_dtype(dtype), [-1, 10]) \
#             .build()
#
#         mock_data = mock_input_data(s)
#         print(mock_data)
#         assert mock_data is not None
#         # is_valid, error_msg = s.validate_input(mock_data)
#         # self.assertTrue(is_valid)
#         # self.assertIsNone(error_msg)
