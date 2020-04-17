import hydro_serving_grpc as hs_grpc


def infer(input):
    output_tensor_proto = hs_grpc.TensorProto(
        int64_val=input.int64_val,
        dtype=hs_grpc.DT_INT64,
        tensor_shape=hs_grpc.TensorShapeProto())

    return hs_grpc.PredictResponse(outputs={"output": output_tensor_proto})
