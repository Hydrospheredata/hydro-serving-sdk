import time

from hydro_serving_grpc.tf import TensorProto, TensorShapeProto, DT_DOUBLE
from hydro_serving_grpc.tf.api.model_pb2 import ModelSpec

from hydrosdk.predictor import PredictServiceClient
from tests.test_application import create_test_application
from tests.test_model import get_cluster


# TODO: Add more valid assert
from tests.test_servable import create_test_servable


def test_predict():
    grpc_cluster = get_cluster("0.0.0.0:9090")

    created_servable = create_test_servable()

    model_spec = ModelSpec(name=created_servable.name)

    # wait for servable to assemble
    time.sleep(10)

    dtype = DT_DOUBLE

    dim_one = TensorShapeProto.Dim(size=-1)
    dim_two = TensorShapeProto.Dim(size=2)
    tensor_shape = TensorShapeProto(dim=[dim_one, dim_two])
    tensor = TensorProto(dtype=dtype, tensor_shape=tensor_shape)
    to_pass = {'in1': tensor}

    psc = PredictServiceClient(cluster=grpc_cluster)
    result = psc.predict(inputs=to_pass, model_spec=model_spec)

    assert result


def test_servable_preditor_create():
    created_servable = create_test_servable()
    assert isinstance(created_servable.predictor(monitoring=True, ssl=True), PredictServiceClient)


def test_application_predictor_create():
    cluster = get_cluster()

    created_application = create_test_application(cluster=cluster)
    assert isinstance(created_application.predictor(monitoring=True, ssl=True), PredictServiceClient)
