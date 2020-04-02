import time

from hydro_serving_grpc.contract import ModelContract
from hydro_serving_grpc.tf import TensorProto, TensorShapeProto, DT_DOUBLE
from hydro_serving_grpc.tf.api.model_pb2 import ModelSpec

from hydrosdk.predictor import PredictServiceClient
from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model, get_signature


# TODO: Add more valid assert
def test_predict():
    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()
    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster)

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
