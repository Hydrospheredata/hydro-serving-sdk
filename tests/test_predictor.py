import time

import numpy as np
from hydro_serving_grpc.contract import ModelContract
from pandas import DataFrame

from hydrosdk.servable import Servable
from tests.test_model import get_cluster, get_local_model, get_signature


# TODO: Add more valid assert
def test_predict_list():
    # test sending values in list [-1,2]

    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()

    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster,
                                       grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)

    predictor_client = created_servable.predictor()

    inputs = {'in1': [-1, 2]}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions['in1'], list)
    assert isinstance(predictions, dict)


def test_predict_pythontype():
    # test sending values in python type -1

    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()

    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster,
                                       grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)

    predictor_client = created_servable.predictor()
    inputs = {'in1': -1}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions, dict)


def test_predict_nparray():
    # test sending values in np.ndarray(1,2,3,4)

    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()

    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster,
                                       grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)

    predictor_client = created_servable.predictor()
    inputs = {'in1': np.array([1])}
    predictions = predictor_client.predict(inputs=inputs)

    assert isinstance(predictions['in1'], np.array)
    assert isinstance(predictions, dict)


def test_predict_df():
    # test sending values in df({"in1":[1,1]})

    grpc_cluster = get_cluster("0.0.0.0:9090")
    http_cluster = get_cluster()

    signature = get_signature()

    contract = ModelContract(predict=signature)

    model = get_local_model(contract=contract)

    upload_resp = model.upload(http_cluster)

    # wait for model to upload
    time.sleep(10)

    created_servable = Servable.create(model_name=upload_resp[model].model.name,
                                       model_version=upload_resp[model].model.version, cluster=http_cluster,
                                       grpc_cluster=grpc_cluster)

    # wait for servable to assemble
    time.sleep(20)

    predictor_client = created_servable.predictor()
    inputs_dict = {'in1': [1]}
    inputs_df = DataFrame(inputs_dict)
    predictions = predictor_client.predict(inputs=inputs_df)

    assert isinstance(predictions, DataFrame)

# TODO: do we need series?
# def test_predict_series():
#     # test sending values in df({"in1":[1,1]})
#
#     grpc_cluster = get_cluster("0.0.0.0:9090")
#     http_cluster = get_cluster()
#
#     signature = get_signature()
#
#     contract = ModelContract(predict=signature)
#
#     model = get_local_model(contract=contract)
#
#     upload_resp = model.upload(http_cluster)
#
#     # wait for model to upload
#     time.sleep(10)
#
#     created_servable = Servable.create(model_name=upload_resp[model].model.name,
#                                        model_version=upload_resp[model].model.version, cluster=http_cluster,
#                                        grpc_cluster=grpc_cluster)
#
#     # wait for servable to assemble
#     time.sleep(20)
#
#     predictor_client = created_servable.predictor()
#     inputs_dict = {'in1': [1]}
#     inputs_df = Series(inputs_dict)
#     predictions = predictor_client.predict(inputs=inputs_df)
#
#     assert isinstance(predictions, DataFrame)
