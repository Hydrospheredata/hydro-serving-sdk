from io import BytesIO

import pytest

from hydrosdk.cluster import Cluster
from hydrosdk.modelversion import ModelVersion, LocalModel
from hydrosdk.utils import read_in_chunks
from tests.common_fixtures import *
from tests.config import *


@pytest.fixture(scope="module")
def scalar_training_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'resources/scalar_identity_model/training-data.csv')


def test_read_in_chunks(scalar_training_data: str):
    buffer = BytesIO()
    gen = read_in_chunks(scalar_training_data, 1)
    for chunk in gen:
        buffer.write(chunk)
    buffer.seek(0)
    destination_content = buffer.read()
    buffer.close()
    with open(scalar_training_data, "rb") as file:
        origin_content = file.read()
    assert origin_content == destination_content


def test_training_data_upload(cluster: Cluster, scalar_local_model: LocalModel, 
                              scalar_training_data: str):
    mv: ModelVersion = scalar_local_model.upload(cluster)
    mv.training_data = scalar_training_data
    data_upload_response = mv.upload_training_data()
    data_upload_response.wait(sleep=5)
    resp = cluster.request("GET", f"/monitoring/profiles/batch/{mv.id}/status")
    assert "Success" == resp.json()["kind"]
    
