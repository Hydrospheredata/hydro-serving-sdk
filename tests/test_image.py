import pytest
from hydrosdk.image import DockerImage

@pytest.mark.parametrize("image", ["test:latest", "test:1", "www.registry.io/test:latest"])
def test_image_parsing(image):
    parsed = DockerImage.from_string(image)
    print(parsed)
    assert parsed.name is not None
    assert parsed.tag is not None
