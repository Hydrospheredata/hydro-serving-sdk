from pydantic import BaseModel
from hydrosdk.image import DockerImage

class Configuration(BaseModel):
    class HttpConfig(BaseModel):
        url: str
    class GrpcConfig(BaseModel):
        url: str
        ssl: bool
    
    runtime: DockerImage
    http_cluster: HttpConfig
    grpc_cluster: GrpcConfig
    lock_timeout: int
    default_application_name: str
    default_model_name: str

config = Configuration.parse_obj({
    "runtime": 
        {
            "name": "hydrosphere/serving-runtime-python-3.7",
            "tag": "3.0.0-dev4"
        },
    "http_cluster":
        {  
            "url": "http://localhost",
        },
    "grpc_cluster":
        {
            "url": "localhost:9090",
            "ssl": False
        },
    "lock_timeout": 120,
    "default_application_name": "infer",
    "default_model_name": "infer"
})