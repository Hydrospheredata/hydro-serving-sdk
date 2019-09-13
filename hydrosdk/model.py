from hydro_serving_grpc.manager import ModelVersion, DockerImage as hs_grpc_Docker_Image

from hydrosdk.contract import Contract
from hydrosdk.image import DockerImage
from hydrosdk.servable import Servable


def require_cluster(func, *args, **kwargs):
    def check_for_cluster(model, *args, **kwargs):
        if model.cluster is None:
            raise ValueError("No Cluster provided")
        return func(model, *args, **kwargs)

    return check_for_cluster


class Model:
    @staticmethod
    def find(cluster, name=None, version=None, id=None):
        pass

    @staticmethod
    def create(path, name, runtime, contract=None, install_command=None):
        if contract:
            contract.validate()
        else:
            pass  # infer the contract
        pass

    @classmethod
    def from_proto(cls, proto: ModelVersion):
        cls(_id=proto.id,
            name=proto.name,
            version=proto.version,
            contract=Contract.from_proto(proto.contract),
            runtime=proto.runtime,
            image=DockerImage(name=proto.image.name, tag=proto.image.tag, sha256=proto.image_sha))

    @property
    def proto(self):
        return ModelVersion(_id=self.id,
                            name=self.name,
                            version=self.version,
                            contract=self.contract.proto(),
                            runtime=self.runtime,
                            image=hs_grpc_Docker_Image(name=self.image.name, tag=self.image.tag),
                            image_sha=self.image.sha256
                            )

    def __init__(self, _id, name, version, contract, runtime, image, cluster=None):
        self.cluster = cluster
        self.id = _id
        self.name = name
        self.version = version
        self.runtime = runtime
        self.image = image
        self.contract = contract

    def with_cluster(self, cluster):
        self.cluster = cluster
        return self

    def __repr__(self):
        return "Model \"{}\" v{}".format(self.name, self.version)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Prediction through Model object is not possible. Use servable instead")

    @require_cluster
    def create_servable(self) -> Servable:
        """
        Alias for 'HydroServingClient.create_servable' method
        :return: new servable of this model
        """
        return self.cluster.create_servable(self.name, self.version)

    @require_cluster
    def list_servables(self):
        return self.cluster.servables({"model_name": self.name, "model_version": self.version, "access": "user"})
