from hydro_serving_grpc.contract import ModelContract, ModelSignature, ModelField

class Contract:
    @staticmethod    
    def from_proto(proto_obj):
        pass

    @staticmethod
    def from_fields(sig_name, inputs, outputs):
        pass

    def __init__(self, signature):
        pass

    def validate(self):
        pass

    def merge_sequential(self, other_contract):
        pass
    
    def merge_parallel(self, other_contract):
        pass

    def mock_data(self):
        pass

    def to_proto(self):
        pass
