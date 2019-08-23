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
    
    def __init__(self, cluster, id, name, version, contract, install_command, runtime, status, status_message, image):
        self.cluster = cluster
        self.id = id
        self.name = name
        self.version = version
        self.contract = contract
        self.install_command = install_command
        self.runtime = runtime
        self.image = image