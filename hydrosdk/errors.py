class InvalidYAMLFile(RuntimeError):
    def __init__(self, file, msg):
        full_msg = "Error while parsing {} file: {}".format(file, msg)
        super().__init__(full_msg)
        self.file = file
        self.msg = msg
