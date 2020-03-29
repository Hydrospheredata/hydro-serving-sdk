class InvalidYAMLFile(RuntimeError):
    """
    Error raised when working with invalid yaml file
    """
    def __init__(self, file, msg):
        full_msg = "Error while parsing {} file: {}".format(file, msg)
        super().__init__(full_msg)
        self.file = file
        self.msg = msg
