import yaml
from yaml.parser import ParserError


def yaml_file(file):
    try:
        yaml_dict = yaml.safe_load(file)
        return yaml_dict
    except ParserError as ex:
        raise ParserError(file, ex)
    except KeyError as ex:
        raise ParserError(file, "Can't find {} field".format(ex))