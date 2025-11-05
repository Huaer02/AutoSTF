from collections import namedtuple

# import ruamel.yaml as yaml
import yaml


# from ruamel.yaml import YAML
# yaml = YAML()
# from ruamel.yaml import RoundTripLoader
def dict_to_namedtuple(dic: dict):
    return namedtuple("tuple", dic.keys())(**dic)


class Config:
    def __init__(self):
        pass

    def load_config(self, config):
        with open(config, "r") as f:
            # setting = yaml.load(f, Loader=RoundTripLoader)
            setting = yaml.load(f)
        self.data = dict_to_namedtuple(setting["data"])
        self.model = dict_to_namedtuple(setting["model"])
        self.trainer = dict_to_namedtuple(setting["trainer"])
