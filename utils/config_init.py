import os

from oba import Obj
from refconfig import RefConfig
from smartdict import DictCompiler


class PathSearcher(DictCompiler):
    compiler = DictCompiler({})

    @classmethod
    def search(cls, d: dict, path: str):
        cls.compiler.d = d
        cls.compiler.circle = {}
        return cls.compiler._get_value(path)


class ConfigInit:
    def __init__(self, makedirs):
        # required_args = ['data', 'model', 'exp']
        self.makedirs = makedirs

    def parse(self, args):
        config = RefConfig().add_yaml(
            config=args.config,
            exp=args.exp,
        ).parse()

        for makedir in self.makedirs:
            try:
                dir_name = PathSearcher.search(config, makedir)
                os.makedirs(dir_name, exist_ok=True)
            except:
                pass

        config = Obj(config)
        return config
        # data, model, exp = config.data, config.model, config.exp
        # return config, data, model, exp
