import os.path


class Processor:
    def __init__(self, data_dir: str, store_dir):
        self.data_dir = os.path.expanduser(data_dir)
        self.store_dir = os.path.expanduser(store_dir)
        os.makedirs(self.store_dir, exist_ok=True)

    def tokenize(self):
        raise NotImplementedError
