class Formatter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, s: str):
        return s.format(**self.kwargs)
