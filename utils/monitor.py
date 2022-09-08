from utils.metric import Metric


class Monitor:
    def __init__(self, monitor):
        self.key = monitor.metric
        self.compare = monitor.compare
        self.best_value = None
        self.patience = monitor.patience
        self.worse_time = 0

    def test(self, metric: Metric):
        value = metric[self.key]
        if self.best_value is None:
            self.best_value = value
            return True
        if (self.best_value > metric[self.key]) ^ (self.compare == 'max'):
            self.best_value = metric[self.key]
            self.worse_time = 0
            return True
        self.worse_time += 1
        return self.worse_time < self.patience
