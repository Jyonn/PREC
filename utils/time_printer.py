import time
from logging import warning


class TimePrinter:
    def __init__(self):
        self.start_time = time.time()
        # self('start time printer')

    @staticmethod
    def div_num(n, base=60):
        return n // base, n % base

    def format_second(self, second):
        second = int(second)
        minutes, second = self.div_num(second)
        hours, minutes = self.div_num(minutes)
        return '[%02d:%02d:%02d]' % (hours, minutes, second)

    def __call__(self, *args):
        delta_time = time.time() - self.start_time
        print(self.format_second(delta_time), *args)

    def with_warn(self, string):
        delta_time = time.time() - self.start_time
        warning('%s %s' % (self.format_second(delta_time), string))


printer = TimePrinter()
