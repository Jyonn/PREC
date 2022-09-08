class Logger:
    def __init__(self, log_file, print_fct):
        self.log_file = log_file
        self.print = print_fct or print

    def __call__(self, s):
        self.print(s)
        with open(self.log_file, 'a') as f:
            f.write(str(s) + '\n')
