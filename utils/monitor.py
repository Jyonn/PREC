import json
import os

import torch

from utils.time_printer import printer


class Monitor:
    def __init__(
            self,
            save_dir=None,
            top=None,
            epoch_skip=None,
            early_stop=None,
            debug=False,
    ):
        self.candidates = []
        self.save_dir = save_dir
        self.top = top or 1
        self.epoch_skip = epoch_skip
        self.early_stop = early_stop
        self.debug = debug
        self.print = printer

    def remove_checkpoint(self, epoch):
        if self.debug:
            print(f'remove {epoch}')
            return
        epoch_path = os.path.join(self.save_dir, 'epoch_{}.bin'.format(epoch))
        if os.path.exists(epoch_path):
            os.system(f'rm {epoch_path}')

    def store_checkpoint(self, epoch, state_dict):
        if self.debug:
            print(f'store {epoch}')
            return
        epoch_path = os.path.join(self.save_dir, 'epoch_{}.bin'.format(epoch))
        torch.save(state_dict, epoch_path)
        self.step_export()

    def push(self, epoch, loss: float, state_dict):
        # print(epoch)
        if self.epoch_skip and epoch < self.epoch_skip:
            return

        self.candidates.append((epoch, loss))

        stay = [True] * len(self.candidates)

        for ia in range(len(self.candidates)):
            for ib in range(len(self.candidates)):
                if ia == ib or not stay[ia] or not stay[ib]:
                    continue
                if self.candidates[ia][1] < self.candidates[ib][1]:
                    stay[ib] = False

        remove = []
        for i in range(len(self.candidates)):
            if not stay[i]:
                remove.append((i, self.candidates[i][0]))

        top_remove = self.top - sum(stay)
        if top_remove > 0:
            for checkpoint in remove[-top_remove:]:
                stay[checkpoint[0]] = True
            remove = remove[:-top_remove]
        for checkpoint in remove:
            self.remove_checkpoint(checkpoint[1])

        self.candidates = [self.candidates[i] for i in range(len(self.candidates)) if stay[i]]

        if not stay[-1]:
            if self.early_stop:
                for i in range(len(self.candidates))[::-1]:
                    if stay[i]:
                        if epoch - self.candidates[i][0] >= self.early_stop:
                            self.print('Early Stop')
                            exit(0)
                        return
            return

        self.store_checkpoint(epoch, state_dict)

    def step_export(self):
        candidates = list(map(lambda x: x[0], self.candidates))
        export_path = os.path.join(self.save_dir, 'candidates.json')
        json.dump(candidates, open(export_path, 'w'))

    def export(self):
        if self.top:
            for candidate in self.candidates[:-self.top]:
                self.remove_checkpoint(candidate[0])
            self.candidates = self.candidates[-self.top:]
        self.step_export()


if __name__ == '__main__':
    m = Monitor(
        save_dir=None,
        top=5,
        epoch_skip=0,
        early_stop=None,
        debug=True
    )
    losses = [1.3518, 1.2661, 1.2446, 1.2297, 1.2367, 1.2472, 1.1911, 1.1674]
    for index, loss in enumerate(losses):
        m.push(index, loss, None)
