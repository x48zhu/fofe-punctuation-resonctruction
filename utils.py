from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LearningCurvePlotter(object):
    def __init__(self, *args):
        self.num_criteria = len(args)
        self.num_epoch = 0
        self.data = {}
        for criteria in args:
            self.data[criteria] = []

    def update(self, *kwargs):
        assert self.num_criteria == len(kwargs)
        for key in kwargs:
            if key not in self.data:
                raise KeyError
            self.data[key].append(kwargs[key])
        self.num_epoch += 1

    def plot(self, save_path):
        plt.interactive(False)
        plt.figure(1)
        for i, key in enumerate(self.data):
            position = self.num_criteria * 100 + 10 + i
            plt.subplot(position)
            plt.plot(range(self.num_epoch), self.data[key], '--')
        plt.savefig("%s.png" % save_path)