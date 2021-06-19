from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def evaluate_metrics(output, target):
    tn, fp, fn, tp = confusion_matrix(target, output).ravel()
    return tp / (tp + 1 / 2 * (fp + fn))


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
