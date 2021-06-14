from sklearn.metrics import f1_score


def evaluate_metrics(output, target):
    return f1_score(output, target)

