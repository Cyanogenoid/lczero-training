import numpy as np


class MetricsManager():
    def __init__(self, weighted=False):
        self.data = {}
        self.weighted = weighted

    def update(self, key, value):
        self.data.setdefault(key, []).append(value)

    def __getitem__(self, key):
        if not self.weighted:
            weights = None
        else:
            num_entries = len(self.data[key])
            weights = np.arange(1, num_entries + 1)
        return np.average(self.data[key], weights=weights)

    def reset(self, key):
        del self.data[key]

    def reset_all(self):
        for key in list(self.data.keys()):
            self.reset(key)

    def keys(self):
        return self.data.keys()


def log_session(session, writer):
    # TODO better organisation of parent tags
    writer.add_scalar('loss/policy', session.metrics['policy_loss'], global_step=session.step)
    writer.add_scalar('loss/value', session.metrics['value_loss'], global_step=session.step)
    if writer == session.train_writer:
        writer.add_scalar('loss/weight', session.metrics['reg_loss'], global_step=session.step)
    writer.add_scalar('loss/total', session.metrics['total_loss'], global_step=session.step)
    writer.add_scalar('metrics/policy_accuracy', session.metrics['policy_accuracy'], global_step=session.step)
    writer.add_scalar('metrics/value_accuracy', session.metrics['value_accuracy'], global_step=session.step)
    if writer == session.train_writer:  # target data comes from same distribution, so no point plotting it again
        writer.add_scalar('metrics/policy_target_entropy', session.metrics['policy_target_entropy'], global_step=session.step)
    if writer == session.train_writer:
        writer.add_scalar('metrics/gradient_norm', session.metrics['gradient_norm'], global_step=session.step)
        writer.add_scalar('hyperparameter/learning_rate', session.lr_scheduler.get_lr()[0], global_step=session.step)


def accuracy(predicted, vector=None, index=None):
    assert vector is not None or index is not None
    # predict class with highest probability
    predicted = predicted.max(dim=1)[1]
    if vector is not None:
        target = vector.max(dim=1)[1]
    else:
        target = index
    return (predicted == target).float().mean()


def entropy(distribution):
    per_element = -distribution * distribution.clamp(min=1e-20).log()
    return per_element.mean(dim=0).sum()


def skewness(weight):
    # ttps://en.wikipedia.org/wiki/Skewness#Pearson's_moment_coefficient_of_skewness
    weight_mean = weight.mean()
    moment = weight - weight_mean
    kappa_2 = moment.pow(2).mean()
    kappa_3 = moment.pow(3).mean()
    return kappa_3 / kappa_2.pow(1.5)
