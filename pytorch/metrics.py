import torch
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


def policy_value_gradient_ratio(session, loss_components):
    policy_loss, value_z_loss, value_q_loss, reg_loss = loss_components
    value_loss = 0.5 * (value_z_loss + value_q_loss)

    # compute policy gradient
    policy_loss.backward(retain_graph=True)
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    policy_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # compute value gradient
    value_loss.backward(retain_graph=True)
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    value_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # ratio of the two
    policy_value_ratio = policy_grad_norm / (policy_grad_norm + value_grad_norm)
    return policy_value_ratio


def zq_gradient_ratio(session, loss_components):
    policy_loss, value_z_loss, value_q_loss, reg_loss = loss_components

    # compute policy gradient
    value_z_loss.backward(retain_graph=True)
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    z_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # compute value gradient
    value_q_loss.backward()
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    q_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # ratio of the two
    zq_ratio = z_grad_norm / (z_grad_norm + q_grad_norm)
    return zq_ratio


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
