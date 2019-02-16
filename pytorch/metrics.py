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


def policy_weight_skewness(net):
    # ttps://en.wikipedia.org/wiki/Skewness#Pearson's_moment_coefficient_of_skewness
    policy_weight = net.module.policy_head.lin.weight
    policy_weight_mean = policy_weight.mean()
    moment = policy_weight - policy_weight_mean
    kappa_2 = moment.pow(2).mean()
    kappa_3 = moment.pow(3).mean()
    return kappa_3 / kappa_2.pow(1.5)
