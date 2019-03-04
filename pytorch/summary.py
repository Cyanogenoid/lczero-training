import torch
import copy
import metrics
import collections

import utils


def model_graph(session):
    dummy_input = torch.zeros(1, 112, 8, 8).cuda()
    session.train_writer.add_graph(session.net.module, dummy_input)


def weight_histograms(session):
    for name, param in utils.named_variables(session.net.module):
        if 'num_batches_tracked' in name:
            continue
        session.train_writer.add_histogram(f'weight/{name}', param.detach().cpu().numpy(), session.step)


def policy_weight_skewness(session):
    skewness = metrics.policy_weight_skewness(session.net)
    session.test_writer.add_scalar('metrics/policy_weight_skewness', skewness, global_step=session.step)


@torch.enable_grad()
def policy_value_gradient_ratio(session, batch):
    # don't do this for swa nets, since we don't compute gradients for them
    if session.swa.active:
        return
    # store session cfg since we want to modify it in here
    old_cfg = session.cfg
    session.cfg = copy.deepcopy(session.cfg)
    # store old metrics since we don't want what we're doing here to affect them
    old_metrics = session.metrics
    session.metrics = collections.defaultdict(list)

    # compute policy gradient
    session.cfg['training']['policy_weight'] = 1.0
    session.cfg['training']['value_weight'] = 0.0
    session.cfg['training']['reg_weight'] = 0.0
    session.forward(batch).backward()
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    policy_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # compute value gradient
    session.cfg['training']['policy_weight'] = 0.0
    session.cfg['training']['value_weight'] = 1.0
    session.cfg['training']['reg_weight'] = 0.0
    session.forward(batch).backward()
    grads = torch.nn.utils.parameters_to_vector(p.grad for p in session.net.parameters())
    value_grad_norm = grads.norm(p=2).item()
    session.optimizer.zero_grad()

    # ratio of the two
    policy_value_ratio = policy_grad_norm / (policy_grad_norm + value_grad_norm)
    session.test_writer.add_scalar('metrics/policy_value_gradient_ratio', policy_value_ratio, global_step=session.step)

    session.cfg = old_cfg
    session.metrics = old_metrics
