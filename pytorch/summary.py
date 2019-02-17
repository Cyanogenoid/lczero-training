import copy
import collections

import torch

import metrics
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
    policy_weight = session.net.module.policy_head.lin.weight
    skewness = metrics.skewness(policy_weight)
    session.test_writer.add_scalar('metrics/policy_weight_skewness', skewness, global_step=session.step)


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
        writer.add_scalar('metrics/gradient_norm', session.metrics['gradient_norm'], global_step=session.step)
        writer.add_scalar('metrics/policy_value_gradient_ratio', session.metrics['policy_value_gradient_ratio'], global_step=session.step)
        writer.add_scalar('hyperparameter/learning_rate', session.lr_scheduler.get_lr()[0], global_step=session.step)
