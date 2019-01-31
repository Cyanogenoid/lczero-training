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
    skewness = metrics.policy_skewness(session.net)
    session.train_writer.add_scalar('metrics/policy_weight_skewness', skewness, global_step=session.step)
