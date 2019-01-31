import torch


def graph(session):
    dummy_input = torch.zeros(1, 112, 8, 8).cuda()
    session.train_writer.add_graph(session.net.module, dummy_input)
