import itertools

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import data
import model


class Session():
    def __init__(self, cfg):
        self.net = model.Net(
            residual_channels=cfg['model']['residual_channels'],
            residual_blocks=cfg['model']['residual_blocks'],
            policy_channels=cfg['model']['policy_channels'],
            se_ratio=cfg['model']['se_ratio'],
        ).cuda()
        self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

        # Construct data loaders
        self.train_loader = data.v3_loader(cfg['data']['train_path'])
        self.test_loader = data.v3_loader(cfg['data']['test_path'])

        # TODO save/resume checkpoint
        # TODO tensorboardX
        # TODO gradient accumulation
        # TODO swa

    def train_loop(self):
        for batch in self.train_loader:
            done = self.train_step(batch)
            # TODO write tensorboard info
            # every n steps run test epoch
            # every m steps checkpoint
            # terminate at k steps

    def train_step(self, batch):
        total_loss = self.forward(batch)

        total_loss.backward()
        if True:  # TODO gradient accumulation condition here
            optimizer.step()
            optimizer.zero_grad()
            return True

    def forward(self, batch):
        ''' Perform one step of either training or evaluation
        '''
        # Move batch to the GPU
        input_planes, policy_target, value_target = batch
        input_planes = input_planes.cuda(async=True)
        policy_target = policy_target.cuda(async=True)
        value_target = value_target.cuda(async=True)

        # Forward batch through the network
        policy, value = net(input_planes)

        # Compute losses
        policy_logits = F.log_softmax(policy)
        policy_loss = F.nll_loss(policy_logits, policy_target)  # this is the same as cross-entropy
        value_loss = F.mse_loss(value, value_target)
        flat_weights = torch.cat([w.view(-1) for w in net.conv_and_linear_weights()])
        reg_loss = flat_weights.dot(flat_weights)
        total_loss = policy_weight * policy_loss + value_weight * value_loss + reg_weight * reg_loss

        # Compute other per-batch metrics
        policy_accuracy = (policy.max(dim=1)[1] == policy_target.max(dim=1)[1]).mean()

        return total_loss
