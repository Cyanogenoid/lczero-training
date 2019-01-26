import collections
import gzip
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

import data
import model
from lr import create_lr_schedule


class Session():
    def __init__(self, cfg):
        self.cfg = cfg

        # let cudnn find the best algorithm
        torch.backends.cudnn.benchmark = True

        print('Building net...')
        self.net = model.Net(
            residual_channels=cfg['model']['residual_channels'],
            residual_blocks=cfg['model']['residual_blocks'],
            policy_channels=cfg['model']['policy_channels'],
            se_ratio=cfg['model']['se_ratio'],
        ).cuda()
        self.net = nn.DataParallel(self.net)  # multi-gpu
        self.optimizer = optim.SGD(self.net.parameters(), lr=1, momentum=0.9, nesterov=True)
        lr_schedule, lr_steps = create_lr_schedule(cfg['training']['lr'])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_schedule)
        print(f'Scheduled LR for {lr_steps} steps')

        print('Constructing data loaders...')
        batch_size = cfg['training']['batch_size']

        self.train_loader = data.v3_loader(
            path=cfg['dataset']['train_path'],
            batch_size=batch_size,
            buffer_size=cfg['training']['shufflebuffer_size'],
            positions_per_game=cfg['training']['positions_per_game'],
        )
        self.test_loader = data.v3_loader(
            path=cfg['dataset']['test_path'],
            # use smaller batch size when doing gradient accumulation in training, doesn't affect test results
            batch_size=batch_size // cfg['training']['batch_splits'],
            buffer_size=cfg['training']['shufflebuffer_size'],
            positions_per_game=cfg['training']['positions_per_game'],
        )
        print('Prefetching data...')
        next(iter(self.train_loader))
        next(iter(self.test_loader))

        # place to store and accumulate per-batch metrics
        # use self.metric(key) to access, since these are results of possibly multiple virtual batches
        self.metrics = collections.defaultdict(list)

        # SummaryWriters to save metrics for tensorboard to display
        run_path = os.path.join(cfg['logging']['directory'], cfg['name'])
        self.train_writer = SummaryWriter(f'{run_path}-train')
        self.test_writer = SummaryWriter(f'{run_path}-test')

        self.step = 0

        # TODO swa

    def train_loop(self):
        print('Training...')
        if self.step_is_multiple(self.cfg['logging']['test_every']) and self.step > 0:
            self.test_epoch()

        t0 = time.perf_counter()
        for batch in self.train_loader:

            t1 = time.perf_counter()
            self.train_step(batch)
            self.step += 1  # TODO decide on best place to increment step. before train? here? after test? end?

            t2 = time.perf_counter()
            print(self.step, self.metric('policy_loss'), self.metric('value_loss'), self.metric('total_loss'), batch[0].size())

            self.log_metrics(self.train_writer)
            self.reset_metrics()

            if self.step_is_multiple(self.cfg['logging']['test_every']):
                self.test_epoch()
            if self.step_is_multiple(self.cfg['training']['checkpoint_every']):
                self.checkpoint()
            if self.step_is_multiple(self.cfg['training']['total_steps']):
                # done with training
                break
            t3 = time.perf_counter()
            print(batch[0].size(0)/(t2-t1), batch[0].size(0)/(t3-t0))
            t0 = time.perf_counter()

        # only need to save end-of-training checkpoint if we haven't just checkpointed
        if not self.step_is_multiple(self.cfg['training']['checkpoint_every']):
            self.checkpoint()

    def test_epoch(self):
        self.net.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                self.forward(batch)
                if i >= self.cfg['logging']['test_steps'] * self.cfg['training']['batch_splits']:
                    break
        self.log_metrics(self.test_writer)
        self.reset_metrics()
        self.net.train()

    def train_step(self, batch):
        self.net.train()
        if self.cfg['training']['batch_splits'] == 1:
            total_loss = self.forward(batch)
            total_loss.backward()
        else:
            splits = [torch.FloatTensor.chunk(x, self.cfg['training']['batch_splits']) for x in batch]
            for split in zip(*splits):
                split_loss = self.forward(split) / len(splits)
                split_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg['training']['max_gradient_norm'])
        self.metrics['gradient_norm'].append(gradient_norm)
        self.lr_scheduler.step(self.step)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def forward(self, batch):
        ''' Perform one step of either training or evaluation
        '''
        # Move batch to the GPU
        input_planes, policy_target, value_target = batch
        input_planes = input_planes.cuda(async=True)
        policy_target = policy_target.cuda(async=True)
        value_target = value_target.cuda(async=True)

        # Forward batch through the network
        policy, value = self.net(input_planes)

        # Compute losses
        policy_logits = F.log_softmax(policy, dim=1)
        policy_loss = F.kl_div(policy_logits, policy_target, reduction='batchmean')  # this has the same gradient as cross-entropy
        value_loss = F.mse_loss(value.squeeze(dim=1), value_target)
        reg_loss = sum(w.view(-1).dot(w.view(-1)) / 2 for w in flat_weights)
        total_loss = \
            self.cfg['training']['policy_weight'] * policy_loss + \
            self.cfg['training']['value_weight'] * value_loss + \
            self.cfg['training']['reg_weight'] * reg_loss

        # Compute other per-batch metrics
        with torch.no_grad():  # no need to keep store gradient for these
            policy_accuracy = (policy.max(dim=1)[1] == policy_target.max(dim=1)[1]).float().mean()
            policy_target_entropy = (policy_target * policy_target.log()).sum(dim=1).mean()

        # store the metrics so that other functions have access to them
        self.metrics['policy_loss'].append(policy_loss.item())
        self.metrics['value_loss'].append(value_loss.item())
        self.metrics['reg_loss'].append(reg_loss.item())
        self.metrics['total_loss'].append(total_loss.item())
        self.metrics['policy_accuracy'].append(policy_accuracy.item())
        self.metrics['policy_target_entropy'].append(policy_target_entropy.item())

        return total_loss

    def resume(self, path=None):
        if path is None:
            directory = os.path.join(self.cfg['training']['checkpoint_directory'], self.cfg['name'])
            with open(os.path.join(directory, 'latest'), 'r') as fd:
                path = fd.read().strip()
        if not os.path.exists(path):
            raise OSError('"{}" does not exist.')
        print(f'Resuming from "{path}"...')
        checkpoint = torch.load(path)
        self.net.module.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']

    def checkpoint(self):
        checkpoint = {
            'net': self.net.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }
        directory = os.path.join(self.cfg['training']['checkpoint_directory'], self.cfg['name'])
        if not os.path.exists(directory):
            os.makedirs(directory)
        # proto weights
        filename = f'net-{self.step}.pb.gz'
        path = os.path.join(directory, filename)
        self.net.module.export_proto(path)
        # checkpoint
        filename = f'checkpoint-{self.step}.pth'
        path = os.path.join(directory, filename)
        torch.save(checkpoint, path)
        print(f'Checkpoint saved to "{path}"')
        # store path so that we know what checkpoint to resume from without specifying it
        with open(os.path.join(directory, 'latest'), 'w') as fd:
            fd.write(f'{path}\n')

    def log_metrics(self, writer):
        writer.add_scalar('Loss/Policy', self.metric('policy_loss'), global_step=self.step)
        writer.add_scalar('Loss/Value', self.metric('value_loss') / 4, global_step=self.step)
        if writer == self.train_writer:
            writer.add_scalar('Loss/Weight', self.metric('reg_loss') * 1e-4, global_step=self.step)
        writer.add_scalar('Loss/Total', self.metric('total_loss'), global_step=self.step)
        writer.add_scalar('Policy/Accuracy', self.metric('policy_accuracy'), global_step=self.step)
        if writer == self.train_writer:
            writer.add_scalar('Gradient Norm', self.metric('gradient_norm'), global_step=self.step)
            writer.add_scalar('LR', self.lr_scheduler.get_lr()[0], global_step=self.step)
        self.reset_metrics()

    def metric(self, key):
        return np.mean(self.metrics[key])

    def reset_metrics(self):
        for key in list(self.metrics.keys()):
            self.metrics[key] = []

    def step_is_multiple(self, factor):
        return self.step % factor == 0
