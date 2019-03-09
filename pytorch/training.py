import time
import os
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import data
import model
import checkpoint
from lr import create_lr_schedule
import swa
import summary
import metrics


class Session():
    def __init__(self, cfg):
        self.cfg = cfg

        # let cudnn find the best algorithm
        torch.backends.cudnn.benchmark = True

        print('Building net...')
        self.net = nn.DataParallel(model.Net(
            residual_channels=cfg['model']['residual_channels'],
            residual_blocks=cfg['model']['residual_blocks'],
            policy_channels=cfg['model']['policy_channels'],
            se_ratio=cfg['model']['se_ratio'],
        ).cuda())
        self.optimizer = optim.SGD(self.net.parameters(), lr=1, momentum=0.9, nesterov=True)
        lr_schedule, lr_steps = create_lr_schedule(cfg['training']['lr'])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_schedule)
        print(f'Scheduled LR for {lr_steps} steps')

        print('Constructing data loaders...')
        batch_size = cfg['training']['batch_size']

        num_workers = multiprocessing.cpu_count()
        self.train_loader = data.data_loader(
            path=cfg['dataset']['train_path'],
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_loader = data.data_loader(
            path=cfg['dataset']['test_path'],
            # use smaller batch size when doing gradient accumulation in training, doesn't affect test results
            batch_size=batch_size // cfg['training']['batch_splits'],
            num_workers=num_workers,
        )

        # place to store and accumulate per-batch metrics
        # use self.metric[key] to access, since these are results of possibly multiple virtual batches
        self.metrics = metrics.MetricsManager()

        # SummaryWriters to save metrics for tensorboard to display
        self.summary_path = os.path.join(cfg['logging']['directory'], cfg['name'])
        self.train_writer = SummaryWriter(f'{self.summary_path}-train')
        self.test_writer = SummaryWriter(f'{self.summary_path}-test')

        self.swa = swa.SWA(self)

        self.step = 0

        # TODO more tensorboard metrics
        # TODO graceful shutdown

    def train_loop(self):
        print('Training...')
        if self.step_is_multiple('logging', 'test_every') and self.step > 0:
            self.test_epoch()
            self.swa.test_epoch()
        if self.step == 0:
            summary.model_graph(self)

        time_step_start = time.perf_counter()
        for batch in self.train_loader:

            time_nn_start = time.perf_counter()
            self.train_step(batch)
            self.step += 1  # TODO decide on best place to increment step. before train? here? after test? end?
            time_nn_end = time.perf_counter()

            if self.step_is_multiple('logging', 'train_every'):
                self.print_metrics(prefix='train')
                summary.log_session(self, self.train_writer)
                self.metrics.reset_all()
            if self.step_is_multiple('training', 'swa_every'):
                self.swa.update()
            if self.step_is_multiple('logging', 'test_every'):
                self.test_epoch()
                self.swa.test_epoch()
            if self.step_is_multiple('training', 'checkpoint_every'):
                checkpoint.save(self)
            if self.step_is_multiple('logging', 'weight_histogram_every'):
                summary.weight_histograms(self)

            if self.step_is_multiple('training', 'total_steps'):
                break  # done with training

            # TODO refactor this
            time_step_end = time.perf_counter()
            num_positions = batch[0].size(0)
            nn_speed = num_positions / (time_nn_end - time_nn_start)
            step_speed = num_positions / (time_step_end - time_step_start)
            print(' ' * 8, f'step_speed={step_speed:.0f} pos/s, nn_speed={nn_speed:.0f} pos/s')
            time_step_start = time.perf_counter()

        # only need to save end-of-training checkpoint if we haven't just checkpointed
        if not self.step_is_multiple('training', 'checkpoint_every'):
            checkpoint.save(self)

    def test_epoch(self, prefix='test'):
        self.net.eval()
        self.prefix = 'test'
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                self.forward(batch)
                if i >= self.cfg['logging']['test_steps'] * self.cfg['training']['batch_splits']:
                    break
        self.print_metrics(prefix=prefix)
        summary.log_session(self, self.test_writer)
        self.metrics.reset_all()

    def train_step(self, batch):
        self.net.train()
        # only need to keep them around if we want to compute gradient ratio later
        retain_grad_buffers = self.step_is_multiple('logging', 'gradient_ratio_every')
        if self.cfg['training']['batch_splits'] == 1:
            total_loss, loss_components = self.forward(batch)
            total_loss.backward(retain_graph=retain_grad_buffers)
        else:
            splits = [x.chunk(self.cfg['training']['batch_splits']) for x in batch]
            for split in zip(*splits):
                split_loss, loss_components = self.forward(split)
                (split_loss / len(splits)).backward(retain_graph=retain_grad_buffers)
        gradient_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg['training']['max_gradient_norm'])
        self.lr_scheduler.step(self.step)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.metrics.update('gradient_norm', gradient_norm)
        if self.step_is_multiple('logging', 'gradient_ratio_every'):
            self.metrics.update('policy_value_gradient_ratio', metrics.policy_value_gradient_ratio(self, loss_components))

    def forward(self, batch):
        ''' Perform one step of either training or evaluation
        '''
        # Move batch to the GPU
        input_planes, policy_target, policy_legals, value_target = batch
        input_planes = input_planes.cuda(non_blocking=True)
        policy_target = policy_target.cuda(non_blocking=True)
        value_target = value_target.cuda(non_blocking=True)

        # Forward batch through the network
        policy, value = self.net(input_planes)

        # Compute losses
        policy_logits = F.log_softmax(policy, dim=1)
        # this has the same gradient as cross-entropy
        policy_loss = F.kl_div(policy_logits, policy_target, reduction='batchmean')
        value_loss = F.cross_entropy(value, value_target)
        flat_weights = nn.utils.parameters_to_vector(self.net.module.conv_and_linear_weights())
        reg_loss = flat_weights.dot(flat_weights) / 2
        total_loss = \
            self.cfg['training']['policy_weight'] * policy_loss + \
            self.cfg['training']['value_weight'] * value_loss + \
            self.cfg['training']['reg_weight'] * reg_loss

        # Compute other per-batch metrics
        with torch.no_grad():  # no need to keep store gradient for these
            policy_accuracy = metrics.accuracy(policy, vector=policy_target)
            value_accuracy = metrics.accuracy(value, index=value_target)
            policy_target_entropy = metrics.entropy(policy_target)

        # store the metrics so that other functions have access to them
        self.metrics.update('policy_loss', policy_loss.item())
        self.metrics.update('value_loss', value_loss.item())
        self.metrics.update('reg_loss', reg_loss.item() * 1e-4)
        self.metrics.update('total_loss', total_loss.item())
        self.metrics.update('policy_accuracy', policy_accuracy.item() * 100)
        self.metrics.update('value_accuracy', value_accuracy.item() * 100)
        self.metrics.update('policy_target_entropy', policy_target_entropy.item())

        return total_loss, (policy_loss, value_loss, reg_loss)

    def print_metrics(self, prefix):
        fields = ['total', 'policy', 'value', 'reg']
        values = ['{:.4f}'.format(self.metrics[f'{field}_loss']) for field in fields]
        pairs = list(zip(fields, values))
        pairs.insert(0, ('step', self.step))
        formatted = (f'{field}={value}' for field, value in pairs)
        print(prefix.ljust(8), ', '.join(formatted))

    def step_is_multiple(self, category, option):
        factor = self.cfg[category][option]
        return self.step % factor == 0
