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
        # use self.metric(key) to access, since these are results of possibly multiple virtual batches
        self.metrics = collections.defaultdict(list)

        # SummaryWriters to save metrics for tensorboard to display
        self.summary_path = os.path.join(cfg['logging']['directory'], cfg['name'])
        self.train_writer = SummaryWriter(f'{self.summary_path}-train')
        self.test_writer = SummaryWriter(f'{self.summary_path}-test')

        self.swa = swa.SWA(self)

        self.step = 0

        # TODO more tensorboard metrics, graph, histograms
        # TODO graceful shutdown
        # TODO put data loader behind an mp.Queue and so that batching is not done on main thread
        # TODO verify against jio net with se_ratio: 8, policy_channel:32, position sample rate 32

    def train_loop(self):
        print('Training...')
        if self.step_is_multiple(self.cfg['logging']['test_every']) and self.step > 0:
            self.test_epoch()
            self.swa.test_epoch()

        time_step_start = time.perf_counter()
        for batch in self.train_loader:

            time_nn_start = time.perf_counter()
            self.train_step(batch)
            self.step += 1  # TODO decide on best place to increment step. before train? here? after test? end?
            time_nn_end = time.perf_counter()

            self.print_metrics(prefix='train')
            self.log_metrics(self.train_writer)
            self.reset_metrics()

            self.swa.update()

            if self.step_is_multiple(self.cfg['logging']['test_every']):
                self.test_epoch()
                self.swa.test_epoch()
            if self.step_is_multiple(self.cfg['training']['checkpoint_every']):
                checkpoint.save(self)
            if self.step == 1:
                summary.model_graph(self)
            if self.step_is_multiple(self.cfg['logging']['weight_histogram_every']):
                summary.weight_histograms(self)

            if self.step_is_multiple(self.cfg['training']['total_steps']):
                # done with training
                break

            # TODO refactor this
            time_step_end = time.perf_counter()
            num_positions = batch[0].size(0)
            nn_speed = num_positions / (time_nn_end - time_nn_start)
            step_speed = num_positions / (time_step_end - time_step_start)
            print(' ' * 8, f'step_speed={step_speed:.0f} pos/s, nn_speed={nn_speed:.0f} pos/s')
            time_step_start = time.perf_counter()

        # only need to save end-of-training checkpoint if we haven't just checkpointed
        if not self.step_is_multiple(self.cfg['training']['checkpoint_every']):
            checkpoint.save(self)

    def test_epoch(self, prefix='test'):
        self.net.eval()
        self.prefix = 'test'
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                self.forward(batch)
                if i >= self.cfg['logging']['test_steps'] * self.cfg['training']['batch_splits']:
                    summary.policy_value_gradient_ratio(self, batch)
                    break
        self.print_metrics(prefix=prefix)
        self.log_metrics(self.test_writer)
        self.reset_metrics()
        summary.policy_weight_skewness(self)

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
        value_loss = F.mse_loss(value.squeeze(dim=1), value_target)
        flat_weights = nn.utils.parameters_to_vector(self.net.module.conv_and_linear_weights())
        reg_loss = flat_weights.dot(flat_weights) / 2
        total_loss = \
            self.cfg['training']['policy_weight'] * policy_loss + \
            self.cfg['training']['value_weight'] * value_loss + \
            self.cfg['training']['reg_weight'] * reg_loss

        # Compute other per-batch metrics
        with torch.no_grad():  # no need to keep store gradient for these
            policy_accuracy = metrics.accuracy(policy, policy_target)
            policy_target_entropy = metrics.entropy(policy_target)

        # store the metrics so that other functions have access to them
        self.metrics['policy_loss'].append(policy_loss.item())
        self.metrics['value_loss'].append(value_loss.item() / 4)
        self.metrics['reg_loss'].append(reg_loss.item() * 1e-4)
        self.metrics['total_loss'].append(total_loss.item())
        self.metrics['policy_accuracy'].append(policy_accuracy.item() * 100)
        self.metrics['policy_target_entropy'].append(policy_target_entropy.item())

        return total_loss

    def log_metrics(self, writer):
        writer.add_scalar('loss/policy', self.metric('policy_loss'), global_step=self.step)
        writer.add_scalar('loss/value', self.metric('value_loss'), global_step=self.step)
        if writer == self.train_writer:
            writer.add_scalar('loss/weight', self.metric('reg_loss'), global_step=self.step)
        writer.add_scalar('loss/total', self.metric('total_loss'), global_step=self.step)
        writer.add_scalar('metrics/policy_accuracy', self.metric('policy_accuracy'), global_step=self.step)
        if writer == self.train_writer:  # target data comes from same distribution, so no point plotting it again
            writer.add_scalar('metrics/policy_target_entropy', self.metric('policy_target_entropy'), global_step=self.step)
        if writer == self.train_writer:
            writer.add_scalar('metrics/gradient_norm', self.metric('gradient_norm'), global_step=self.step)
            writer.add_scalar('hyperparameter/learning_rate', self.lr_scheduler.get_lr()[0], global_step=self.step)
        self.reset_metrics()

    def metric(self, key):
        return np.mean(self.metrics[key])

    def reset_metrics(self):
        for key in list(self.metrics.keys()):
            self.metrics[key] = []

    def print_metrics(self, prefix):
        fields = ['total', 'policy', 'value', 'reg']
        values = ['{:.4f}'.format(self.metric(f'{field}_loss')) for field in fields]
        pairs = list(zip(fields, values))
        pairs.insert(0, ('step', self.step))
        formatted = (f'{field}={value}' for field, value in pairs)
        print(prefix.ljust(8), ', '.join(formatted))

    def step_is_multiple(self, factor):
        return self.step % factor == 0
