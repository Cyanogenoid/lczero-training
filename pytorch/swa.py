from contextlib import contextmanager
import copy

import torch
from tensorboardX import SummaryWriter

import utils


class SWA():
    """ While this is called Stochastic Weight Averaging [0], the version that is actually used
        is closer to [1] because samples are weighted by an exponential moving average instead of
        a cumulative moving average.

        [0]: https://arxiv.org/abs/1803.05407
        [1]: https://arxiv.org/abs/1703.01780
    """
    def __init__(self, session):
        self.session = session
        self.momentum = session.cfg['training']['swa_momentum']
        self.active = False
        if self.enabled:
            self.net = copy.deepcopy(session.net)
            for param in self.net.parameters():
                param.requires_grad = False
            self.test_writer = SummaryWriter(f'{session.summary_path}-swa-test')

    @property
    def enabled(self):
        return self.momentum > 0.0

    def update(self):
        if not self.enabled:
            return
        for swa, base in zip(utils.variables(self.net), utils.variables(self.session.net)):
            if swa.dtype == torch.int64:
                continue  # only apply to things that can be averaged
            # exponential moving average without bias correction
            # just don't use the swa nets early on and everything is ok
            swa.data = self.momentum * swa + (1 - self.momentum) * base.detach()

    def test_epoch(self):
        if not self.enabled:
            return
        with self.activate():
            self.session.test_epoch(prefix='test swa')

    @contextmanager
    def activate(self):
        # use swa version of net
        self.session.net, self.net = self.net, self.session.net
        # use swa test writer
        self.session.test_writer, self.test_writer = self.test_writer, self.session.test_writer
        # make it known that swa is being used
        self.active = True
        yield
        # revert the changes
        self.session.net, self.net = self.net, self.session.net
        self.session.test_writer, self.test_writer = self.test_writer, self.session.test_writer
        self.active = False
