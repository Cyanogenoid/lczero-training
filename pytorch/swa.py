from contextlib import contextmanager
import copy

import torch
from tensorboardX import SummaryWriter


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
        for swa, base in zip(self.net.parameters(), self.session.net.parameters()):
            # exponential moving average without bias correction
            # just don't use the swa nets early on and everything is ok
            w = self.momentum * swa + (1 - self.momentum) * base
            swa[:] = w

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
        yield
        # revert the changes
        self.session.net, self.net = self.net, self.session.net
        self.session.test_writer, self.test_writer = self.test_writer, self.session.test_writer
