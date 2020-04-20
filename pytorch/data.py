import glob
import gzip
import random
import struct

import torch
import torch.utils.data as data
import numpy as np

import dataloader


V4_STRUCT = struct.Struct('4s7432s832sBBBBBBBb')
V4_STRUCT = struct.Struct('4s7432s832sBBBBBBBbffff')


def v3_loader(path, batch_size, sample_method, sample_argument, shufflebuffer_size, num_workers=None):
    dataset = Positions(path, sample_method, sample_argument, shufflebuffer_size, num_workers=num_workers)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        # shufflebuffer isn't threadsafe, so only use one thread to
        # do shufflebuffer work, parse v3 records, and form batches
        num_workers=1,
    )
    return loader


class Positions(data.Dataset):
    def __init__(self, path, sample_method, sample_argument, shufflebuffer_size, num_workers=None):
        position_sampler = RandomPosition(**{sample_method: sample_argument}, record_size=V4_STRUCT.size)
        # load chunks from folder
        dataset = Folder(path, transform=position_sampler)
        # infinite generator of positions
        position_loader = loop_positions(dataset)
        # multi-threaded data loader
        loader = dataloader.ShufflingDataLoader(
            lambda: position_loader,
            shuffle_size=shufflebuffer_size,
            struct_size=V4_STRUCT.size,
        )
        # parse v3 records into PyTorch tensors
        loader = map(parse_v4, loader)
        # only include correctly parsed v3 records
        loader = filter(lambda x: x is not None, loader)
        self.loader = loader

    def __getitem__(self, _):
        return next(self.loader)

    def __len__(self):
        return 2**30


def parse_v4(position):
    try:
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner, root_q, best_q, root_d, best_d = V4_STRUCT.unpack(position)
    except struct.error:
        return None
    move_count = 0

    planes = torch.from_numpy(np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32))
    planes = planes.view(104, 8, 8)
    flat_planes = torch.FloatTensor([
        us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count / 99, move_count, 1
    ])
    flat_planes = flat_planes.unsqueeze(1).unsqueeze(2).expand(flat_planes.size(0), 8, 8)
    planes = torch.cat([planes, flat_planes], dim=0)

    probs = torch.from_numpy(np.frombuffer(probs, dtype=np.float32))
    z_wdl = torch.FloatTensor([winner == 1, winner == 0, winner == -1])

    best_q_w = 0.5 * (1 - best_d + best_q)
    best_q_l = 0.5 * (1 - best_d - best_q)
    q_wdl = torch.FloatTensor([best_q_w, best_d, best_q_l])

    return planes, probs, z_wdl, q_wdl


def loop_positions(dataset):
    """ Infinitely yield positions from a dataset """
    while True:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for index in indices:
            yield from dataset[index]


class Folder(data.Dataset):
    def __init__(self, path, transform=lambda x: x):
        self.files = self.find_files(path)
        self.transform = transform

    def find_files(self, path):
        files = glob.glob(path)
#        files = [f for f in files if os.path.getsize(f) > 0]
        return files

    def __getitem__(self, i):
        path = self.files[i]
        try:
            with gzip.open(path, 'rb') as fd:
                chunk = fd.read()
            return self.transform(chunk)
        except EOFError:
            print('Skipping', path)
            return []

    def __len__(self):
        return len(self.files)


class RandomPosition():
    def __init__(self, record_size, fixed=None, fixed_strict=None, subsample=None):
        self.record_size = record_size
        assert (fixed is not None) + (fixed_strict is not None) + (subsample is not None) == 1, 'can only specify one sampling ype'
        if fixed is not None:
            self.sample = lambda chunk: self.sample_fixed(chunk, fixed)
        elif fixed_strict is not None:
            self.sample = lambda chunk: self.sample_fixed_strict(chunk, fixed_strict)
        elif subsample is not None:
            self.sample = lambda chunk: self.sample_subsample(chunk, subsample)

    def __call__(self, chunk):
        for pos in self.sample(chunk):
            yield self.index(chunk, pos)

    def sample_fixed(self, chunk, n):
        """ Sample a fixed (if possible) number of records from a game without replacement"""
        num_records = len(chunk) // self.record_size
        max_len = min(num_records, n)
        return random.sample(range(num_records), k=max_len)

    def sample_fixed_strict(self, chunk, n):
        """ Sample a fixed number of records from a game with replacement """
        num_records = len(chunk) // self.record_size
        return random.choices(range(num_records), k=n)

    def sample_subsample(self, chunk, nth):
        """ Sample a varying number of records from a game, subsample every nth on average """
        num_records = len(chunk) // self.record_size
        for pos in range(num_records):
            if random.random() < 1/nth:
                yield pos

    def index(self, chunk, number):
        start = number * self.record_size
        end = (number + 1) * self.record_size
        return chunk[start:end]
