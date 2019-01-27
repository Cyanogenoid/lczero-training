import glob
import gzip
import os
import random
import struct
import itertools

import torch
import torch.utils.data as data
import numpy as np

import dataloader
import utils


V3_STRUCT = struct.Struct('4s7432s832sBBBBBBBb')


def v3_loader(path, batch_size, positions_per_game, shufflebuffer_size, num_workers=None):
    # load chunks from folder
    dataset = Folder(path, positions_per_game)
    # infinite generator of positions
    position_loader = loop_positions(dataset)
    # multi-threaded data loader
    loader = dataloader.ShufflingDataLoader(lambda: position_loader, shuffle_size=shufflebuffer_size, struct_size=V3_STRUCT.size)
    # parse v3 records into PyTorch tensors
    loader = map(parse_v3, loader)
    # only include correctly parsed v3 records
    loader = filter(lambda x: x is not None, loader)
    # group records into groups of size batch_size
    loader = utils.grouper(loader, batch_size)
    # turn groups into PyTorch batches
    loader = map(collate_positions, loader)
    return loader


def collate_positions(batch):
    planes, probs, winner = zip(*batch)
    planes = torch.stack(planes)
    probs = torch.stack(probs)
    winner = torch.FloatTensor(winner)
    return planes, probs, winner


def parse_v3(position):
    try:
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner = V3_STRUCT.unpack(position)
    except:
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

    return planes, probs, winner


def loop_positions(dataset):
    """ Infinitely yield positions from a dataset """
    while True:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for index in indices:
            yield from dataset[index]


class Folder(data.Dataset):
    def __init__(self, path, positions_per_game):
        self.files = self.find_files(path)
        self.positions_per_game = positions_per_game
        self.record_size = V3_STRUCT.size

    def find_files(self, path):
        files = glob.glob(path)
#        files = [f for f in files if os.path.getsize(f) > 0]
        return files

    def __getitem__(self, i):
        path = self.files[i]
        try:
            with gzip.open(path, 'rb') as fd:
                chunk = fd.read()
            return self.random_positions(chunk, n=self.positions_per_game)
        except EOFError:
            print('Skipping', path)
            return []

    def random_positions(self, chunk, n=1):
        num_records = len(chunk) // self.record_size
        for _ in range(n):
            pos = random.randint(0, num_records - 1)
            start = pos * self.record_size
            end = (pos + 1) * self.record_size
            yield chunk[start:end]

    def __len__(self):
        return len(self.files)
