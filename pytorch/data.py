import glob
import gzip
import random

import numpy as np
import torch
import torch.utils.data as data

import utils
from flatlczero.Game import Game
from flatlczero.Result import Result
from flatlczero.Result import Result
from flatlczero.PieceType import PieceType
import lczero_training_worker as worker


def data_loader(path, batch_size, num_workers=0):
    dataset = Folder(path, transform=worker.load)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=num_workers,
    )

    def infinite_loop(l):
        while True:
            yield from l

    return infinite_loop(loader)


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


if __name__ == '__main__':
    import torch
    import time
    import lczero_training_worker as worker
    t0 = time.perf_counter()
    for _ in range(1_000):
        with gzip.open('game_000000.fbs.gz', 'rb') as fd:
        #with open('game_000000.fbs', 'rb') as fd:
            a = worker.load(fd.read())
    t1 = time.perf_counter()
    print(t1 - t0)

    '''
    rp = RandomPosition(V3_STRUCT.size, fixed=1)
    t0 = time.perf_counter()
    for _ in range(1_000):
        with gzip.open('training.1.gz', 'rb') as fd:
        #with open('training.1', 'rb') as fd:
            a = parse_v3(next(rp(fd.read())))
    t1 = time.perf_counter()
    print(t1 - t0)
    '''
