import glob
import gzip
import os
import struct
import random

import torch
import torch.utils.data as data
import numpy as np


def v3_loader(path, batch_size, positions_per_game, buffer_size, num_workers=None):
    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()
    # load chunks from folder
    dataset = Folder(path, positions_per_game)
    # shuffle positions around to decorrelate positions from the same game
    dataset = PositionShuffler(dataset, buffer_size=buffer_size)
    # parse V3 records
    dataset = V3(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    # make loader infinite
    def loop(iterable):
        while True:
            yield from iterable
    loader = loop(loader)
    return loader


class V3(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.v3_struct = struct.Struct('4s7432s832sBBBBBBBb')

    def __getitem__(self, _):
        position = self.dataset[None]
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner = self.v3_struct.unpack(position)
        move_count = 0

        planes = torch.from_numpy(np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32))
        planes = planes.view(104, 8, 8)
        flat_planes = torch.FloatTensor([
            us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count / 99, move_count, 1
        ])
        flat_planes = flat_planes.unsqueeze(1).unsqueeze(2).expand(flat_planes.size(0), 8, 8)
        planes = torch.cat([planes, flat_planes], dim=0)

        probs = torch.from_numpy(np.frombuffer(probs, dtype=np.float32))

        winner = np.float32(winner)

        return planes, probs, winner

    def __len__(self):
        return 2**31


class PositionShuffler(data.Dataset):
    def __init__(self, dataset, buffer_size):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.buffer = []
        self.queue = []

    def __getitem__(self, _):
        ''' Return a random position from a random game '''
        # insert positions as long as either buffer isn't full yet or the position queue is empty
        while len(self.buffer) < self.buffer_size or len(self.queue) == 0:
            self.insert_many(self.random_game())
        return self.queue.pop(0)

    def __len__(self):
        return self.buffer_size

    def random_game(self):
        index = random.randint(0, len(self.dataset))
        return self.dataset[index]

    def insert_many(self, positions):
        for position in positions:
            self.insert(position)

    def insert(self, position):
        if len(self.buffer) < self.buffer_size:
            # just fill the buffer, no need for random ordering
            self.buffer.append(position)
            return None
        else:
            # replace random element in buffer
            index = random.randint(0, self.buffer_size - 1)
            pushed_out, self.buffer[index] = self.buffer[index], position
            self.queue.append(pushed_out)
            return pushed_out


class Folder(data.Dataset):
    def __init__(self, path, positions_per_game):
        self.files = self.find_files(path)
        self.positions_per_game = positions_per_game
        self.record_size = 8276

    def find_files(self, path):
        files = glob.glob(path)
#        files = [f for f in files if os.path.getsize(f) > 0]
        return files

    def __getitem__(self, i):
        path = self.files[i]
        with gzip.open(path, 'rb') as fd:
            chunk = fd.read()
        return self.random_positions(chunk, n=self.positions_per_game)

    def random_positions(self, chunk, n=1):
        num_records = len(chunk) // self.record_size
        for _ in range(n):
            pos = random.randint(0, num_records - 1)
            start = pos * self.record_size
            end = (pos + 1) * self.record_size
            yield chunk[start:end]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    d = V3(Folder('data/v3/train/*'))
    for i in range(len(d)):
        planes, probs, winner = d[i]
        print(f'moves: {probs.nonzero().tolist()}, winner: {winner}')
