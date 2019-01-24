import glob
import gzip
import os
import struct
import random

import torch
import torch.utils.data as data
import numpy as np


def v3_loader(path, batch_size, num_workers=8):
    return data.DataLoader(
        V3(Folder(path)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


class V3(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.v3_struct = struct.Struct('4s7432s832sBBBBBBBb')

    def __getitem__(self, item):
        position = self.dataset[item]
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner = self.v3_struct.    unpack(position)
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

    def __len__(self):
        return len(self.dataset)


class Folder(data.Dataset):
    def __init__(self, path):
        self.files = self.find_files(path)
        self.record_size = 8276

    def find_files(self, path):
        files = glob.glob(path)
        files = [f for f in files if os.path.getsize(f) > 0]
        return files

    def __getitem__(self, i):
        path = self.files[i]
        with gzip.open(path, 'rb') as fd:
            chunk = fd.read()
        position = self.select_random_position(chunk)
        return position

    def select_random_position(self, chunk):
        num_chunks = len(chunk) // self.record_size
        pos = random.randint(0, num_chunks - 1)
        return chunk[pos * self.record_size:(pos + 1) * self.record_size]

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    d = V3(Folder('data/v3/train/*'))
    for i in range(len(d)):
        planes, probs, winner = d[i]
        print(f'moves: {probs.nonzero().tolist()}, winner: {winner}')
