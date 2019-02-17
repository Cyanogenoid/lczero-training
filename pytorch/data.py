import glob
import gzip
import random
import struct

import torch
import torch.utils.data as data
import numpy as np

import dataloader
import proto.chunk_pb2 as chunk_pb2


V3_STRUCT = struct.Struct('4s7432s832sBBBBBBBb')


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
        position_sampler = RandomPosition(**{sample_method: sample_argument}, record_size=V3_STRUCT.size)
        # load chunks from folder
        dataset = Folder(path, transform=position_sampler)
        # infinite generator of positions
        position_loader = loop_positions(dataset)
        # multi-threaded data loader
        loader = dataloader.ShufflingDataLoader(
            lambda: position_loader,
            shuffle_size=shufflebuffer_size,
            struct_size=V3_STRUCT.size,
        )
        # parse v3 records into PyTorch tensors
        loader = map(parse_v3, loader)
        # only include correctly parsed v3 records
        loader = filter(lambda x: x is not None, loader)
        self.loader = loader

    def __getitem__(self, _):
        return next(self.loader)

    def __len__(self):
        return 2**30


def parse_v3(position):
    try:
        ver, probs, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_count, move_count, winner = V3_STRUCT.unpack(position)
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
    wdl = [1, 0, -1].index(winner)

    return planes, probs, wdl


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


def bit_indices(n):
    index = 0
    while n:
        if n & 1:
            yield index
        n >>= 1
        index += 1


class Protobuf():
    def __init__(self, history):
        self.history = history

    def __call__(self, data):
        chunk = chunk_pb2.Chunk.FromString(data)
        game = chunk.game[0]  # assumes 1 game per chunk
        # select random position
        position_index = random.randrange(len(game.state))
        side_to_move = game.state[position_index].side_to_move  # 0 if white, 1 if black

        wdl = self.build_wdl(game, side_to_move)
        policy, legals = self.build_policy(game.policy[position_index])
        planes = self.build_input(game, position_index)
        return planes, policy, wdl

    def build_wdl(self, game, side_to_move):
        if game.result == chunk_pb2.Game.Result.Value('DRAW'):
           return [0, 1, 0]
        # only true when winner is white and playing as white, or winner is black and playing as black
        if (game.result == chunk_pb2.Game.Result.Value('WHITE')) != side_to_move:
            return [1, 0, 0]
        return [0, 0, 1]

    def build_policy(self, policy):
        targets = torch.zeros(1858)
        legals = torch.zeros(1858)
        priors = policy.prior
        indices = policy.index
        for index, prior in zip(indices, priors):
            targets[index] = prior
            legals[index] = 1
        return targets, legals

    def build_input(self, game, position_index):
        planes_per_position = 13
        planes = torch.zeros(self.history * planes_per_position + 8, 64)
        base = 0
        for current_position in range(position_index, position_index - self.history, -1):
            if current_position < 0:
                continue
            state = game.state[current_position]
            self.build_position(planes[base:base + planes_per_position], state)
            base += planes_per_position

        state = game.state[position_index]
        planes[base + 0] = state.us_ooo
        planes[base + 1] = state.us_oo
        planes[base + 2] = state.them_ooo
        planes[base + 3] = state.them_oo
        planes[base + 4] = state.side_to_move
        planes[base + 5] = state.rule_50
        # Move count is no longer fed into the net
        # planes[base + 6] = state.move_count
        planes[base + 7] = 1

        return planes.view(planes.size(0), 8, 8)

    def build_position(self, planes, state, mirror=False):
        self.build_plane(planes[0], state.our_pawns, mirror)
        self.build_plane(planes[1], state.our_knights, mirror)
        self.build_plane(planes[2], state.our_bishops, mirror)
        self.build_plane(planes[3], state.our_rooks, mirror)
        self.build_plane(planes[4], state.our_queens, mirror)
        self.build_plane(planes[5], state.our_king, mirror)
        self.build_plane(planes[6], state.our_pawns, mirror)
        self.build_plane(planes[7], state.our_knights, mirror)
        self.build_plane(planes[8], state.our_bishops, mirror)
        self.build_plane(planes[9], state.our_rooks, mirror)
        self.build_plane(planes[10], state.our_queens, mirror)
        self.build_plane(planes[11], state.our_king, mirror)
        planes[12] = state.repetitions

    def build_plane(self, plane, bitstring, mirror=False):
        if mirror:
            bitstring = mirror(bitstring)
        for i in bit_indices(bitstring):
            plane[i] = 1
        return plane


if __name__ == '__main__':
    import gzip
    import time
    p = Protobuf(history=8)
    t0 = time.perf_counter()
    for _ in range(1_000):
        with gzip.open('game_000000.gz', 'rb') as fd:
        #with open('game_000000', 'rb') as fd:
            a = p(fd.read())
    t1 = time.perf_counter()
    print(t1 - t0)

    rp = RandomPosition(V3_STRUCT.size, fixed=1)
    t0 = time.perf_counter()
    for _ in range(1_000):
        #with gzip.open('training.1.gz', 'rb') as fd:
        with open('training.1', 'rb') as fd:
            a = parse_v3(next(rp(fd.read())))
    t1 = time.perf_counter()
    print(t1 - t0)
