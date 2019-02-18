import glob
import gzip
import random

import torch
import torch.utils.data as data

import utils
import proto.chunk_pb2 as chunk_pb2


def data_loader(path, batch_size, num_workers=0):
    dataset = Folder(path, transform=Protobuf(history=8))
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
        # returns the index of wdl
        if game.result == chunk_pb2.Game.Result.Value('DRAW'):
            return 1
        # only true when winner is white and playing as white, or winner is black and playing as black
        if (game.result == chunk_pb2.Game.Result.Value('WHITE')) != side_to_move:
            return 0
        # else, side to play lost
        return 2

    def build_policy(self, policy):
        targets = torch.zeros(1858)
        legals = torch.zeros(1858)
        index = torch.LongTensor(policy.index)
        targets.scatter_(dim=0, index=index, src=torch.FloatTensor(policy.prior))
        legals.scatter_(dim=0, index=index, value=1)
        return targets, legals

    def build_input(self, game, position_index):
        planes_per_position = 13
        planes = torch.zeros(self.history * planes_per_position + 8, 64)
        index = 0
        for current_position in range(position_index, position_index - self.history, -1):
            if current_position < 0:
                continue
            state = game.state[current_position]
            self.build_position(planes[index:index + planes_per_position], state)
            index += planes_per_position

        state = game.state[position_index]
        planes[index:index + 8] = torch.FloatTensor([
            state.us_ooo,
            state.us_oo,
            state.them_ooo,
            state.them_oo,
            state.side_to_move,
            state.rule_50,
            0,  # Move count is no longer fed into the net
            1,
        ]).unsqueeze(1)

        planes = planes.view(planes.size(0), 8, 8)
        if state.side_to_move:
            # flip top-to-bottom when playing from black side
            planes = planes.flip(dims=[1])
        return planes

    def build_position(self, planes, state):
        indices = [
            state.white_pawns,
            state.white_knights,
            state.white_bishops,
            state.white_rooks,
            state.white_queens,
            [state.white_king],
            state.black_pawns,
            state.black_knights,
            state.black_bishops,
            state.black_rooks,
            state.black_queens,
            [state.black_king],
        ]
        if state.repetitions:
            indices.append(range(64))
        indices = torch.LongTensor(list(utils.indices_to_plane_indices(indices)))
        planes.view(-1).scatter_(dim=0, index=indices, value=1)


if __name__ == '__main__':
    import time
    p = Protobuf(history=8)
    t0 = time.perf_counter()
    for _ in range(1_000):
        with gzip.open('game_000000.gz', 'rb') as fd:
        #with open('game_000000', 'rb') as fd:
            a = p(fd.read())
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
