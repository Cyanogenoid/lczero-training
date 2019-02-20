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


class Flatbuffers():
    def __init__(self, history):
        self.history = history
        self.piece_type_to_index = np.vectorize([
            #PieceType.Pawn: 0 * 64,
            #PieceType.Knight: 1 * 64,
            #PieceType.Bishop: 2 * 64,
            #PieceType.Rook: 3 * 64,
            #PieceType.Queen: 4 * 64,
            #PieceType.King: 5 * 64,
            0 * 64,
            1 * 64,
            2 * 64,
            3 * 64,
            4 * 64,
            5 * 64,
        ].__getitem__)

    #@profile
    def __call__(self, data):
        game = Game.GetRootAsGame(data, 0)
        # select random position
        position_index = random.randrange(game.StatesLength())
        state = game.States(position_index)

        wdl = self.build_wdl(game, state.Position().SideToMove())
        policy, legals = self.build_policy(state.Policy())
        planes = self.build_input(game, position_index)
        return planes, policy, wdl

    #@profile
    def build_wdl(self, game, side_to_move):
        winner = game.Winner()
        if winner == Result.Draw:
            return 1
        # only true when winner is white and playing as white, or winner is black and playing as black
        if (winner == Result.White) != side_to_move:
            return 0
        # else, side to play lost
        return 2

    #@profile
    def build_policy(self, policy):
        targets = torch.zeros(1858)
        legals = torch.zeros(1858)
        index = torch.from_numpy(policy.IndexAsNumpy().astype(np.int64))
        probability = torch.from_numpy(policy.ProbabilityAsNumpy())
        targets.scatter_(dim=0, index=index, src=probability)
        legals.scatter_(dim=0, index=index, value=1)
        return targets, legals

    #@profile
    def build_input(self, game, position_index):
        planes_per_position = 13
        planes = torch.zeros(self.history * planes_per_position + 8, 64)
        index = 0
        for current_position in range(position_index, position_index - self.history, -1):
            if current_position < 0:
                continue
            position = game.States(current_position).Position()
            self.build_position(planes[index:index + planes_per_position], position)
            index += planes_per_position

        position = game.States(position_index).Position()
        planes[index:index + 8] = torch.FloatTensor([
            position.UsOoo(),
            position.UsOo(),
            position.ThemOoo(),
            position.ThemOo(),
            position.SideToMove(),
            position.Rule50(),
            0,  # Move count is no longer fed into the net
            1,
        ]).unsqueeze(1)

        planes = planes.view(planes.size(0), 8, 8)
        if position.SideToMove():
            # flip top-to-bottom when playing from black side
            planes = planes.flip(dims=[1])
        return planes

    #@profile
    def build_position(self, planes, position):
        white = position.White()
        black = position.Black()
        self.build_pieces(planes[0:6], white)
        self.build_pieces(planes[6:12], black)
        if position.Repetitions():
            planes[12].fill_(1)

    #@profile
    def build_pieces(self, planes, pieces):
        indices = pieces.IndicesAsNumpy()
        types = pieces.TypesAsNumpy()
        type_indices = self.piece_type_to_index(types)
        indices = type_indices + indices
        indices = torch.from_numpy(indices.astype(np.int64))
        planes.view(-1).scatter_(dim=0, index=indices, value=1)


if __name__ == '__main__':
    import time
    p = Flatbuffers(history=8)
    t0 = time.perf_counter()
    for _ in range(1_000):
        with gzip.open('game_000000.fbs.gz', 'rb') as fd:
        #with open('game_000000', 'rb') as fd:
            a = p(bytearray(fd.read()))
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
