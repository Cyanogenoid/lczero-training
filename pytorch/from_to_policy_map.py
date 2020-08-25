import torch
from policy_index import policy_index


def square_to_index(square):
    file, rank = square
    file = ord(file) - ord('a')
    rank = int(rank) - 1
    return 8 * rank + file


def promote_to_index(target, piece):
    file, rank = target
    file = ord(file) - ord('a')
    rank = int(rank) - 1
    if rank < 7:  # trying to promote not on the backrank
        return None
    if not piece:  # a bit iffy, because knight is default promotion
        index = 2
    else:
        index = ['q', 'b', 'k', 'r'].index(piece)
    return 8 * index + file


# castling should really have its own move


def create_from_to_indices():
    source_indices = []
    target_indices = []
    promotion_indices = []
    promotion_valid = []

    for i, move in enumerate(policy_index):
        source = move[0:2]
        target = move[2:4]
        promotion = move[4:]

        source_indices.append(square_to_index(source))
        target_indices.append(square_to_index(target))
        promotion_indices.append(promote_to_index(target, promotion) or 2)
        promotion_valid.append(promote_to_index(target, promotion) is not None)

    source_indices = torch.LongTensor(source_indices)
    target_indices = torch.LongTensor(target_indices)
    promotion_indices = torch.LongTensor(promotion_indices)
    promotion_valid = torch.FloatTensor(promotion_valid)

    return source_indices, target_indices, promotion_indices, promotion_valid


if __name__ == '__main__':
    source, target, promote = create_from_to_indices()
    for a, b, c, d in zip(policy_index, source, target, promote):
        print(a, b.item(), c.item(), d.item())
