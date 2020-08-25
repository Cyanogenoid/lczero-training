import torch
from policy_index import policy_index


def square_to_index(square):
    file, rank = parse_square(square)
    return 8 * rank + file


def parse_square(square):
    file, rank = square
    file = ord(file) - ord('a')
    rank = int(rank) - 1
    return file, rank

def promote_to_index(source, target, piece):
    source_file, source_rank = parse_square(source)
    target_file, target_rank = parse_square(target)
    if target_rank != 7:  # trying to promote not on the backrank
        return None
    if source_rank != 6:  # not a 1 step pawn move
        return None
    if abs(source_file - target_file) >= 2:  # actually a knight move
        return None
    if not piece:  # a bit iffy, because knight is default promotion
        index = 2
    else:
        index = ['q', 'b', 'k', 'r'].index(piece)
    return 8 * index + target_file


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
        promotion_indices.append(promote_to_index(source, target, promotion) or 2)
        promotion_valid.append(promote_to_index(source, target, promotion) is not None)

    source_indices = torch.LongTensor(source_indices)
    target_indices = torch.LongTensor(target_indices)
    promotion_indices = torch.LongTensor(promotion_indices)
    promotion_valid = torch.FloatTensor(promotion_valid)

    return source_indices, target_indices, promotion_indices, promotion_valid


if __name__ == '__main__':
    source, target, promote = create_from_to_indices()
    for a, b, c, d in zip(policy_index, source, target, promote):
        print(a, b.item(), c.item(), d.item())
