import bisect
import numpy as np


def create_lr_schedule(schedule, starting_lr=0):
    """ Construct a piecewise function that given a step returns the corresponding learning rate
    """

    # each of these returns a function that, given the ratio in the interval [0, 1], returns a learning rate
    def constant(entry, previous_lr):
        lr = entry.get('at', previous_lr)
        return lambda ratio: lr

    def linear(entry, previous_lr):
        start = entry.get('start', previous_lr)
        end = entry['end']
        return lambda ratio: (1 - ratio) * start + ratio * end

    def log(entry, previous_lr):
        start = np.log(entry.get('start', previous_lr))
        end = np.log(entry['end'])
        return lambda ratio: np.exp((1 - ratio) * start + ratio * end)

    def cosine(entry, previous_lr):
        low = entry['low']
        high = entry.get('high', previous_lr) - low
        offset = float(entry.get('offset', 0))
        cycle = float(entry.get('cycle', 1))
        assert 0 < cycle <= 1
        return lambda ratio: high * (np.cos((ratio * cycle + offset) * 2 * np.pi) / 2 + 0.5) + low

    def repeat(entry, previous_lr):
        schedule, steps = create_lr_schedule(entry['schedule'], previous_lr)
        times = entry['times']
        entry['steps'] = steps * times
        return lambda ratio: schedule(int(ratio * steps * times) % steps) if ratio < 1 else schedule(steps)

    fn_map = {
        'constant': constant,
        'linear': linear,
        'log': log,
        'cosine': cosine,
        'repeat': repeat,
    }

    current_lr = starting_lr
    current_steps = 0
    pieces = []
    boundaries = [0]
    for entry in schedule:
        try:
            lr_type = fn_map[entry['type']]
        except KeyError:
            raise KeyError('unknown lr schedule type: {}'.format(entry['type']))
        piece = lr_type(entry, current_lr)
        current_steps += entry['steps']
        current_lr = piece(1)

        pieces.append(piece)
        boundaries.append(current_steps)

    def get_lr(step):
        # subtract 1 because first entry in boundaries is a 0
        idx = bisect.bisect_right(boundaries, step) - 1
        try:
            left, right = boundaries[idx], boundaries[idx + 1]
        except IndexError:
            # when more steps than scheduled, just output last lr as constant
            return pieces[-1](1)
        ratio = (step - left) / (right - left)
        return pieces[idx](ratio)

    return get_lr, current_steps
