import itertools


def grouper(iterable, n, fillvalue=None):
    """ Collect data into fixed-length chunks or blocks
    From https://docs.python.org/3.7/library/itertools.html#itertools-recipes
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def variables(net):
    yield from net.parameters()
    yield from net.buffers()


def named_variables(net):
    yield from net.named_parameters()
    yield from net.named_buffers()


Mod37BitPosition = (
        32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
  7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
  20, 8, 19, 18
)

def ctz(v):
    right_half = v & 0x0000FFFF
    if right_half != 0:
        v = right_half
        offset = 0
    else:
        v >>= 32  # only use left half
        offset = 32
    idx = Mod37BitPosition[(-v & v) % 37]
    return idx + offset


def bit_indices(x):
    while x:
        yield ctz(x)
        x &= x - 1


def grouped_bit_indices(ns, group_size=64):
    offset = 0
    for n in ns:
        for index in bit_indices(n):
            yield offset + index
        offset += group_size
