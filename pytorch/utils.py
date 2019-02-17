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


def ctz(x):
    if x == 0:
        return None
    n = 0
    if x & 0xFFFFFFFF == 0:
        n += 32
        x >>= 32
    if x & 0x0000FFFF == 0:
        n += 16
        x >>= 16
    if x & 0x000000FF == 0:
        n += 8
        x >>= 8
    if x & 0x0000000F == 0:
        n += 4
        x >>= 4
    if x & 0x00000003 == 0:
        n += 2
        x >>= 2
    if x & 0x00000001 == 0:
        n += 1
    return n


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
