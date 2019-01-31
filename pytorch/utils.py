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
