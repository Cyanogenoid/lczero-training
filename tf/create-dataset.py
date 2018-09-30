import argparse

import h5py
import numpy as np
from tqdm import tqdm

import train


def make_db(base_dir, size):
    vlen_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
    h5_file = h5py.File(f'{base_dir}chunks.h5')
    dataset = h5_file.create_dataset('chunks', shape=(size,), dtype=vlen_dtype)
    return dataset


def fill_db(dataset, src):
    i = 0
    src.chunks, src.done = src.done, src.done
    print(f'processing {len(src.chunks)} chunks')
    while src.chunks:
        chunk = src.next()
        dataset[i] = tuple(chunk)
        i += 1
        if i % 10_000 == 0:
            print(i)

def fill_db_from_chunks(dataset, chunks):
    for i, filename in enumerate(tqdm(chunks)):
        with open(filename, 'rb') as fd:
            dataset[i] = tuple(fd.read())


db = None
def main(cmd):
    global db
    chunks = train.get_latest_chunks(cmd.path, cmd.num_chunks)
    data_src = train.FileDataSrc(chunks)

    db = make_db(cmd.path, len(chunks))
    fill_db_from_chunks(db, chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--num-chunks', type=int)
    main(parser.parse_args())
