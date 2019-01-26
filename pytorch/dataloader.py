#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

import multiprocessing as mp
import shufflebuffer as sb


class ShufflingDataLoader:
    def __init__(self, chunkdatasrc, struct_size, shuffle_size=1, workers=None):
        """
        Read data and yield batches of raw tensors.

        'chunkdatasrc' is an object yeilding chunkdata
        'shuffle_size' is the size of the shuffle buffer.
        'workers' is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v3 format where each record
        consists of (state, policy, result)
        """

        self.shuffle_size = shuffle_size
        self.struct_size = struct_size
        workers = workers or mp.cpu_count()

        print("Using {} worker processes.".format(workers))

        # Start the child workers running
        self.readers = []
        self.writers = []
        self.processes = []
        for _ in range(workers):
            read, write = mp.Pipe(duplex=False)
            p = mp.Process(target=self.task, args=(chunkdatasrc(), write))
            self.processes.append(p)
            p.start()
            self.readers.append(read)
            self.writers.append(write)


    def shutdown(self):
        """
        Terminates all the workers
        """
        for i in range(len(self.readers)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.readers[i].close()
            self.writers[i].close()


    def task(self, chunkdatasrc, writer):
        """
        Run in fork'ed process, read data from chunkdatasrc
        and send through pipe back to main process.
        """
        for item in chunkdatasrc:
            writer.send_bytes(item)


    def __iter__(self):
        """
        Read v3 records from child workers, shuffle, and yield
        records.
        """
        sbuff = sb.ShuffleBuffer(self.struct_size, self.shuffle_size)
        while len(self.readers):
            for r in mp.connection.wait(self.readers):
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s
