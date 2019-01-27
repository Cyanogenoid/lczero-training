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

import utils


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
        self.records_per_worker = 16
        workers = workers or mp.cpu_count()
        self.chunkdatasrc = chunkdatasrc

        print("Using {} worker processes.".format(workers))

        self.output_stack = []

        # Start the child workers running
        self.processes = []
        self.queue = mp.SimpleQueue()
        for i in range(workers):
            p = mp.Process(target=self.task)
            self.processes.append(p)
            p.start()


    def shutdown(self):
        """
        Terminates all the workers
        """
        for process in self.processes:
            process.terminate()
            process.join()


    def task(self):
        """
        Run in fork'ed process, read data from chunkdatasrc
        and send through pipe back to main process.
        """
        chunkdatasrc = self.chunkdatasrc()
        for item in utils.grouper(chunkdatasrc, self.records_per_worker):
            self.queue.put(b''.join(item))


    def __iter__(self):
        """
        Read v3 records from child workers, shuffle, and yield
        records.
        """
        sbuff = sb.ShuffleBuffer(self.shuffle_size)
        while True:
            records = self.queue.get()
            for i in range(self.records_per_worker):
                s = records[i * self.struct_size : (i+1) * self.struct_size]
                s = sbuff.insert_or_replace(s)
                if s is not None:
                    self.output_stack.append(s)
            while self.output_stack:
                yield self.output_stack.pop()
