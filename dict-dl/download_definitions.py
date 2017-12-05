#!/usr/bin/env python3
#
# Copyright (c) 2017-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of Dict2vec.
#
# Dict2vec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Dict2vec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.

from queue import Queue
from threading import Thread, Lock
from multiprocessing import cpu_count
from downloader import *
from os.path import splitext
import argparse
import time
import sys

# global variables used (and shared) by all ThreadDown instances
exitFlag = 0
counterLock = Lock()
request_counter  = {"Cam": 0, "Dic": 0, "Col": 0, "Oxf": 0}
download_counter = {"Cam": 0, "Dic": 0, "Col": 0, "Oxf": 0}

class ThreadDown(Thread):
    """Class representing a thread that download definitions."""
    def __init__(self, dict_name, data_queue, res_queue):
        Thread.__init__(self)
        self.dict_name  = dict_name
        self.data_queue = data_queue
        self.res_queue  = res_queue

    def run(self):
        while not exitFlag:
            if not self.data_queue.empty():
                word = self.data_queue.get()
                result = download_word_definition(self.dict_name, word)
                counterLock.acquire()
                request_counter[self.dict_name] += 1
                counterLock.release()

                if len(result) > 0:
                    # if len > 0, the downloaded definition contains at least
                    # one word, we can add 1 to the number of downloads
                    counterLock.acquire()
                    download_counter[self.dict_name] += 1
                    counterLock.release()

                    # then add the fetched definition, the word and the
                    # dictionary used as a message for ThreadWrite
                    self.res_queue.put("{} {} {}".format(self.dict_name, word,
                                                        " ".join(result)))

class ThreadWrite(Thread):
    """Class representing a thread that write definitions to a file."""
    def __init__(self, filename, msg_queue):
        Thread.__init__(self)
        self.msg_queue = msg_queue
        self.of = open(filename, "w")

    def run(self):
        while not exitFlag:
            if not self.msg_queue.empty():
                msg = self.msg_queue.get()
                self.of.write(msg + "\n")

        while True:
            try:
                msg = self.msg_queue.get(True, 5)
                self.of.write(msg + "\n")
            except:
                break

        self.of.close()

def main(filename):
    # 0. to measure download time + use global to be able to modify exitFlag
    globalStart = time.time()
    global exitFlag


    # 1. load all words from the filename argument
    vocabulary = {}
    with open(filename) as f:
        for line in f:
            if not line.strip() in vocabulary:
                vocabulary[line.strip()] = []

    vocabulary_size = len(vocabulary)
    # add "-definitions" before the file extension to create output filename
    output_fn = splitext(filename)[0] + "-definitions.txt"
    print("Writing definitions in", output_fn)


    # 2. create queues containing all words to fetch (1 queue per dictionary)
    queue_Cam = Queue() # no init val -> infinite size
    queue_Dic = Queue()
    queue_Col = Queue()
    queue_Oxf = Queue()
    queue_msg = Queue()

    for w in vocabulary:
        queue_Cam.put(w)
        queue_Dic.put(w)
        queue_Col.put(w)
        queue_Oxf.put(w)


    # 3. create threads
    threads = []
    thread_writer = ThreadWrite(output_fn, queue_msg)
    thread_writer.start()
    threads.append(thread_writer)

    # in my case, fastest fetching was achieved with 12 threads per core,
    # but since there are 4 types of threads (1 for each dictionary), it
    # gives a number of thread per type equal to :
    # (NB_CORE * 12 ) / 4 = NB_CORE * 3
    NB_THREAD = cpu_count() * 3

    # start all the download threads.
    for x in range(NB_THREAD):
        thread_Cam = ThreadDown("Cam", queue_Cam, queue_msg)
        thread_Cam.start()
        thread_Dic = ThreadDown("Dic", queue_Dic, queue_msg)
        thread_Dic.start()
        thread_Col = ThreadDown("Col", queue_Col, queue_msg)
        thread_Col.start()
        thread_Oxf = ThreadDown("Oxf", queue_Oxf, queue_msg)
        thread_Oxf.start()

        threads.append(thread_Cam)
        threads.append(thread_Dic)
        threads.append(thread_Col)
        threads.append(thread_Oxf)


    # 4. wait until all queues are empty, show progress, terminate all threads
    percent = 0
    while not queue_msg.empty() or \
          not queue_Cam.empty() or \
          not queue_Dic.empty() or \
          not queue_Col.empty() or \
          not queue_Oxf.empty():

        # the expected number of fetched words is the number of words
        # in the vocabulary times 4 because we are using 4 dictionaries.
        # The number of fetched words is simply the sum of each counter.
        tmp = sum(request_counter.values()) / (4.0 * vocabulary_size) * 100
        tmp = int(tmp) + 1
        if tmp != percent:
            print('\r{0}%'.format(tmp), end="")
            percent = tmp

    exitFlag = 1
    # we only wait thread_writer to join because this is the most important
    # thread in the code and we are only interested in what the program
    # is writing. And sometimes, some ThreadDown are stuck in a
    # connection timeout and cause the entire code to freeze and never
    # terminate.
    print("\nWaiting thread_writer to join.")
    thread_writer.join()


    # 5. get total time and some results infos.
    print("Total time: {:.2f} sec\n".format(time.time() - globalStart))
    print("S T A T S (# successful download / # requests)")
    print("==============================================")
    for dic in sorted(request_counter.keys()):
        print("{}   {}/{}  ({:.1f}%)".format(dic, download_counter[dic],
          request_counter[dic], download_counter[dic] * 100 / request_counter[dic]))

    print("\nVocabulary size:", vocabulary_size)
    print("Results written in", output_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("list_words", metavar="list-words",
           help="file containing a list of words to download definitions")
    args = parser.parse_args()

    print(args)
    main(args.list_words)
