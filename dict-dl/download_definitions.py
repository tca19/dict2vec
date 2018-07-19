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
from os.path import splitext, isfile
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
    def __init__(self, dict_name, pos, data_queue, res_queue):
        Thread.__init__(self)
        self.dict_name  = dict_name
        self.pos        = pos # part of speech (noun, verb, adjective or all)
        self.data_queue = data_queue
        self.res_queue  = res_queue

    def run(self):
        while not exitFlag:
            if not self.data_queue.empty():
                word = self.data_queue.get()
                result = download_word_definition(self.dict_name, word,
                                                  self.pos)
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
        self.of = open(filename, "a")

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

def main(filename, pos="all"):
    # 0. to measure download time; use `global` to be able to modify exitFlag
    globalStart = time.time()
    global exitFlag

    # 1. read the file to get the list of words to download definitions
    vocabulary = set()
    with open(filename) as f:
        for line in f:
            vocabulary.add(line.strip())

    vocabulary_size = len(vocabulary)
    # add "-definitions" before the file extension to create output filename.
    # If pos is noun/verb/adjective, add it also to the output filename
    if pos in ["noun", "verb", "adjective"]:
        output_fn = splitext(filename)[0] + "-definitions-{}.txt".format(pos)
    else:
        output_fn = splitext(filename)[0] + "-definitions.txt"
    print("Writing definitions in", output_fn)

    # look if some definitions have already been downloaded. If that's the case,
    # add the words present in output_fn in the aleady_done variable
    already_done = {"Cam": set(), "Dic": set(), "Col": set(), "Oxf": set()}
    if isfile(output_fn): # need to read the file, first test if it exists
        with open(output_fn) as f:
            for line in f:
                line = line.split()
                # line[0] is dictionary name, line[1] the word (already) fetched
                if len(line) < 2:
                    continue
                already_done[line[0]].add(line[1])

    # 2. create queues containing all words to fetch (1 queue per dictionary)
    # The words to download are in queue_{Cam, Dic, Col, Oxf}, the downloaded
    # definitions are pushed in queue_msg
    queue_Cam = Queue() # no init val -> infinite size
    queue_Dic = Queue()
    queue_Col = Queue()
    queue_Oxf = Queue()
    queue_msg = Queue()

    # only add words in queue if they are not already done
    for w in vocabulary:
        if not w in already_done["Cam"]:
            queue_Cam.put(w)
        if not w in already_done["Dic"]:
            queue_Dic.put(w)
        if not w in already_done["Col"]:
            queue_Col.put(w)
        if not w in already_done["Oxf"]:
            queue_Oxf.put(w)

    # 3. create threads
    threads = []
    thread_writer = ThreadWrite(output_fn, queue_msg)
    thread_writer.start()
    threads.append(thread_writer)

    # in my case, fastest fetching was achieved with 12 threads per core,
    # but since there are 4 types of threads (1 for each dictionary), it
    # gives a number of thread per dictionary equal to :
    # (NB_CORE * 12 ) / 4 = NB_CORE * 3
    NB_THREAD = cpu_count() * 3

    # start all the download threads
    for x in range(NB_THREAD):
        thread_Cam = ThreadDown("Cam", pos, queue_Cam, queue_msg)
        thread_Cam.start()
        thread_Dic = ThreadDown("Dic", pos, queue_Dic, queue_msg)
        thread_Dic.start()
        thread_Col = ThreadDown("Col", pos, queue_Col, queue_msg)
        thread_Col.start()
        thread_Oxf = ThreadDown("Oxf", pos, queue_Oxf, queue_msg)
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
        print("{}   {}/{}".format(
              dic, download_counter[dic], request_counter[dic]),
              end="")
        if (request_counter[dic] > 0): # so no division by zero
            print("  ({:.1f}%)".format(
                download_counter[dic] * 100 / request_counter[dic]))
        else:
            print() # add newline if no percentage printed

    print("\nVocabulary size:", vocabulary_size)
    print("Results written in", output_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("list_words", metavar="list-words",
        help="""File containing a list of words (one per line). The script will
        download the definitions for each word.""")
    parser.add_argument("-pos", help="""Either NOUN/VERB/ADJECTIVE. If POS (Part
        Of Speech) is given, the script will only download the definitions that
        corresponds to that POS, not the other ones. By default, it downloads
        the definitions for all POS""", type=str.lower, default="all")
    args = parser.parse_args()

    if args.pos not in ["noun", "verb", "adjective", "all"]:
        print("WARNING: invalid POS argument \"{}\"".format(args.pos))
        print("It can be NOUN, VERB or ADJECTIVE. Using default POS (ALL)\n")
        args.pos = "all"

    main(args.list_words, pos=args.pos)
