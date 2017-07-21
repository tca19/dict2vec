#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from queue import Queue
from threading import Thread, Lock
from multiprocessing import cpu_count
from downloader import *
from os.path import splitext
import argparse
import time
import sys

# STEP 0 : declaration of constants
##

globalStart = time.time()
exitFlag = 0
counterLock = Lock()
counter = {"Cam": 0, "Dic": 0, "Col": 0, "Oxf": 0}

class threadDown(Thread):
    def __init__(self, dict_name, data_queue, res_queue):
        Thread.__init__(self)
        self.dict_name  = dict_name
        self.data_queue = data_queue
        self.res_queue  = res_queue

    def run(self):
        while not exitFlag:
            if not self.data_queue.empty():
                word = self.data_queue.get()
                result = downloadCleaned(self.dict_name, word)
                counterLock.acquire()
                counter[self.dict_name] += 1
                counterLock.release()

                if len(result) > 0:
                    self.res_queue.put("{} {} {}".format(self.dict_name, word,
                                                        " ".join(result)))

class threadWrite(Thread):
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

parser = argparse.ArgumentParser()
parser.add_argument("list_words", metavar="list-words", help="""file containing
                    a list of words to download definitions""")
args = parser.parse_args()

# STEP 1 : load all words from the filename parameter
##

filename = args.list_words
vocabulary = {}
with open(filename) as f:
    for line in f:
        if not line.strip() in vocabulary:
            vocabulary[line.strip()] = []

vocabulary_size = len(vocabulary)
# add "-definitions" before the file extension
output_fn = splitext(filename)[0] + "-definitions.txt"
print("Writing definitions in", output_fn)

# STEP 2 : populate all the queues with words
##

queue_Cam = Queue() # no init val -> infinite size
queue_Dic = Queue()
queue_Col = Queue()
queue_Oxf = Queue()
queue_msg = Queue()
threads = []

for w in vocabulary:
    queue_Cam.put(w)
    queue_Dic.put(w)
    queue_Col.put(w)
    queue_Oxf.put(w)

# STEP 3 : spawn the threads, SPAWN THEM ALL!
##

# Start the writer thread.
thread_writer = threadWrite(output_fn, queue_msg)
thread_writer.start()
threads.append(thread_writer)

# In my case, fastest fetching was achieved with 12 threads per core,
# but since there are 4 types of threads (1 for each dictionary), it
# gives a number of thread per type equal to :
# (NB_CORE * 12 ) / 4 = NB_CORE * 3
NB_THREAD = cpu_count() * 3

# Start all the download threads.
for x in range(NB_THREAD):
    thread_Cam = threadDown("Cam", queue_Cam, queue_msg)
    thread_Cam.start()
    thread_Dic = threadDown("Dic", queue_Dic, queue_msg)
    thread_Dic.start()
    thread_Col = threadDown("Col", queue_Col, queue_msg)
    thread_Col.start()
    thread_Oxf = threadDown("Oxf", queue_Oxf, queue_msg)
    thread_Oxf.start()

    threads.append(thread_Cam)
    threads.append(thread_Dic)
    threads.append(thread_Col)
    threads.append(thread_Oxf)

# STEP 4 : wait until all queues are empty. Show progress.
#          Tell threads show is over.
##

percent = 0

while not queue_msg.empty() or \
      not queue_Cam.empty() or \
      not queue_Dic.empty() or \
      not queue_Col.empty() or \
      not queue_Oxf.empty():

    # The expected number of fetched words is the number of words
    # in the vocabulary times 4 because we are using 4 dictionaries.
    # The number of fetched words is simply the sum of each counter.
    tmp = sum(counter.values()) / (4.0 * vocabulary_size) * 100
    tmp = int(tmp) + 1
    if tmp != percent:
        print('\r{0}%'.format(tmp), end="")
        percent = tmp

exitFlag = 1

# We only wait thread_writer to join because this is the most important
# thread in the code and we are only interested in what the program
# is writing. And sometimes, some threadDown are stuck in a
# connection timeout and cause the entire code to freeze and never
# terminate.
print("\nWaiting thread_writer to join.")
thread_writer.join()
#for t in threads:
#    t.join()

# STEP 5 : get total time and some results infos.
##

print("Total time:", time.time() - globalStart, "sec")
print("Words fetched:", counter)
print("Vocabulary size:", vocabulary_size)
print("Results written in", output_fn)

