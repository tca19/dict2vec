#!/usr/bin/env python3

import os
import sys
import math
import argparse
import numpy as np
import scipy.stats as st

FILE_DIR = "data/eval/"
results = dict()
oov     = dict()


def tanimotoSim(v1, v2):
    """Return the Tanimoto similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2 - dotProd)


def cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1) * np.linalg.norm(v2))


def init_results():
    """Read the filename for each file in the evaluation directory"""
    for filename in os.listdir(FILE_DIR):
        if not filename in results:
            results[filename] = []


def evaluate(filename):
    """Compute Spearman rank coefficient for each evaluation file"""

    # step 0 : read the first line to get the number of words and the dimension
    nb_line = 0
    nb_dims = 0
    with open(filename) as f:
        line = f.readline().split()
        nb_line = int(line[0])
        nb_dims = int(line[1])

    mat = np.zeros((nb_line, nb_dims))
    wordToNum = {}
    count = 0

    with open(filename) as f:
        f.readline() # skip first line because it does not contains a vector
        for line in f:
            line = line.split()
            word, vals = line[0], list(map(float, line[1:]))
            mat[count] = np.array(vals)
            wordToNum[word] = count
            count += 1

    # step 1 : iterate over each evaluation data file and compute spearman
    for filename in results:
        found, not_found = 0, 0
        with open(os.path.join(FILE_DIR, filename)) as f:
            file_similarity = []
            embedding_similarity = []
            for line in f:
                w1, w2, val = line.split()
                w1, w2, val = w1.lower(), w2.lower(), float(val)
                if not w1 in wordToNum or not w2 in wordToNum:
                    not_found += 1
                else:
                    found += 1
                    v1, v2 = mat[wordToNum[w1]], mat[wordToNum[w2]]
                    cosine = cosineSim(v1, v2)
                    file_similarity.append(val)
                    embedding_similarity.append(cosine)

                    #tanimoto = tanimotoSim(v1, v2)
                    #file_similarity.append(val)
                    #embedding_similarity.append(tanimoto)

            rho, p_val = st.spearmanr(file_similarity, embedding_similarity)
            results[filename].append(rho)
            oov[filename] = (found, found+not_found)


def stats():
    """Compute statistics on results"""
    title = "{}| {}| {}| {}| {}| {}".format("Filename".ljust(16),
                              "AVG".ljust(5), "MIN".ljust(5), "MAX".ljust(5),
                              "STD".ljust(5), "oov".ljust(5))
    print(title)
    print("="*len(title))

    weighted_avg = 0
    total_found  = 0

    for filename in sorted(results.keys()):
        average = sum(results[filename]) / float(len(results[filename]))
        minimum = min(results[filename])
        maximum = max(results[filename])
        std = sum([(results[filename][i] - average)**2 for i in
                   range(len(results[filename]))])
        std /= float(len(results[filename]))
        std = math.sqrt(std)

        weighted_avg += oov[filename][0] * average
        total_found  += oov[filename][0]

        ratio_oov = 100 - (oov[filename][0] /  oov[filename][1]) * 100

        print("{0}| {1:.3f}| {2:.3f}| {3:.3f}| {4:.3f}|  {5}%".format(
              filename.ljust(16),
              average, minimum, maximum, std, int(ratio_oov)))

    print("-"*len(title))
    print("{0}| {1:.3f}".format("W.Average".ljust(16), 
                                weighted_avg / total_found))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
             description="Evaluate semantic similarities of word embeddings.",
             )

    parser.add_argument('filenames', metavar='FILE', nargs='+',
                        help='Filename of word embedding to evaluate.')

    args = parser.parse_args()

    init_results()
    for f in args.filenames:
        evaluate(f)
    stats()
