#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
import numpy as np
import time
from numpy.linalg import norm
from collections import Counter


def cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dot_prod = np.dot(v1, v2)
    return dot_prod / (norm(v1) * norm(v2))


def loadEmbedding(filename, list_words):
    """
    Read the file <filename> and generate the embedding matrix. Only load
    embeddings of words in <list_words>. There is no reason to load the
    embedding of a word if we are not going to do computation with it.
    """

    # Read the file to get the number of words and the embedding dimension
    print("   Reading \"{}\" to get the dimension and the number of"
          " words ... ".format(filename), end="")
    nb_word = 0
    with open(filename) as f:
        first_line = f.readline().split()
        word, nb_dims = first_line[0], len(first_line[1:])

        if word in list_words:
            nb_word += 1

        # for each line, add 1 to nb_word if word is in list_words
        for line in f:
            if line.split()[0] in list_words:
                nb_word += 1

    print("Done.\n   Loading {} embeddings of dimension"
          " {} ... ".format(nb_word, nb_dims), end="")

    # each row is the embedding of a word
    embedding = np.zeros((nb_word, nb_dims))

    # dictionaries to map each word and their respective index
    numToWords, wordsToNum = {}, {}

    # Read again the file to extract embeddings and put them into the matrix
    with open(filename) as f:
        idx = 0
        for line in f:
            line = line.split()
            word, vals = line[0], list(map(float, line[1:]))

            # add embedding only if the word is in list_words
            if word in list_words:
                embedding[idx] = vals
                numToWords[idx] = word
                wordsToNum[word] = idx
                idx += 1

    print("Done.")
    print("   Normalizing the embeddings ... ", end="")

    # norm(., axis=1) gives the norm of each rows. It is an array with
    # dimension (n, ). To divide each coefficicent of embedding with the
    # corresponding norm, we need to reshape the array to (n, 1)
    embedding = embedding / norm(embedding, axis=1)[:, np.newaxis]

    print("Done.")

    return embedding, numToWords, wordsToNum


def generate_pairs(definition_fn, embedding_fn, strg_fn, weak_fn, K):
    """
    Generate weak and strong pairs of words based on definitions in
    defs_fn. A and B are a strong pair if :
        - A is in definition of B
        - B is in definition of A
    All others pairs of words (ie a word and a word from its definition)
    are considered as a weak pair.

    """

    # load all words and their definitions.
    print("-- Loading definitions from \"{}\"".format(definition_fn))
    print("   Reading file ... ", end="")

    dictionary = {}
    uniq_words = set() # list of all words in definition_file

    with open(definition_fn) as f:
        for line in f:
            ar = line.strip().split()
            # use Counter to take into account the number of occurence
            # of each definition words
            word, definition_words = ar[0], Counter(ar[1:])
            dictionary[word] = definition_words

            uniq_words.add(word)
            uniq_words.update(set(ar[1:]))

    print("Done.")
    print("   Entries in \"{}\":\t{}".format(definition_fn, len(dictionary)))
    print("   Uniq words in \"{}\":\t{}".format(definition_fn, len(uniq_words)))


    # load pre-existing embeddings.
    print("\n-- Loading embedding from \"{}\"".format(embedding_fn))
    embedding, numToWords, wordsToNum = loadEmbedding(embedding_fn, uniq_words)


    # generate strong and weak pairs
    print("\n-- Generating strong and weak pairs")
    weak, strong = set(), set()

    nb_words_done = 0;
    for word in dictionary:
        nb_words_done += 1
        if nb_words_done % 100 == 0:
            progress = nb_words_done / len(dictionary) * 100
            print("\r", "{:.2f}%".format(progress), end="")

        for definition_token in dictionary[word]:

            # case 0: word is used in its definition. Obvious strong pair,
            # but not interesting.
            if word == definition_token:
                continue

            # case 1: strong pair
            # Some words (like eurynome) are in vocabulary, and are used in
            # some definitions, but do not have a definition themselves. So
            # we need to be sure that definition_token is in the dictionary.
            if definition_token in dictionary and \
               word in dictionary[definition_token]:

                # use alphabetical order -> no duplicate
                w1, w2 = min(word, definition_token), max(word,definition_token)
                if not (w1,w2) in strong:
                    strong.add((w1,w2))

                # |- Artificial strong pairs generation -|
                if K > 0:

                    # to create more strong pairs, we need the embedding of
                    # definition_token. If it does not exist, can't do anything
                    if not definition_token in wordsToNum:
                        continue

                    embed_def_token = embedding[wordsToNum[definition_token]]

                    # To generate K other strong pairs, we need to find the K
                    # closest word to definition_token. Then we can create the
                    # pairs :
                    #   * (word, closest_1)
                    #   * (word, closest_2)
                    #   * ...
                    #   * (word, closest_K)
                    #
                    # Instead of taking each row of the embedding matrix and
                    # computing the cosine similarity with embed_def_token and
                    # take the K best scores, we do the dot product between the
                    # embedding matrix and embed_def_token. Because our
                    # embedding matrix is normalized, we'll get a vector
                    # containing all cosine similarities. Then we only need to
                    # find the K indexes of the maximum scores with the
                    # argpartition function. But when we compute the dot product
                    # between the matrix and the vector, we'll compute the dot
                    # product between embed_def_token and itself (hence getting
                    # a cosine sim of 1). So we need to get the K+1 best scores
                    # of similarities.

                    #start = time.time()
                    cosine_sim = embedding.dot(embed_def_token)
                    max_indexes = np.argpartition(cosine_sim, -(K+1))[-(K+1):]
                    #duree = time.time() - start
                    #print("duree:", duree, "\n")

                    for index in max_indexes:
                        close_word = numToWords[index]

                        if close_word != definition_token:
                            #print(close_word, "\t", cosine_sim[index])
                            w1, w2 = min(word,close_word), max(word,close_word)
                            strong.add((w1,w2))

            # case 2: weak pair
            else:
                w1, w2 = min(word, definition_token), max(word,definition_token)
                if not (w1,w2) in weak:
                    weak.add((w1,w2))


    # write pairs into files
    print("\n\n-- Writing pairs")
    strg_of = open("{}-K{}.txt".format(strg_fn, K), "w")
    weak_of = open("{}-K{}.txt".format(weak_fn, K), "w")

    for s in strong:
        strg_of.write(' '.join(s) + '\n')

    for s in weak:
        weak_of.write(' '.join(s) + '\n')

    strg_of.close()
    weak_of.close()

    total = (len(strong) + len(weak)) / 100.0
    print("   # strong pairs: % 8d (%.2f%%)" % (len(strong), len(strong)/total))
    print("   # weak   pairs: % 8d (%.2f%%)" % (len(weak), len(weak)/total))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
             description = "Strong & weak pairs generator from definitions.",
             )

    parser.add_argument('-d', '--definitions', help="""File containing word
                        definitions.""", required=True)
    parser.add_argument('-e', '--embedding', help="""File containing words
                        embeddings. The script is able to determine the number
                        of words and the dimension automatically.""",
                        required=True)
    parser.add_argument('-sf', '--strong-file', help="""Filename where the
                        strong pairs will be saved (default: strong-pairs).""",
                        default="strong-pairs")
    parser.add_argument('-wf', '--weak-file', help="""Filename where the
                        weak pairs will be saved (default: weak-pairs).""",
                        default="weak-pairs")
    parser.add_argument('-K', help="""Number of artificially generated strong
                        pairs for each real strong pairs (default: 5).""",
                        default=5, type=int)
    args = parser.parse_args()

    generate_pairs(args.definitions,
                   args.embedding,
                   args.strong_file,
                   args.weak_file,
                   args.K
                  )
