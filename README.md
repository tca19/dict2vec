Dict2vec
========

Dict2vec is a framework to learn word embeddings using lexical dictionaries.

Requirements
------------

To compile and run our Dict2vec model, you will need :

  * gcc (4.8.4 or newer)
  * make

To evaluate the learned embeddings on the word similarity task, you will need :

  * python3
  * numpy (python3 version)
  * scipy (python3 version)

To fetch definitions from online dictionaries, you will need :

  * python3

To run demo scripts and download training data, you will also need a machine
with **wget**, **bzip2**, **perl** and **bash** installed.

Run the code
------------

Before running the example script, open `demo-train.sh` and modify the line 62
so the variable THREADS is equal to the number of cores in your machine. By
default, it is equal to 8, so if your machine only has 4 cores, update it to be
:

```
THREADS=4
```

Then run `demo-train.sh` to have a quick glimpse of Dict2vec performances.

```bash
$ ./demo-train.sh
```

This will :

  * download a training file of 50M words
  * download strong and weak pairs for training
  * compile Dict2vec source code into a binary executable
  * train word embeddings with a dimension of 100
  * evaluate the embeddings on 11 word similarity datasets

To directly compile the code and interact with the sotfware, run the produced
binary without any arguments. Full documentation and description of each
possible parameters are available [here](doc/dict2vec-documentation.md).

```bash
$ make
$ ./dict2vec
```

Evaluate word embeddings
------------------------

Run `evaluate.py` to evaluate a trained word embedding. Once the evaluation is
done, you get something like this :

```bash
$ ./evaluate.py embeddings.txt
Filename        | AVG  | MIN  | MAX  | STD  | oov
===================================================
MC-30.txt       | 0.848| 0.848| 0.848| 0.000|  0%
MEN-TR-3k.txt   | 0.726| 0.726| 0.726| 0.000|  0%
MTurk-287.txt   | 0.659| 0.659| 0.659| 0.000|  0%
MTurk-771.txt   | 0.684| 0.684| 0.684| 0.000|  0%
RG-65.txt       | 0.833| 0.833| 0.833| 0.000|  0%
RW-STANFORD.txt | 0.503| 0.503| 0.503| 0.000|  36%
SimVerb-3500.txt| 0.399| 0.399| 0.399| 0.000|  3%
WS-353-ALL.txt  | 0.743| 0.743| 0.743| 0.000|  0%
WS-353-REL.txt  | 0.670| 0.670| 0.670| 0.000|  0%
WS-353-SIM.txt  | 0.763| 0.763| 0.763| 0.000|  0%
YP-130.txt      | 0.630| 0.630| 0.630| 0.000|  3%
---------------------------------------------------
W.Average       | 0.577
```

The script computes the Spearman's rank correlation score for some word
similarity datasets, as well as the OOV rate for each dataset and the weighted
average based on the number of pairs evaluated on each dataset. We provide the
evaluation datasets in `data/eval/`.

  * MC-30        (Miller and Charles, 1991)
  * MEN          (Bruni et al., 2014)
  * MTurk-287    (Radinsky et al., 2011)
  * MTurk-771    (Halawi et al., 2012)
  * RG-65        (Rubenstein and Goodenough, 1965)
  * RW           (Luong et al., 2013)
  * SimVerb-3500 (Gerz et al., 2016)
  * WordSim-353  (Finkelstein et al., 2001)
  * YP-130       (Yang and Powers, 2006)

This script is also able to evaluate several embeddings files at the same time,
and compute the average score as well as the standard deviation. To evaluate
several embeddings, simply add the filenames as arguments :

```bash
$ ./evaluate.py embedding-1.txt embedding-2.txt embedding-3.txt
```

This script will report :

  * AVG : the average score of all embeddings for each dataset
  * MIN : the minimum score of all embeddings for each dataset
  * MAX : the maximum score of all embeddings for each dataset
  * STD : the standard deviation score of all embeddings for each dataset

When you evaluate only one embedding, you get the same value for AVG/MIN/MAX and
a standard deviation STD of 0.


Download pre-trained vectors
---------------------------

We provide word embeddings trained with the Dict2vec model on the July 2017
English version of Wikipedia. Vectors with dimension 100 (resp. 200) were
trained on the first 50M (resp. 200M) words of this corpus whereas vectors with
dimension 300 were trained on the full corpus. First line is composed of (number
of words / dimension). Each following line contain the word and all its space
separated vector values.

If you use these word embeddings, please cite the paper as explained in section
[Cite this paper](#cite-this-paper).

  * [dimension 100](https://s3.us-east-2.amazonaws.com/dict2vec-data/dict2vec100.tar.bz2) (85MB)
  * [dimension 200](https://s3.us-east-2.amazonaws.com/dict2vec-data/dict2vec200.tar.bz2) (354MB)
  * [dimension 300](https://s3.us-east-2.amazonaws.com/dict2vec-data/dict2vec300.tar.bz2) (4.3GB)


Download more data
------------------

### Definitions

We provide scripts to download online definitions and generate strong/weak pairs
based on these definitions. More information and full documentation can be found
[here](dict-dl/).

### Wikipedia

You can generate the same 3 files (50M, 200M and full) we use for training in
the paper by running the script `wiki-dl.sh`.

```bash
$ ./wiki-dl.sh
```

This script will download the full English Wikipedia dump of July 2017,
uncompress it and directly feed it into [Mahoney's parser
script](http://mattmahoney.net/dc/textdata#appendixa). It also cuts the entire
dump into two smaller datasets : one containing the first 50M tokens
(enwiki-50M), and the other one containing the first 200M tokens (enwiki-200M).
We report the following filesizes :

  * enwiki-50M  : 283MB
  * enwiki-200M : 1.1GB
  * enwiki-full : 23GB


Cite this paper
---------------

Please cite this paper if you use our code to learn word embeddings or download
definitions or use our pre-trained word embeddings.

J. Tissier, C. Gravier, A. Habrard, *Dict2vec : Learning Word Embeddings using
Lexical Dictionaries*

```
@inproceedings{tissier2017dict2vec,
  title     = {Dict2vec : Learning Word Embeddings using Lexical Dictionaries},
  author    = {Tissier, Julien and Gravier, Christophe and Habrard, Amaury},
  booktitle = {Proceedings of EMNLP},
  year      = {2017}
}
```


License
-------

This project is licensed under the GNU GPL v3 license. See the
[LICENSE](LICENSE) file for details.
