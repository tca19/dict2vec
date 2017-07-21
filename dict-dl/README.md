dict-dl
=======

dict-dl is a tool from Dict2vec to download online lexical word definitions, and
generate strong and weak pairs from these definitions.  The dictionaries used to
download definitions are :

  * [Cambridge](http://dictionary.cambridge.org/dictionary/english/)
  * [dictionary.com](http://www.dictionary.com/)
  * [Oxford](https://en.oxforddictionaries.com/)
  * [Collins](http://www.collinsdictionary.com/dictionary/english/)


Requirements
------------

To use scripts in this folder, you will need :

  * python3
  * numpy (python3 version)


Download definitions
--------------------

To download online definitions, you first need a file containing a list of words
you want to download the definitions. This file needs to contain one word per
line. We provide an example with the file 1000-words.txt.

Then run `download_definitions.txt` with the filename of the file containing the
list of words, like :

```bash
$ ./download_definitions.py 1000-words.txt
```

This will write all the fetched definitions in the file
1000-words-definitions.txt. Each line will look like this :

```
            XXX     word    def1 def2 def3 def4 def5 def6
             ^       ^      \___________________________/
             1       2                   3

        1: code representing the dictionary the definition comes from.

                          Cam -> Cambridge
                          Dic -> dictionary.com
                          Oxf -> Oxford
                          Col -> Collins

        2: the fetched word.
        3: words from the definition. Stopwords have been removed. All words
           are lowercased.
```

It is possible that you see :

```
ERROR: * timeout error.
       * retry Collins - natural
```

during the execution of the script. This means the server was not able to
respond to the HTTP request, so we can not have the definition for this word.
The error message tells you the word and the dictionary that were faulty, so
you can redownload the definition manually. This error happens rarely and is
only dependent on the web server load.


Clean definitions
-----------------

Fetched definitions are not regrouped and can contain words that are not in
the vocabulary. To clean the definitions (remove words we don't want and
regroup the 4 definitions together for each word), run :

```bash
$ ./clean_definitions.py -d 1000-words-definitions.txt -v vocab.txt
```

The options are :

  * `-d <file> : ` <file> contain the fetched definitions from previous script
  * `-v <file> : ` <file> is a list of words you want to keep (the vocabulary)

This will produce a file named all-definitions-cleaned.txt where the first
word of each line is the fetched word, and the rest of the line are the words
from all its definitions.


Generate strong/weak pairs
--------------------------

To generate strong and weak pairs, you need a pre-trained word embeddings to
compute the K closest neighbours (used to form additional strong pairs). Then
run :

```bash
$ ./generate_pairs.py -d all-definitions-cleaned.txt -e vectors.vec -K 5
```

The options are :

  * `-d <file> : ` <file> contain the cleaned definitions from previous script
  * `-e <file> : ` <file> is the word embeddings to use to compute K closest neighbours
  * `-K <int>  : ` <int> is the number of neighbours to use for additional pairs
  * `[-sf <file>] :` [optional] <file> is the output filename for strong pairs
  * `[-wf <file>] :` [optional] <file> is the output filename for weak pairs

