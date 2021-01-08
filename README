         Dict2vec : Learning Word Embeddings using Lexical Dictionaries
         ==============================================================

CONTENT

	1. PREAMBLE
	2. ABOUT
	3. REQUIREMENTS
	4. USAGE
	  1. Train word embeddings
	  2. Evaluate word embeddings
	  3. Download Dict2vec pre-trained word embeddings
	  4. Download Wikipedia training corpora and dictionary definitions
	5. AUTHOR
	6. COPYRIGHT
                         ------------------------------

1. PREAMBLE

	This work  is  one  of  my  contributions  of  my  PhD  thesis  entitled
	"Improving methods to learn word representations for efficient  semantic
	similarities computations" in which  I  propose  new  methods  to  learn
	better word  embeddings.   You  can  find  and  read  my  thesis  freely
	available at https://github.com/tca19/phd-thesis.

2. ABOUT

	This repository contains source code to train word embeddings  with  the
	Dict2vec model, which uses both  Wikipedia  and  dictionary  definitions
	during training.  It also contains  scripts  to  evaluate  learned  word
	embeddings (trained with Dict2vec or  any  other  method),  to  download
	Wikipedia training corpora, to fetch dictionary definitions from  online
	dictionaries and to generate strong and weak pairs from the definitions.
	Related  paper  describing  the  Dict2vec  model   can   be   found   at
	https://www.aclweb.org/anthology/D17-1024/.

	If you use this repository, please cite:

	@inproceedings{tissier2017dict2vec,
	  title     = {Dict2vec : Learning Word Embeddings using Lexical Dictionaries},
	  author    = {Tissier, Julien and Gravier, Christophe and Habrard, Amaury},
	  booktitle = {Proceedings of the 2017 Conference on Empirical Methods
	               in Natural Language Processing},
	  month     = {sep},
	  year      = {2017},
	  address   = {Copenhagen, Denmark},
	  publisher = {Association for Computational Linguistics},
	  url       = {https://www.aclweb.org/anthology/D17-1024},
	  doi       = {10.18653/v1/D17-1024},
	  pages     = {254--263},
	}

3. REQUIREMENTS

	To compile and run the Dict2vec  model,  you  will  need  the  programs:
	  - gcc (4.8.4 or newer)
	  - make

	To evaluate the learned embeddings on  the  word  similarity  task,  you
	will need to install on you system:
	  - python3
	  - numpy (python3 version)
	  - scipy (python3 version)

	To fetch definitions from online dictionaries, you will need to  install
	on your system:
	  - python3

	To run demo scripts and download training data, you  will  also  need  a
	system   with   `wget`,   `bzip2`,   `perl`   and   `bash`    installed.

4. USAGE

	1. Train word embeddings
	------------------------
	Before running the example script, open  the  file  `demo-train.sh`  and
	modify the line 62 so the variable THREADS is equal  to  the  number  of
	cores in your machine.  By default, it is equal to 8, so if your machine
	only has 4 cores, update it to be:

	THREADS=4

	Run `demo-train.sh` to have a quick  glimpse of  Dict2vec  performances.

	./demo-train.sh

	This will:
	  - download a training file of 50M words
	  - download strong and weak pairs for training
	  - compile Dict2vec source code into a binary executable
	  - train word embeddings with a dimension of 100
	  - evaluate the embeddings on 11 word similarity datasets

	To directly compile the  code  and  interact  with  the  sotfware,  run:

	make && ./dict2vec

	Full documentation of each possible parameters is displayed when you run
	`./dict2vec` without any arguments.

	2. Evaluate word embeddings
	---------------------------
	Run  `evaluate.py`  to  evaluate  trained  word  embeddings.   Once  the
	evaluation is done, you get something like this:

	./evaluate.py embeddings.txt

	Filename        | AVG  | MIN  | MAX  | STD  | Missed words/pairs
	=================================================================
	Card-660.txt    | 0.598| 0.598| 0.598| 0.000|      33% / 50%
	MC-30.txt       | 0.861| 0.861| 0.861| 0.000|       0% / 0%
	MEN-TR-3k.txt   | 0.746| 0.746| 0.746| 0.000|       0% / 0%
	MTurk-287.txt   | 0.648| 0.648| 0.648| 0.000|       0% / 0%
	MTurk-771.txt   | 0.675| 0.675| 0.675| 0.000|       0% / 0%
	RG-65.txt       | 0.860| 0.860| 0.860| 0.000|       0% / 0%
	RW-STANFORD.txt | 0.505| 0.505| 0.505| 0.000|       1% / 2%
	SimLex999.txt   | 0.452| 0.452| 0.452| 0.000|       0% / 0%
	SimVerb-3500.txt| 0.417| 0.417| 0.417| 0.000|       0% / 0%
	WS-353-ALL.txt  | 0.725| 0.725| 0.725| 0.000|       0% / 0%
	WS-353-REL.txt  | 0.637| 0.637| 0.637| 0.000|       0% / 0%
	WS-353-SIM.txt  | 0.741| 0.741| 0.741| 0.000|       0% / 0%
	YP-130.txt      | 0.635| 0.635| 0.635| 0.000|       0% / 0%
	-----------------------------------------------------------------
	W.Average       | 0.570

	The script computes the Spearman's rank correlation score for some  word
	similarity datasets, as well as the OOV rate for each  dataset  and  the
	weighted average based on the number of pairs evaluated on each dataset.
	We provide the following evaluation datasets in `data/eval/`:
	  - Card-660     (Pilehvar et al., 2018)
	  - MC-30        (Miller and Charles, 1991)
	  - MEN          (Bruni et al., 2014)
	  - MTurk-287    (Radinsky et al., 2011)
	  - MTurk-771    (Halawi et al., 2012)
	  - RG-65        (Rubenstein and Goodenough, 1965)
	  - RW           (Luong et al., 2013)
	  - SimLex-999   (Hill et al., 2014)
	  - SimVerb-3500 (Gerz et al., 2016)
	  - WordSim-353  (Finkelstein et al., 2001)
	  - YP-130       (Yang and Powers, 2006)

	This script is also able to evaluate several  embeddings  files  at  the
	same time, and compute  the  average  score  as  well  as  the  standard
	deviation. To evaluate several embeddings, simply add multiple filenames
	as arguments:

	./evaluate.py embedding-1.txt embedding-2.txt embedding-3.txt

	The evaluation script indicates:
	  - AVG: the average score of all embeddings for each dataset
	  - MIN: the minimum score of all embeddings for each dataset
	  - MAX: the maximum score of all embeddings for each dataset
	  - STD: the standard deviation score of all embeddings for each dataset

	When you evaluate only  one  embedding,  you  get  the  same  value  for
	AVG/MIN/MAX and a standard deviation STD of 0.


	3. Download Dict2vec pre-trained word embeddings
	------------------------------------------------
	We provide word embeddings trained with the Dict2vec model on  the  July
	2017 English version of Wikipedia.  Vectors with  dimension  100  (resp.
	200) were trained on the first 50M (resp.  200M) words  of  this  corpus
	whereas vectors with dimension 300 were  trained  on  the  full  corpus.
	First line is composed of (number of words / dimension).  Each following
	line contains the word  and  all  its  space  separated  vector  values.
	If you use these word embeddings, please cite the paper as explained  in
	section "2. ABOUT".
	  - dimension 100 [https://mega.nz/file/Y0RmyI5S#SlupdHC2R7wMpHYWhaN9wYEKxsxEmZO_7Z-64hHnwqM]
	  - dimension 200 [https://mega.nz/file/UowxyBKA#nbiP5Os6GXmk-dGFEZkuj4aS0Uewcd81Z2NWGvcc460]
	  - dimension 300 [https://mega.nz/file/Et53UJrB#O4TAagLBgrBRnEi2liWzhOHuAaVsxUqKRfARYgK_n4o]

	You need to extract the embeddings before using them.  Use the following
	command to do so:

	tar xvjf dict2vec100.tar.bz2

	4. Download Wikipedia training corpora and dictionary definitions
	-----------------------------------------------------------------
	For Wikipedia corpora, you can generate the same 3 files (50M, 200M  and
	full) we  use for  training  in  the  paper  by  running `./wiki-dl.sh`.

	This script will download the full English  Wikipedia  dump  of  January
	2021, uncompress it and directly feed it into  Mahoney's  parser  script
	[1].  It also cuts the  entire  dump  into  two  smaller  datasets:  one
	containing  the  first  50M  tokens  (enwiki-50M),  and  the  other  one
	containing the first 200M tokens (enwiki-200M).   The  training  corpora
	have the following filesizes:
	  - enwiki-50M: 296MB
	  - enwiki-200M: 1.16GB
	  - enwiki-full: 29.5GB

	[1] http://mattmahoney.net/dc/textdata#appendixa

	For dictionary  definitions,  we  provide  scripts  to  download  online
	definitions and generate strong/weak pairs based on  these  definitions.
	More information and full documentation  can  be  found  in  the  folder
	dict-dl/ of this repository.

5. AUTHOR

	Written  by  Julien  Tissier  <30314448+tca19@users.noreply.github.com>.

6. COPYRIGHT

	This software is licensed under the GNU GPLv3 license.  See the  LICENSE
	file for more details.
