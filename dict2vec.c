/* Copyright (c) 2017-present, All rights reserved.
 * Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
 *
 * This file is part of Dict2vec.
 *
 * Dict2vec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Dict2vec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License at the root of this repository for
 * more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>      /* strcat */
#include <math.h>
#include <pthread.h>

#define MAXLEN       100
#define MAXLINE      1000

#define SIGMOID_SIZE 1000
#define MAX_SIGMOID  6

#define HASHSIZE     30000000

struct entry
{
	/* Words forming a strong pair with this entry are stored in the array
	 * sp[] (only the index of words are stored). Instead of calculating a
	 * new random index in sp[], the indexes are randomly shuffled at the
	 * beginning and a sliding cursor indicates the current word to draw
	 * (faster because no need to compute a lot of random indexes).
	 * Weak pairs follow the same implementation.
	 */
	int n_sp;       /* number of strong pairs of entry */
	int pos_sp;     /* current cursor position in sp[] */
	int *sp;

	int n_wp;       /* number of weak pairs of entry */
	int pos_wp;     /* current cursor position in wp[] */
	int *wp;


	long  count;    /* number of occurrences of entry in input file */
	char  *word;    /* string associated to the entry */
	float pdiscard; /* probability to discard entry when found in input */
};


char input_file[MAXLEN], output_file[MAXLEN];
char spairs_file[MAXLEN], wpairs_file[MAXLEN];

/* dynamic array containing 1 entry for each word in vocabulary */
struct entry *vocab;

/* default hyperparameters */
int dim = 100, window = 5, min_count = 5, negative = 0,
	strong_draws = 0, weak_draws = 0,
	num_threads = 1, epoch = 1, save_each_epoch = 0;
float   alpha = 0.025, starting_alpha, sample = 0,
	beta_strong = 1.0, beta_weak = 0.25;

/* variables required for processing input file */
long vocab_max_size = 10000, vocab_size = 0, train_words = 0, file_size = 0,
	word_count_actual = 0;


int *vocab_hash;   /* hash table to know index of a word */
float *WI, *WO;    /* weight matrices */
float *sigmoid;    /* precomputed sigmoid values */
int *table;        /* array of indexes for negative sampling */

/* other variables */
clock_t start;
int current_epoch = 0, table_size = 1e7, neg_pos = 0;


/* contains: return 1 if value is inside array. 0 otherwise. */
int contains(int *array, int value, int size)
{
	int j;

	for (j = 0; j < size; ++j)
		if (array[j] == value)
			return 1;
	return 0;
}

/* shuffle: arrange the elements of array in random order. Swap two random
 * cells N times (N is the size of array).
 */
void shuffle(int *array, int size)
{
	int i, j, tmp;

	for (i = 0; i < size - 1; ++i)
	{
		j = i + rand() / (RAND_MAX / (size - i) + 1);
		tmp = array[j];
		array[j] = array[i];
		array[i] = tmp;
	}
}

/* init_negative_table: initialize the negative table used for negative
 * sampling. The table is composed of indexes of words, each one proportional
 * to the number of occurrence of this word.
 */
void init_negative_table()
{
	int i, n_cells, pos;
	float sum, d;

	/* allocate memory for the negative table*/
	table = calloc(table_size, sizeof *table);

	if (table == NULL)
	{
		printf("Cannot allocate memory for the negative table\n");
		exit(1);
	}

	/* compute the sum of count^0.75 for all words */
	for (i = 0, sum = 0.0; i < vocab_size; ++i)
		sum += pow(vocab[i].count, 0.75);

	/* multiply is faster than divide so precompute d = 1/sum */
	for (i = 0, pos = 0, d = 1.0/sum; i < vocab_size; ++i)
	{
		/* compute number of cells reserved for word[i] */
		n_cells = pow(vocab[i].count, 0.75) * table_size * d;

		while (n_cells--)
			table[pos++] = i;
	}

	/* due to rounding point error, it is possible that we inserted less
	 * than table_size values. So update the table_size to the real size. */
	table_size = pos-1;

	/* shuffle the array so we don't have to draw a new random index each
	 * time we want a negative sample, simply keep a position_index, get the
	 * table[position_index] value and increment position_index. This is the
	 * same as drawing a new random index i and get table[i]. */
	shuffle(table, table_size);
}

/* compute_discard_prob: compute the discard probabilty of each word. The
 * probability is defined as: p(w) = 1 - sqrt(t / f(w)) where t is the
 * threshold and f(w) is the frequency of word w. But we store
 * Y = sqrt(t / f(w)). Then we draw a random number X between 0 and 1 and
 * look if X > Y. X has a probability p(w) of being greater than Y.
 * If N is the total number of words, we have:
 *   Y = sqrt(t / ( count(w) / N )) = sqrt(t * N) / sqrt(count(w))
 */
void compute_discard_prob()
{
	int i;
	float w;

	/* precompute sqrt(t * n) */
	w = sqrt(sample * train_words);
	for (i = 0; i < vocab_size; ++i)
		vocab[i].pdiscard = w / sqrt(vocab[i].count);
}

/* read_word: read a single word from a file, assume word boundaries are non
 * letter char. Return -1 if EOF, else return the size of read word.
 */
int read_word(char *word, FILE *fi)
{
	int c, i;
	i = 0;

	/* move until we find a valid letter or EOF */
	while (EOF != (c = fgetc(fi)) && !isalpha(c))
		continue;

	/* add letters to word until EOF or non letter char */
	if (c != EOF)
	{
		do
		{
			word[i++] = c;
			/* reduce i so we won't exceed size of word (which
			 has a size of MAX_STRING) */
			if (i > MAXLEN-1) i--;
		} while (EOF != (c = fgetc(fi)) && isalpha(c));

		word[i] = '\0';
		return i;
	}
	else
		return -1;
}

/* hash: form hash value for string s */
unsigned int hash(char *s)
{
	unsigned int hashval;

	for (hashval = 0; *s != '\0'; ++s)
		hashval = hashval * 257 + *s;
	return hashval % HASHSIZE;
}

/* find: return the position of string s in vocab_hash. If word has never been
 * met, the cell at index hash(word) in vocab_hash will be -1. If the cell is
 * not -1, we start to compare word to each element with the same hash.
 */
unsigned int find(char *s)
{
	unsigned int h = hash(s);

	while (vocab_hash[h] != -1 && strcmp(s, vocab[vocab_hash[h]].word))
		h = (h + 1) % HASHSIZE;
	return h;
}

/* add word to the vocabulary. If word already exists, increment its count */
void add_word(char *word)
{
	unsigned int h = find(word);
	if (vocab_hash[h] == -1)
	{
		/* create new entry */
		struct entry e;
		e.word = malloc(sizeof(char) * (strlen(word)+1));
		strcpy(e.word, word);
		e.count    = 1;
		e.pdiscard = 1.0;
		e.n_sp     = 0;
		e.pos_sp   = 0;
		e.n_wp     = 0;
		e.pos_wp   = 0;
		e.sp       = NULL;
		e.wp       = NULL;

		/* add it to vocab and set its index in vocab_hash */
		vocab[vocab_size] = e;
		vocab_hash[h] = vocab_size++;

		/* reallocate more space if needed */
		if (vocab_size >= vocab_max_size)
		{
			vocab_max_size += 10000;
			vocab = realloc(vocab, vocab_max_size * sizeof(struct entry));
		}
	}
	else
	{
		vocab[vocab_hash[h]].count++;
	}
}

/* compare_words: used to sort two words */
int compare_words(const void *a, const void *b)
{
	return ((struct entry *)b)->count - ((struct entry *)a)->count;
}

/* destroy_vocab: free all memory used to create stong/weak pairs arrays, free
 * memory used to store words (char *) and free the entire array of entries.
 */
void destroy_vocab()
{
	int i;

	for (i = 0; i < vocab_size; ++i)
	{
		free(vocab[i].word);

		if (vocab[i].sp != NULL)
			free(vocab[i].sp);

		if (vocab[i].wp != NULL)
			free(vocab[i].wp);
	}

	free(vocab);
}

/* sort_and_reduce_vocab: sort the words in vocabulary by their number of
 * occurrences. Remove all words with less than min_count occurrences.
 */
void sort_and_reduce_vocab()
{
	int i, valid_words;

	/* sort vocab in descending order by number of word occurrence */
	qsort(vocab, vocab_size, sizeof(struct entry), compare_words);

	/* get the number of valid words (words with count >= min_count) */
	valid_words = 0;
	while (vocab[valid_words++].count >= min_count)
		continue;

	/* because valid_words has been incremented after the condition has
	 * been evaluated to false */
	valid_words--;

	/* remove words with less than min_count occurrences. Strong and weak
	 * pairs have not been added to vocab yet, so no need to free the
	 * allocated memory of strong/weak pairs arrays. */
	for (i = valid_words; i < vocab_size; ++i)
	{
		train_words -= vocab[i].count;
		free(vocab[i].word);
		vocab[i].word = NULL;
	}

	/* resize the vocab array with its new size */
	vocab_size = valid_words;
	vocab = realloc(vocab, vocab_size * sizeof(struct entry));

	/* sorting has changed the index of each word, so update the value
	 in vocab_hash. Start by reseting all values to -1.*/
	for (i = 0; i < HASHSIZE; ++i)
		vocab_hash[i] = -1;
	for (i = 0; i < vocab_size; ++i)
		vocab_hash[find(vocab[i].word)] = i;
}

/* read_strong_pairs; read the file containing the strong pairs. For each pair,
 * add it in the vocab for both words involved.
 */
int read_strong_pairs()
{
	FILE *fi;
	char word1[MAXLEN], word2[MAXLEN];
	int i1, i2, len;

	if ((fi = fopen(spairs_file, "r")) == NULL)
	{
		printf("WARNING: strong pairs data not found!\n"
		       "Not taken into account during learning.\n");
		return 1;
	}

	/* read file until EOF */
	while (read_word(word1, fi) != -1)
	{
		/* there are 2 words per line, so if we were able to read word1,
		 we can read word2 */
		read_word(word2, fi);

		i1 = find(word1);
		i2 = find(word2);

		/* nothing to do if one of the word is not in vocab */
		if (vocab_hash[i1] == -1 || vocab_hash[i2] == -1)
			continue;

		/* get the real indexes (not the index in the hash table) */
		i1 = vocab_hash[i1];
		i2 = vocab_hash[i2];

		/* look if we already have added one strong pair for i1. If not,
		 * create the array. Else, expand by one cell. */
		len = vocab[i1].n_sp;
		if (len == 0)
			vocab[i1].sp = calloc(1, sizeof(int));
		else
			vocab[i1].sp = realloc(vocab[i1].sp, (len+1) * sizeof(int));

		/* add i2 to the list of strong pairs indexes of word1 */
		vocab[i1].sp[len] = i2;
		vocab[i1].n_sp++;

		/*   ---------   */

		/* look if we already have added one pair for i2. If not, create
		 the array. Else, expand by one cell. */
		len = vocab[i2].n_sp;
		if (len == 0)
			vocab[i2].sp = calloc(1, sizeof(int));
		else
			vocab[i2].sp = realloc(vocab[i2].sp, (len+1) * sizeof(int));

		/* add i2 to the list of strong pairs indexes of word1 */
		vocab[i2].sp[len] = i1;
		vocab[i2].n_sp++;
	}

	fclose(fi);
	return 0;
}

/* read_weak_pairs: read the file containing the weak pairs. For each pair, add
 * it in the vocab for both words involved.
 */
int read_weak_pairs()
{
	FILE *fi;
	char word1[MAXLEN], word2[MAXLEN];
	int i1, i2, len;


	if ((fi = fopen(wpairs_file, "r")) == NULL)
	{
		printf("WARNING: weak pairs data not found!\n"
		       "Not taken into account during learning.\n");
		return 1;
	}

	/* read file until EOF */
	while (read_word(word1, fi) != -1)
	{
		/* there are 2 words per line, so if we were able to read word1,
		 we can read word2 */
		read_word(word2, fi);

		i1 = find(word1);
		i2 = find(word2);

		/* nothing to do if one of the word is not in vocab */
		if (vocab_hash[i1] == -1 || vocab_hash[i2] == -1)
			continue;

		/* get the real indexes (not the index in the hash table) */
		i1 = vocab_hash[i1];
		i2 = vocab_hash[i2];

		/* look if we already have added one pair for i1. If not, create
		the array. Else, expand by one cell. */
		len = vocab[i1].n_wp;
		if (len == 0)
			vocab[i1].wp = calloc(1, sizeof(int));
		else
			vocab[i1].wp = realloc(vocab[i1].wp, (len+1) * sizeof(int));

		/* add i2 to the list of weak pairs indexes of word1 */
		vocab[i1].wp[len] = i2;
		vocab[i1].n_wp++;

		/*   ---------   */

		/* look if we already have added one pair for i2. If not, create
		 the array. Else, expand by one cell. */
		len = vocab[i2].n_wp;
		if (len == 0)
			vocab[i2].wp = calloc(1, sizeof(int));
		else
			vocab[i2].wp = realloc(vocab[i2].wp, (len+1) * sizeof(int));

		/* add i1 to the list of weak pairs indexes of word2 */
		vocab[i2].wp[len] = i1;
		vocab[i2].n_wp++;
	}

	fclose(fi);
	return 0;
}

/* read_vocab: read the file given as -input. For each word, either add it in
 * the vocab or increment its occurrence. Also read the strong and weak pairs
 * files if provided. Sort the vocabulary by occurrences and display some infos.
 */
void read_vocab()
{
	FILE *fi;
	int i, failure_strong, failure_weak;
	char word[MAXLEN];

	if ((fi = fopen(input_file, "r")) == NULL)
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	/* init the hash table with -1 */
	for (i = 0; i < HASHSIZE; ++i)
		vocab_hash[i] = -1;

	/* read file until we get -1 (which is EOF) */
	while (read_word(word, fi) != -1)
	{
		/* increment total number of read words */
		train_words++;
		if (train_words % 500000 == 0)
		{
			printf("%ldK%c", train_words / 1000, 13);
			fflush(stdout);
		}

		/* add word we just read or increment its count if needed */
		add_word(word);


		/* Wikipedia has around 8M unique words, so we never hit the
		 * 21M words limit and therefore never need to reduce the vocab
		 * during creation of vocabulary. If your input file contains
		 * much more unique words, you might need to uncomment it. */
		/*if (vocab_size > HASHSIZE * 0.7)
			sort_and_reduce_vocab();*/
	}

	sort_and_reduce_vocab(HASHSIZE);

	printf("Vocab size: %ld\n", vocab_size);
	printf("Words in train file: %ld\n", train_words);

	printf("Adding strong pairs...");
	failure_strong = read_strong_pairs();
	printf("\nAdding weak pairs...");
	failure_weak = read_weak_pairs();
	if (!failure_strong || !failure_weak)
		printf("\nAdding pairs done.\n");

	/* compute the discard probability for each word (only if we
	 * subsample)*/
	if (sample > 0)
		compute_discard_prob();

	/* each thread is assigned a part of the input file. To distribute
	 the work to each thread, we need to know the total size of the file */
	file_size = ftell(fi);
	fclose(fi);
}

/* init_network: initialize matrix WI (random values) and WO (zero values) */
void init_network()
{
	float r, l;
	int i, j;

	if ((WI = malloc(sizeof *WI * vocab_size * dim)) == NULL)
	{
		printf("Memory allocation failed for WI\n");
		exit(1);
	}


	if ((WO = calloc(vocab_size * dim, sizeof *WO)) == NULL)
	{
		printf("Memory allocation failed for WO\n");
		exit(1);
	}

	/* WI is initialized with random values from (-0.5 / vec_dimension)
	 * and (0.5 / vec_dimension). Multiply is faster than divide so
	 * precompute 1 / RAND_MAX and 1 / dim. */
	r = 1.0 / RAND_MAX;
	l = 1.0 / dim;
	for (i = 0; i < vocab_size; ++i)
		for (j = 0; j < dim; ++j)
			WI[i * dim + j] = ( (rand() * r) - 0.5 ) * l;
}

/* destroy_network: free the memory allocated for matrices WI and WO */
void destroy_network()
{
	if (WI != NULL)
		free(WI);

	if (WO != NULL)
		free(WO);
}

void *train_thread(void *id)
{
	FILE *fi;
	char word[MAXLEN];
	int w_t, w_c, c, d, target, line_size, pos, line[MAXLINE];
	int index1, index2, k, half_ws, helper;
	long word_count_local, negsamp_discarded, negsamp_total;
	float label, dot_prod, grad, *hidden;
	double progress, wts, discarded, cps, d_train, lr_coef;

	clock_t now;
	int rnd = (intptr_t) id;

	if ((fi = fopen(input_file, "r")) == NULL)
	{
		printf("ERROR: training data file not found!\n");
		exit(1);
	}

	/* init variables */
	fseek(fi, file_size / num_threads * rnd, SEEK_SET);
	word_count_local = negsamp_discarded = negsamp_total = 0;
	hidden           = calloc(dim, sizeof *hidden);
	half_ws          = window / 2;
	helper           = SIGMOID_SIZE / MAX_SIGMOID / 2;
	wts = discarded  = 0.0f;
	cps              = 1000.0f / CLOCKS_PER_SEC;
	d_train          = 1.0f / train_words;
	lr_coef          = starting_alpha / ((double) (epoch * train_words));

	while (word_count_actual < (train_words * (current_epoch + 1)))
	{
		/* update learning rate and print progress */
		if (word_count_local > 20000)
		{
			alpha -= word_count_local * lr_coef;
			word_count_actual += word_count_local;
			word_count_local = 0;
			now = clock();

			/* "Discarded" is the percentage of discarded negative
			 * samples because they form either a strong or a weak
			 * pair with context word */
			progress = word_count_actual * d_train * 100;
			progress -= 100 * current_epoch;
			wts = word_count_actual / ((double)(now - start) * cps);
			discarded = negsamp_discarded * 100.0 / negsamp_total;
			printf("%clr: %f  Progress: %.2f%%  Words/thread/sec:"
			       " %.2fk  Discarded: %.2f%% ",
			       13, alpha, progress, wts, discarded);
			fflush(stdout);
		}

		/* read MAXLINE words from input file. Add words in line[] if
		 * they are in vocabulary and not discarded. So length of line
		 * might be less than MAXLINE (in practice, length of line is
		 * 500 +/- 50. */
		line_size = 0;
		for (k = MAXLINE; k--;)
		{
			read_word(word, fi);
			w_t = vocab_hash[find(word)];

			/* word is not in vocabulary, move to next one */
			if (w_t == -1)
				continue;

			/* we processed one more word */
			++word_count_local;

			/* discard word or add it in the sentence */
			rnd = rnd * 1103515245 + 12345;
			if (vocab[w_t].pdiscard < (rnd & 0xFFFF) / 65536.0)
				continue;
			else
				line[line_size++] = w_t;
		}

		/* for each word of the line */
		for (pos = half_ws; pos < line_size - half_ws; ++pos)
		{
			w_t = line[pos];  /* central word */

			/* for each word of the context window */
			for (c = pos - half_ws; c < pos + half_ws +1; ++c)
			{
				if (c == pos)
					continue;

				w_c = line[c];
				index1 = w_c * dim;

				/* zero the hidden vector */
				memset(hidden, 0.0, dim * sizeof *hidden);

				/* STANDARD AND NEGATIVE SAMPLING UPDATE */
				for (d = negative+1; d--;)
				{
					/* target is central word */
					if (d == 0)
					{
						target = w_t;
						label = 1.0;
					}

					/* target is random word */
					else
					{
						do
						{
							target = table[neg_pos++];
							if (neg_pos > table_size-1)
								neg_pos = 0;
						} while (target == w_t);

						/* if random word form a strong a weak pair
						 with w_c, move to next one */
						if (contains(vocab[w_c].sp, target,
						             vocab[w_c].n_sp) ||
						    contains(vocab[w_c].wp, target,
						             vocab[w_c].n_wp))
						{
							++negsamp_discarded;
							continue;
						}

						++negsamp_total;
						label = 0.0;
					}


					/* forward propagation */
					index2 = target * dim;
					dot_prod = 0.0;
					for (k = 0; k < dim; ++k)
						dot_prod += WI[index1 + k] * WO[index2 + k];

					if (dot_prod > MAX_SIGMOID)
						grad = alpha * (label - 1.0);
					else if (dot_prod < -MAX_SIGMOID)
						grad = alpha * label;
					else
						grad = alpha * (label - sigmoid[(int)
						       ((dot_prod + MAX_SIGMOID) * helper)]);

					/* back-propagation. 2 for loops is more
					 cache friendly because processor
					 can load the entire hidden array in
					 cache. Using a unique for loop to do
					 the 2 operations is slower. */
					for (k = 0; k < dim; ++k)
						hidden[k] += grad * WO[index2 + k];
					for (k = 0; k < dim; ++k)
						WO[index2 + k] += grad * WI[index1 + k];
				}

				/* POSITIVE SAMPLING UPDATE (strong pairs) */
				for (d = strong_draws; d--;)
				{
					/* can't do anything if no strong pairs
					 */
					if (vocab[w_c].n_sp == 0)
						break;

					if (vocab[w_c].pos_sp > vocab[w_c].n_sp - 1)
						vocab[w_c].pos_sp = 0;
					target = vocab[w_c].sp[vocab[w_c].pos_sp++];

					index2 = target * dim;
					dot_prod = 0;
					for (k = 0; k < dim; ++k)
						dot_prod += WI[index1 + k] * WO[index2 + k];

					/* dot product is already high, nothing to do */
					if (dot_prod > MAX_SIGMOID)
						continue;
					else if (dot_prod < -MAX_SIGMOID)
						grad = alpha * beta_strong;
					else
						grad = alpha * beta_strong *
						    (1 - sigmoid[(int)((dot_prod + MAX_SIGMOID) * helper)]);


					for (k = 0; k < dim; ++k)
						hidden[k] += grad * WO[index2 + k];
					for (k = 0; k < dim; ++k)
						WO[index2 + k] += grad * WI[index1 + k];
				}

				/* POSITIVE SAMPLING UPDATE (weak pairs) */
				for (d = weak_draws; d--;)
				{
					/* can't do anything if no weak pairs */
					if (vocab[w_c].n_wp == 0)
						break;

					if (vocab[w_c].pos_wp > vocab[w_c].n_wp - 1)
						vocab[w_c].pos_wp = 0;
					target = vocab[w_c].wp[vocab[w_c].pos_wp++];

					index2 = target * dim;
					dot_prod = 0;
					for (k = 0; k < dim; ++k)
						dot_prod += WI[index1 + k] * WO[index2 + k];

					if (dot_prod > MAX_SIGMOID)
						continue;
					else if (dot_prod < -MAX_SIGMOID)
						grad = alpha * beta_weak;
					else
						grad = alpha * beta_weak *
						    (1 - sigmoid[(int)((dot_prod + MAX_SIGMOID) * helper)]);

					for (k = 0; k < dim; ++k)
						hidden[k] += grad * WO[index2 + k];
					for (k = 0; k < dim; ++k)
						WO[index2 + k] += grad * WI[index1 + k];
				}

				/* Back-propagate hidden -> input */
				for (k = 0; k < dim; ++k)
					WI[index1 + k] += hidden[k];

			} /* end for each word in the context window */

		}     /* end for each word in line */
	}         /* end while() loop for reading file */

	/* sometimes, progress go over 100% because of rounding float error.
	print a proper 100% progress */
	if (alpha < 0) alpha = 0;
	printf("%clr: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  Discarded:"
	       " %.2f%% ", 13, alpha, 100.0, wts, discarded);
	fflush(stdout);

	fclose(fi);
	free(hidden);
	pthread_exit(NULL);
}

/* save the word vectors in output file. If epoch > 0, add the suffix
 * indicating the epoch */
void save_vectors(char *output, int epoch)
{
	FILE *fo;
	char suffix[15], copy[MAXLEN];
	int i, j;

	if (epoch > 0)
	{
		sprintf(suffix, "-epoch-%d.vec", epoch);
		strcpy(copy, output);
		fo = fopen(strcat(copy, suffix), "w");
	}

	else
		fo = fopen(strcat(output, ".vec"), "w");

	if (fo == NULL)
	{
		printf("Cannot open %s: permission denied\n", output);
		exit(1);
	}

	/* first line is number of vectors + dimension */
	fprintf(fo, "%ld %d\n", vocab_size, dim);

	for (i = 0; i < vocab_size; i++)
	{
		fprintf(fo, "%s ", vocab[i].word);

		for (j = 0; j < dim; j++)
			fprintf(fo, "%.3f ", WI[i * dim + j]);

		fprintf(fo, "\n");
	}

	fclose(fo);
}

/* function called from main */
void train()
{
	/* variable for future use */
	int i;
	pthread_t *threads;

	if ((threads = calloc(num_threads, sizeof *threads)) == NULL)
	{
		printf("Cannot allocate memory for threads\n");
		exit(1);
	}

	/* get words from input file */
	printf("Starting training using file %s\n", input_file);
	read_vocab();

	/* we need to save the starting alpha to update alpha during learning */
	starting_alpha = alpha;

	/* instantiate the network */
	init_network();

	/* instantiate negative table (for negative sampling) */
	if (negative > 0)
		init_negative_table();

	/* train the model for multiple epoch */
	start = clock();
	for (current_epoch = 0; current_epoch < epoch; current_epoch++)
	{
		printf("\n-- Epoch %d/%d\n", current_epoch+1, epoch);

		/* create threads */
		for (i = 0; i < num_threads; i++)
			pthread_create(&threads[i], NULL, train_thread, (void *) (intptr_t) i);

		/* wait for threads to join. When they join, epoch is finished
		 */
		for (i = 0; i < num_threads; i++)
			pthread_join(threads[i], NULL);

		if (save_each_epoch)
		{
			printf("\nSaving vectors for epoch %d.", current_epoch+1);
			save_vectors(output_file, current_epoch+1);
		}

	}

	/* save the file only if we didn't save it earlier with the
	 * save-each-epoch option */
	if (!save_each_epoch)
	{
		printf("\n-- Saving word embeddings\n");
		save_vectors(output_file, -1);
	}

	free(table);
	free(threads);
	destroy_vocab();
}

int arg_pos(char *str, int argc, char **argv)
{
	int a;

	/* goes from 1 to argc-1 because position of option str can't be
	 0 (program name) or last element of argv */
	for (a = 1; a < argc - 1; ++a)
	{
		if (!strcmp(str, argv[a]))
			return a;
	}

	return -1;
}

void print_help()
{
	/* ISO 90 forbids strings longer than 509 characters (including
	 * concatenated strings, so documentation is splitted into multiple
	 * strings. */
	printf(
	"Dict2vec: Learning Word Embeddings using Lexical Dictionaries\n"
	"Author: Julien Tissier <30314448+tca19@users.noreply.github.com>\n\n"
	);

	printf(
	"Options:\n"
	"  -input <file>\n"
	"    Train the model with text data from <file>\n\n"
	"  -strong-file <file>\n"
	"    Add strong pairs data from <file> to improve the model\n\n"
	"  -weak-file <file>\n"
	"    Add weak pairs data from <file> to improve the model\n\n"
	"  -output <file>\n"
	"    Save word embeddings in <file>\n\n"
	);

	printf(
	"  -size <int>\n"
	"    Size of word vectors; default 100\n\n"
	"  -window <int>\n"
	"    Window size for target/context pairs generation; default 5\n\n"
	"  -sample <float>\n"
	"    Value of the threshold t used for subsampling frequent words in\n"
	"    the original word2vec paper of Mikolov; default 1e-4\n\n"
	"  -min-count <int>\n"
	"    Do not train words with less than <int> occurrences; default 5\n\n"
	"  -negative <int>\n"
	"    Number of random words used for negative sampling; default 5\n\n"
	"  -alpha <float>\n"
	"    Starting learning rate; default 0.025\n\n"
	);

	printf(
	"  -strong-draws <int>\n"
	"    Number of strong pairs picked for positive sampling; default 0\n\n"
	"  -weak-draws <int>\n"
	"    Number of weak pairs picked for positive sampling; default 0\n\n"
	"  -beta-strong <float>\n"
	"    Coefficient for strong pairs; default 1.0\n\n"
	"  -beta-weak <float>\n"
	"    Coefficient for weak pairs; default 0.25\n\n"
	"  -threads <int>\n"
	"    Number of threads to use; default 1\n\n"
	"  -epoch <int>\n"
	"    Number of epoch; default 1\n\n"
	"  -save-each-epoch <int>\n"
	"    Save the embeddings after each epoch; 0 (off, default), 1 (on)"
	);

	printf(
	"\n\nUsage:\n"
	"./dict2vec -input data/enwiki-50M -output data/enwiki-50M \\\n"
	"-strong-file data/strong-pairs.txt -weak-file data/weak-pairs.txt \\\n"
	"-size 100 -window 5 -sample 1e-4 -min-count 5 -negative 5 \\\n"
	"-strong-draws 4 -beta-strong 0.8 -weak-draws 5 -beta-weak 0.45 \\\n"
	"-alpha 0.025 -threads 8 -epoch 5 -save-each-epoch 0\n\n"
	);
}

int main(int argc, char **argv)
{
	int i;
	float d;

	/* no arguments given. Print help and exit */
	if (argc == 1)
	{
		print_help();
		return 0;
	}

	/* get the value of each parameter */
	if ((i = arg_pos("-input", argc, argv)) > 0)
		strcpy(input_file, argv[i + 1]);
	else
	{
		printf("Cannot train the model without: -input <file>\n");
		exit(1);
	}

	if ((i = arg_pos("-strong-file", argc, argv)) > 0)
		strcpy(spairs_file, argv[i + 1]);

	if ((i = arg_pos("-weak-file", argc, argv)) > 0)
		strcpy(wpairs_file, argv[i + 1]);

	if ((i = arg_pos("-output", argc, argv)) > 0)
		strcpy(output_file, argv[i + 1]);

	if ((i = arg_pos("-size", argc, argv)) > 0)
		dim = atoi(argv[i + 1]);

	if ((i = arg_pos("-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);

	if ((i = arg_pos("-sample", argc, argv)) > 0)
		sample = atof(argv[i + 1]);

	if ((i = arg_pos("-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);

	if ((i = arg_pos("-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);

	if ((i = arg_pos("-strong-draws", argc, argv)) > 0)
		strong_draws = atoi(argv[i + 1]);

	if ((i = arg_pos("-weak-draws", argc, argv)) > 0)
		weak_draws = atoi(argv[i + 1]);

	if ((i = arg_pos("-beta-strong", argc, argv)) > 0)
		beta_strong = atof(argv[i + 1]);

	if ((i = arg_pos("-beta-weak", argc, argv)) > 0)
		beta_weak = atof(argv[i + 1]);

	if ((i = arg_pos("-alpha", argc, argv)) > 0)
		alpha = atof(argv[i + 1]);

	if ((i = arg_pos("-threads", argc, argv)) > 0)
		num_threads = atoi(argv[i + 1]);

	if ((i = arg_pos("-epoch", argc, argv)) > 0)
		epoch = atoi(argv[i + 1]);

	if ((i = arg_pos("-save-each-epoch", argc, argv)) > 0)
		save_each_epoch = atoi(argv[i + 1]);

	/* initialise vocabulary table */
	vocab = (struct entry *)calloc(vocab_max_size, sizeof(struct entry));
	vocab_hash = (int *)calloc(HASHSIZE, sizeof(int));

	/* initialise the sigmoid table. The array represents the values of the
	sigmoid function from -MAX_SIGMOID to MAX_SIGMOID. The array
	contains SIGMOID_SIZE cells so each cell i contains the sigmoid of :

	X = ( (-SIGMOID_SIZE + (i * 2)) / SIGMOID_SIZE ) * MAX_SIGMOID

	  for i = [0 .. SIGMOID_SIZE]

	One can verify that X goes from -MAX_SIGMOID to MAX_SIGMOID */
	sigmoid = (float *)calloc(SIGMOID_SIZE + 1, sizeof(float));

	if (sigmoid == NULL)
	{
		printf("Not enough memory to instantiate EXP table.\n");
		exit(1);
	}

	/* faster to multiply than divide so pre-compute 1/ SIGMOID_SIZE */
	d = 1 / (float) SIGMOID_SIZE;
	for (i = 0; i < SIGMOID_SIZE + 1; ++i)
	{
		/* start by computing exp(X) */
		sigmoid[i] = exp( (((i * 2) - SIGMOID_SIZE) * d) * MAX_SIGMOID);
		/* then compute sigmoid(X) = exp(X) / (1 + exp(X)) */
		sigmoid[i] = sigmoid[i] / (sigmoid[i] + 1);
	}

	train();
	destroy_network();
	free(vocab_hash);
	free(sigmoid);

	return 0;
}
