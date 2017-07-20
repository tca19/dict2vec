#!/bin/bash

DATA_DIR=./data

# download training file if not already downloaded
TEXT_DATA=enwiki-50M
if [ ! -e "$DATA_DIR/$TEXT_DATA" ]; then
  echo "Downloading training file $TEXT_DATA..."
  wget -qO- https://s3.us-east-2.amazonaws.com/dict2vec-data/enwiki-50M.tar.bz2 \
  | tar -jxv -C $DATA_DIR
  echo "Done."
  echo
fi

# download strong pairs if not already downloaded
STRONG_PAIRS=strong-pairs.txt
if [ ! -e "$DATA_DIR/$STRONG_PAIRS" ]; then
  echo "Downloading file $STRONG_PAIRS..."
  wget -qO- https://s3.us-east-2.amazonaws.com/dict2vec-data/strong-pairs.tar.bz2 \
  | tar -jxv -C $DATA_DIR
  echo "Done."
  echo
fi

# download weak pairs if not already downloaded
WEAK_PAIRS=weak-pairs.txt
if [ ! -e "$DATA_DIR/$WEAK_PAIRS" ]; then
  echo "Downloading file $WEAK_PAIRS..."
  wget -qO- https://s3.us-east-2.amazonaws.com/dict2vec-data/weak-pairs.tar.bz2 \
  | tar -jxv -C $DATA_DIR
  echo "Done."
  echo
fi

# hyperparameters
VECTOR_SIZE=100
WINDOW_SIZE=5
NEGATIVE=5
NB_STRONG=4
NB_WEAK=5
BETA_STRONG=0.8
BETA_WEAK=0.45
THREADS=8
EPOCH=5

# remove existing trained vectors
if [ -e "$DATA_DIR/$TEXT_DATA.vec" ]; then
  rm "$DATA_DIR/$TEXT_DATA.vec"
fi

make

# executable adds the .vec extension when saving vectors so don't worry if
# -output is the same as -input
echo
echo --------------- Training vectors... ---------------
time ./dict2vec -input "$DATA_DIR/$TEXT_DATA" -output "$DATA_DIR/$TEXT_DATA" \
     -strong-file "$DATA_DIR/$STRONG_PAIRS" -weak-file "$DATA_DIR/$WEAK_PAIRS" \
     -size $VECTOR_SIZE -window $WINDOW_SIZE -negative $NEGATIVE \
     -strong-draws $NB_STRONG -weak-draws $NB_WEAK \
     -beta-strong $BETA_STRONG  -beta-weak $BETA_WEAK \
     -sample 1e-4 -threads $THREADS -epoch $EPOCH #-save-each-epoch 1

# evaluate the trained embeddings
echo
echo -------------- Evaluating vectors... --------------
./evaluate.py "$DATA_DIR/$TEXT_DATA.vec"

