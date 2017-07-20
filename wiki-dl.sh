#!/bin/bash

DATA_DIR=./data

echo "Download English Wikipedia dump of July 2017..."
URL=https://dumps.wikimedia.org/enwiki/20170701/enwiki-20170701-pages-articles-multistream.xml.bz2
time wget -qO- $URL | bzip2 -d | perl wiki-parser.pl > "$DATA_DIR/enwiki-full"
echo "Done."
echo

echo "Creating enwiki-50M and enwiki-200M..."
head -c 296093483 "$DATA_DIR/enwiki-full" > "$DATA_DIR/enwiki-50M"
head -c 1164406185 "$DATA_DIR/enwiki-full" > "$DATA_DIR/enwiki-200M"
echo "Done."
echo
