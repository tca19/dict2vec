#!/bin/bash
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

DATA_DIR=./data

echo "Downloading English Wikipedia dump of May 2018..."
URL=https://dumps.wikimedia.org/enwiki/20180520/enwiki-20180520-pages-articles-multistream.xml.bz2
time wget -qO- $URL | bzip2 -d | perl wiki-parser.pl > "$DATA_DIR/enwiki-full"
echo "Done."
echo

echo "Creating enwiki-50M and enwiki-200M..."
head -c 296018828 "$DATA_DIR/enwiki-full" > "$DATA_DIR/enwiki-50M"
head -c 1164281377 "$DATA_DIR/enwiki-full" > "$DATA_DIR/enwiki-200M"
echo "Done."
echo
