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

CC = gcc

# The -Ofast might not work with older versions of gcc; in that case, use -O2
#
# -lm : for pow() and exp()
# -pthread : for multithreading
# -Ofast : code optimization
# -funroll-loops : unroll loops that can be determined at compile time
# -Wall -Wextra -Wno-unused-result : turn on warning messages
CFLAGS = -std=c11 -lm -pthread -Ofast -funroll-loops -Wall -Wextra -Wno-unused-result

all: dict2vec

dict2vec : dict2vec.c
	$(CC) dict2vec.c -o ./dict2vec $(CFLAGS)

clean:
	rm -rf dict2vec
