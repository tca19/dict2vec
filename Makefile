CC = gcc

# The -Ofast might not work with older versions of gcc; in that case, use -O2
#
# -lm : for pow() and exp()
# -pthread : for multithreading
# -Ofast : code optimisation
# -funroll-loops : unroll loops that can be determined at compile time
# -Wall -Wextra -Wno-unused-result : turn on warning messages
CFLAGS = -std=c11 -lm -pthread -Ofast -funroll-loops -Wall -Wextra -Wno-unused-result

all: dict2vec

dict2vec : dict2vec.c
	$(CC) dict2vec.c -o ./dict2vec $(CFLAGS)

clean:
	rm -rf dict2vec
