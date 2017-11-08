all: test_jtnormal test_lrthresh t2sh_intel.a

CC=icc
CFLAGS=-O3 -ipo -no-prec-div -fp-model fast=2 -xCORE-AVX512 -mkl -DMKL_Complex8="complex float" -qopenmp -std=c99 -vec-threshold0 -static
LDFLAGS=-mkl -qopenmp

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

test_jtnormal: jtnormal_intel.o test_jtnormal.o
	$(CC) test_jtnormal.o jtnormal_intel.o -o test_jtnormal $(LDFLAGS)

test_lrthresh: lrthresh_intel.o test_lrthresh.o
	$(CC) test_lrthresh.o lrthresh_intel.o -o test_lrthresh $(LDFLAGS)

t2sh_intel.a: test_jtnormal test_lrthresh
	ar rcs t2sh_intel.a jtnormal_intel.o lrthresh_intel.o

clean:
	rm -f test_jtnormal test_lrthresh *.o *.a
