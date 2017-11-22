all: test_jtnormal test_jtnormal_adjoint test_lrthresh test_sort t2sh_intel.a 

CC=icc
CXX=icpc
CFLAGS=-g -O3 -ipo -no-prec-div -fp-model fast=2 -xHOST -mkl -DMKL_Complex8="complex float" -DMKL_Complex16="complex double" -qopenmp -std=c99 -vec-threshold0 -static
CXXFLAGS=-g -O3 -ipo -no-prec-div -fp-model fast=2 -xHOST -mkl -DMKL_Complex8="complex float" -DMKL_Complex16="complex double" -qopenmp -vec-threshold0 -static
LDFLAGS=-mkl -qopenmp

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cc.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

test_jtnormal: jtnormal_intel.o test_jtnormal.o
	$(CC) test_jtnormal.o jtnormal_intel.o -o test_jtnormal $(LDFLAGS)

test_jtnormal_adjoint : jtnormal_intel.o test_jtnormal_adjoint.o
	$(CC) test_jtnormal_adjoint.o jtnormal_intel.o -o test_jtnormal_adjoint $(LDFLAGS)

test_lrthresh: lrthresh_intel.o test_lrthresh.o
	$(CC) test_lrthresh.o lrthresh_intel.o -o test_lrthresh $(LDFLAGS)

test_sort: sort_intel.o test_sort.o
	$(CC) test_sort.o sort_intel.o -o test_sort $(LDFLAGS)

t2sh_intel.a: test_jtnormal test_lrthresh sort_intel.o
	ar rcs t2sh_intel.a jtnormal_intel.o lrthresh_intel.o sort_intel.o

clean:
	rm -f test_jtnormal test_lrthresh test_jtnormal_adjoint test_sort *.o *.a
