#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <mkl.h>
#include <omp.h>

#include "t2sh_intel_kernels.h"

void set_all(complex float *arr, unsigned long dim0, unsigned long dim1) {
  int nthr = 1;
  int P = (dim1 + nthr-1) / nthr;
  {
      int tid = 0;
      int row_start = tid * P;
      int row_end = (tid+1) * P;
      if(row_end > dim1) row_end = dim1;
      for(int row = row_start ; row < row_end ; row++)
      {
        complex float * _arr = arr + row * dim0;
	for(int i = 0 ; i < dim0 ; i++)
	{
          _arr[i] = ((double)rand()) / ((double)RAND_MAX);
	}
     }
   }
}

void load_mat(const char *fname, void *mat, unsigned long nbytes) {
  FILE *f = fopen(fname, "rb");
  assert(f);
  fread(mat, sizeof(char), nbytes, f);
  fclose(f);
}

int main(int argc, char *argv[]) {
  // Create matrices
  unsigned long nmaps = 7;
  unsigned long nimg = 4;
  unsigned long dim0, dim1;
  dim0 = 240;
  dim1 = 260;

  struct timeval start, end;
  complex float *src =
      (complex float *)malloc(dim0 * dim1 * nmaps * nimg * sizeof(complex float)); // 8
  complex float *dst =
      (complex float *)malloc(dim0 * dim1 * 2 * nimg * sizeof(complex float)); // 8
  complex float *dst_ref =
      (complex float *)malloc(dim0 * dim1 * 2 * nimg * sizeof(complex float));
  complex float *sens =
      (complex float *)malloc(dim0 * dim1 * 2 * nmaps * sizeof(complex float)); // 14

  DFTI_DESCRIPTOR_HANDLE plan2d;
  MKL_LONG len[2] = {dim0, dim1};

  DftiCreateDescriptor(&plan2d, DFTI_SINGLE, DFTI_COMPLEX, 2, len);
  DftiSetValue(plan2d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiCommitDescriptor(plan2d);

  load_mat("test_data/adjoint_in.bin", src, dim0*dim1*nmaps*nimg*sizeof(complex float)); 
  load_mat("test_data/adjoint_sens.bin", sens, dim0*dim1*2*nmaps*sizeof(complex float)); 
  load_mat("test_data/adjoint_out.bin", dst_ref, dim0*dim1*2*nimg*sizeof(complex float));

  complex float *cfksp3 = (complex float *)malloc(dim1 * dim0 * 
                                                  sizeof(complex float)); 

  jtmodel_adjoint_benchmark_fast_parallel(sens, dst, src, dim0,
                                dim1, nmaps, 2, nimg, plan2d, cfksp3);
  free(cfksp3);

  double diff = 0.0;
  double ref = 0.0;
  for(int img = 0 ; img < 2*nimg ; img++)
  {
    complex float * _t1 = dst_ref + dim0*dim1*img;
    complex float * _t2 = dst + dim0*dim1*img;
    for(int i = 0 ; i < dim1 ; i++)
    {
      for(int j = 0 ; j < dim0 ; j++)
      {
        complex float t1 = _t1[j + i * dim0];
        complex float t2 = _t2[j + i * dim0];
        double _diff = (t1-t2)*(t1-t2);
        double _ref = (t1) * (t1);
        diff += _diff;
        ref += _ref;
      }
    }
  }

  printf("Relative norm error: %.10e\n", sqrt(diff / ref));

  free(src);
  free(dst);
  free(sens);
}
