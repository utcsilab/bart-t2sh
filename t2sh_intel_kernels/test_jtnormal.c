#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <mkl.h>
#include <omp.h>

void jtmodel_normal_benchmark_fast_parallel(
    const complex float * __restrict__ sens, const float * __restrict__ stkern_mat, 
    complex float * dst, const complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long nmaps,
    const unsigned long nimg,
    DFTI_DESCRIPTOR_HANDLE plan1d_0, DFTI_DESCRIPTOR_HANDLE plan1d_1,
    complex float * cfksp3,
    complex float * cfksp4);

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
  dim0 = 246;
  dim1 = 240;

  struct timeval start, end;
  complex float *src =
      (complex float *)malloc(dim0 * dim1 * 2 * nimg * sizeof(complex float)); // 8
  complex float *dst =
      (complex float *)malloc(dim0 * dim1 * 2 * nimg * sizeof(complex float)); // 8
  complex float *dst_ref =
      (complex float *)malloc(dim0 * dim1 * 2 * nimg * sizeof(complex float));
  complex float *sens =
      (complex float *)malloc(dim0 * dim1 * 2 * nmaps * sizeof(complex float)); // 14
  complex float *stkern_mat = (complex float *)malloc(
      dim0 * dim1 * nimg * nimg * sizeof(complex float));
  float *stkern_mat_trans = (float *)malloc(
      dim0 * dim1 * nimg * nimg * sizeof(float));
  complex float *tst1 = (complex float *)malloc(
       dim0*dim1*nmaps*nimg*sizeof(complex float));
  complex float *tst2 = (complex float *)malloc(
       dim0*dim1*nmaps*nimg*sizeof(complex float));

  DFTI_DESCRIPTOR_HANDLE plan, plan1d_0, plan1d_1;

  MKL_LONG len0 = dim0;
  MKL_LONG len1 = dim1;

  DftiCreateDescriptor(&plan1d_0, DFTI_SINGLE, DFTI_COMPLEX, 1, dim0);
  DftiSetValue(plan1d_0, DFTI_PLACEMENT, DFTI_INPLACE);
  DftiCommitDescriptor(plan1d_0);

  DftiCreateDescriptor(&plan1d_1, DFTI_SINGLE, DFTI_COMPLEX, 1, dim1);
  DftiSetValue(plan1d_1, DFTI_PLACEMENT, DFTI_INPLACE);
  DftiCommitDescriptor(plan1d_1);

  load_mat("test_data/sens30.mat", sens, dim0*dim1*2*nmaps*sizeof(complex float)); 
  load_mat("test_data/stkern30.mat", stkern_mat, dim0*dim1*nimg*nimg*sizeof(complex float)); 
  load_mat("test_data/src30.mat", src, dim0*dim1*2*nimg*sizeof(complex float)); 
  load_mat("test_data/dst30.mat", dst_ref, dim0*dim1*2*nimg*sizeof(complex float));

  // transpose 4x4 matrices and set to float
  for(int img = 0 ; img < nimg*nimg ; img++)
  {
    complex float * nontrans = stkern_mat + img * dim0 * dim1;
    float * trans = stkern_mat_trans + img * dim0 * dim1;
    for(int i = 0 ; i < dim1 ; i++)
    {
      for(int j = 0 ; j < dim0 ; j++)
      {
        trans[i + j * dim1] = creal(nontrans[j + i * dim0]);
      }
    }
  }
  double elapsed;
  double total_elapsed = 0.0;

  complex float *cfksp3 = (complex float *)malloc(dim1 * dim0 * nimg *
                                                  sizeof(complex float)); 
  complex float *cfksp4 = (complex float *)malloc(dim1 * dim0 * nimg *
                                                  sizeof(complex float)); 

  jtmodel_normal_benchmark_fast_parallel(sens, stkern_mat_trans, dst, src, dim0,
                                dim1, nmaps, nimg, plan1d_0, plan1d_1, cfksp3, cfksp4);
  free(cfksp3);
  free(cfksp4);

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
  free(stkern_mat);
}
