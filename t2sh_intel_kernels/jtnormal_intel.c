#include <complex.h>
#include <fftw/fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <xmmintrin.h>
#include <mkl.h>

#define BLK 4
inline void TransposeBLKxBLK(complex float * __restrict__ A, complex float * __restrict__ B) {
int i, j;
for (i = 0; i < BLK; i++)
for (j = 0; j < BLK; j++)
B[i*BLK + j] = A[j*BLK + i];
}

inline void TransposePanel(complex float * __restrict__ cor_out, 
                           complex float * __restrict__ cor_out2, 
			   int _p, 
			   int tid,
			   int dim0, 
			   int dim1)
{
	int nblk0 = dim0 / BLK;
	int nblk1 = _p / BLK;
	for(int cc = 0 ; cc < nblk1 ; cc++)
	{
	for(int bb = 0 ; bb < nblk0 ; bb++)
	{
	  int mine = (bb+tid)%nblk0;
	  int b = mine * BLK;
	    int c = cc * BLK;
	    complex float buf1[BLK*BLK];
	    complex float buf2[BLK*BLK];
	    for(int i = 0 ; i < BLK ; i++)
	    {
	      #pragma simd
	      for(int j = 0 ; j < BLK ; j++)
	      {
	        buf1[j + i*BLK] = cor_out[b + j + (c+i)*dim0];
	      }
	    }
	    TransposeBLKxBLK(buf1, buf2);
	    for(int i = 0 ; i < BLK ; i++)
	    {
	      #pragma simd
	      for(int j = 0 ; j < BLK ; j++)
	      {
	        cor_out2[c + j + (b+i)*dim1] = buf2[j + i*BLK];
	      }
	    }
	  }
	}
	for(int cc = nblk1*BLK ; cc < _p ; cc++)
	{
	  #pragma simd
	  for(int i = 0 ; i < dim0 ; i++)
	  {
	    cor_out2[cc + i*dim1] = cor_out[i + cc*dim0];
	  }
	}
}


void jtmodel_normal_benchmark_fast_parallel(
    const complex float * __restrict__ sens, const float * __restrict__ stkern_mat, 
    complex float * dst, const complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long nmaps,
    const unsigned long nimg,
    DFTI_DESCRIPTOR_HANDLE plan1d_0, DFTI_DESCRIPTOR_HANDLE plan1d_1,
    complex float * cfksp3,
    complex float * cfksp4) {

  struct timeval start, end;
  int nthr = omp_get_max_threads();
  int P = (dim1 + nthr-1) / nthr;
  int P0 = (dim0 + nthr-1) / nthr;
  float sc = 1.0 / sqrt((double)dim0 * dim1);

  for(int map = 0 ; map < nmaps ; map++)
  {
    #pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      int row_start = tid * P;
      int row_end = (tid+1) * P;
      if(row_end > dim1) row_end = dim1;

      for(int img = 0 ; img < nimg ; img++)
      {
        for(int row = row_start ; row < row_end; row++)
        {
          const complex float *map0 = sens + map * dim1 * dim0 + dim0 * row;
          const complex float *map1 =
              sens + map * dim1 * dim0 + nmaps * dim0 * dim1 + dim0 * row;
          const  complex float *img0 = src + img * dim0 * dim1 * 2 + dim0 * row;
          const complex float *img1 =
              src + dim0 * dim1 + img * dim0 * dim1 * 2 + dim0 * row;
          complex float *cor_out =
              cfksp3 + img * dim1 * dim0 + dim0 * row;

          #pragma simd
          for (int i = 0; i < dim0; i++) {
            cor_out[i] = (map0[i] * img0[i] + map1[i] * img1[i]) * sc;
          }
          DftiComputeForward(plan1d_0, cor_out, cor_out);
        }

        complex float *cor_out =
            cfksp3 + img * dim1 * dim0 + dim0 * row_start;
        complex float *cor_out2 =
            cfksp4 + img * dim1 * dim0 + row_start;
	
	TransposePanel(cor_out, cor_out2, row_end-row_start, tid, dim0, dim1);
      }
    }

    #pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      int row_start = tid * P0;
      int row_end = (tid+1) * P0;
      if(row_end > dim0) row_end = dim0;
      complex float * stkern_tmp = (complex float*) malloc(dim1 * nimg * sizeof(complex float));
      for (int row = row_start ; row < row_end ; row++) {
        for(int img = 0 ; img < nimg ; img++)
        {
          complex float *cor_out =
              cfksp4 + img * dim1 * dim0 + dim1 * row;
          DftiComputeForward(plan1d_1, cor_out, cor_out);
        }
        for(int img_i = 0 ; img_i < nimg ; img_i++)
        {
          complex float *tmp = stkern_tmp + img_i * dim1;
          for (int img_j = 0; img_j < nimg; img_j++) {
            complex float *img_in = cfksp4 + 
                                    img_j * dim0 * dim1 + row * dim1;
            const float *mat = (img_i > img_j) ? stkern_mat + img_i * dim1 * dim0 + img_j * dim1 * dim0 * nimg + row * dim1 :
                                           stkern_mat + img_j * dim1 * dim0 + img_i * dim1 * dim0 * nimg + row * dim1;
	  if(img_j == 0)
	  {
	   #pragma simd
	    for (int pix = 0; pix < dim1; pix++) {
              tmp[pix] = (img_in[pix] * mat[pix]);
            }
	  }
	  else
	  {
	   #pragma simd
	    for (int pix = 0; pix < dim1; pix++) {
              tmp[pix] += (img_in[pix] * mat[pix]);
            }
          }
	  }
          DftiComputeBackward(plan1d_1, tmp, tmp);
	}
        for(int img_i = 0 ; img_i < nimg ; img_i++)
        {
          complex float *img_in = cfksp4 + 
                                 img_i * dim0 * dim1 + row * dim1;
#pragma simd
          for (int pix = 0; pix < dim1; pix++) {
            img_in[pix] = stkern_tmp[pix + img_i*dim1];
          }
        }
      }
      free(stkern_tmp);

    for(int img_i = 0 ; img_i < nimg ; img_i++)
    {
      complex float *img_in = cfksp4 + 
                               img_i * dim0 * dim1 + row_start * dim1;
      complex float *img_in2 = cfksp3 + 
                               img_i * dim0 * dim1 + row_start;
      TransposePanel(img_in, img_in2, row_end-row_start, tid, dim1, dim0);
    }
    }

    #pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      int row_start = tid * P;
      int row_end = (tid+1) * P;
      if(row_end > dim1) row_end = dim1;
      for (int row = row_start ; row < row_end ; row++) {
        for (int img = 0; img < nimg; img++) {
          const complex float *map0 = sens + map*dim1*dim0 + row * dim0;
          const complex float *map1 = sens + map*dim1*dim0 + nmaps *dim0 * dim1 + row * dim0;
          complex float *cor0 = dst + img *dim1*dim0*2 + row * dim0;
          complex float *cor1 = dst + dim1*dim0+img*dim1*dim0*2 + row * dim0;
          complex float *img_in = cfksp3 + img*dim0*dim1 + row * dim0;
          DftiComputeBackward(plan1d_0, img_in, img_in);
          if(map == 0)
          {
#pragma simd
            for (int i = 0; i < dim0; i++) {
              cor0[i] = 0;
              cor1[i] = 0;
            }
          }
#pragma simd
          for (int i = 0; i < dim0; i++) {
            float r0 = __real__ map0[i];
            float r1 = __real__ map1[i];
            float i0 = __imag__ map0[i];
            float i1 = __imag__ map1[i];
            float _r = __real__ img_in[i];
            float _i = __imag__ img_in[i];
            cor0[i] += ((r0 * _r + i0 * _i) + (r0 * _i - i0 * _r) * _Complex_I) * sc;
            cor1[i] += ((r1 * _r + i1 * _i) + (r1 * _i - i1 * _r) * _Complex_I) * sc;
          }
        }
      }
    }
  }
}

void jtmodel_adjoint_benchmark_fast_parallel(
    const complex float * __restrict__ sens, 
    complex float * dst, complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long nmaps,
    const unsigned long nimg,
    DFTI_DESCRIPTOR_HANDLE plan2d,
    complex float * cfksp3)
{
  float sc = 1.0 / sqrt((double)dim0 * dim1);
  for(int map = 0 ; map < nmaps ; map++)
  {
    const complex float * map0 = sens + map * dim0 * dim1;
    const complex float * map1 = sens + map * dim0 * dim1 + nmaps * dim0*dim1;

    for(int img = 0 ; img < nimg ; img++)
    {
      complex float * ksp = src + map*dim0*dim1 + img*nmaps*dim0*dim1;
      DftiComputeBackward(plan2d, ksp, cfksp3);

      complex float * cor0 = dst + 2 * img * dim0 * dim1;
      complex float * cor1 = dst + 2 * img * dim0 * dim1 + dim0*dim1;

      if(map == 0)
      {
#pragma omp parallel for
#pragma simd
        for (int i = 0; i < dim0*dim1; i++) {
          cor0[i] = 0;
          cor1[i] = 0;
        }
      }
#pragma omp parallel for
#pragma simd
      for (int i = 0; i < dim0*dim1; i++) {
        float r0 = __real__ map0[i];
        float r1 = __real__ map1[i];
        float i0 = __imag__ map0[i];
        float i1 = __imag__ map1[i];
        float _r = __real__ cfksp3[i];
        float _i = __imag__ cfksp3[i];
        cor0[i] += ((r0 * _r + i0 * _i) + (r0 * _i - i0 * _r) * _Complex_I) * sc;
        cor1[i] += ((r1 * _r + i1 * _i) + (r1 * _i - i1 * _r) * _Complex_I) * sc;
      }
    }
  }
}

