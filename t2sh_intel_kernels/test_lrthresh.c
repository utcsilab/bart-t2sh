#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <complex.h>
#include <math.h>
#include <sys/time.h>

void mylrthresh(complex float *mat1, complex float *mat2, float lambda, int M,
                int N, int nimg, int nmap, int blksize, int shift0, int shift1);


int main(int argc, char *argv[]) {
  int nimg = 4;
  int M = 240;
  int N = 260;
  int nmap = 2;
  int blksize = 12;
  float lambda = 0.0f;
  complex float *mat1 =
      (complex float *)malloc(nmap * nimg * M * N * sizeof(complex float));
  complex float *mat2 =
      (complex float *)malloc(nmap * nimg * M * N * sizeof(complex float));
  complex float *ref =
      (complex float *)malloc(nmap * nimg * M * N * sizeof(complex float));

  // Load from file
  {
    FILE *f = fopen("test_data/lrthresh_in.bin", "rb");
    fread(mat1, sizeof(complex float), nmap * nimg * M * N, f);
    fclose(f);
  }
  {
    FILE *f = fopen("test_data/lrthresh_out.bin", "rb");
    fread(&lambda, sizeof(float), 1, f);
    fread(ref, sizeof(complex float), nmap * nimg * M * N, f);
    fclose(f);
  }

  mylrthresh(mat1, mat2, lambda, M, N, nimg, nmap, blksize,0,0);

  // sum
  double sm1 = 0.;
  double sm2 = 0.;
  for (int i = 0; i < nmap * nimg * M * N; i++) {
    complex double diff = mat2[i] - ref[i];
    sm1 += conj(diff) * diff;
    sm2 += conj(ref[i]) * ref[i];
  }

  printf("Relative norm error: %.10e\n", sqrt(sm1) / sqrt(sm2));

  free(mat1);
  free(mat2);
  free(ref);
}
