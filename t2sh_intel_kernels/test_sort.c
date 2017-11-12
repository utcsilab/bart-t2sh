#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

void gnu_sort_wrapper(float __complex__ * base, size_t len);

static int compare_cmpl_magn(const void* a, const void* b)
{
	return (int)copysignf(1., (cabsf(*(complex float*)a) - cabsf(*(complex float*)b)));
}

int main(int argc, char * argv[])
{
  int N = 100;

  // Create list
  float __complex__ * in1 = (float __complex__ *)malloc(N * sizeof(float __complex__));
  float __complex__ * in2 = (float __complex__ *)malloc(N * sizeof(float __complex__));

  for(int i = 0 ; i < N ; i++)
  {
    in1[i] = in2[i] = (rand() % N) + (rand() % N) * _Complex_I;
  }
  qsort(in1, (size_t)N, sizeof(float __complex__), compare_cmpl_magn);
  gnu_sort_wrapper(in2, (size_t)N);

  double err = 0.0;
  for(int i = 0 ; i < N ; i++)
  {
    double diff = cabsf(in1[i]) -cabsf(in2[i]);
    err += diff;
    if(diff > 0)
    {
      printf("%f %f %f %f\n", creal(in1[i]), creal(in2[i]), cimag(in1[i]), cimag(in2[i]));
    }
  }
  printf("error magnitude: %.10e \n", err);
}
