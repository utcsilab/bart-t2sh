
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"
#include <fftw3.h>
#ifdef USE_MKL
#include <mkl.h>
#endif

void jtmodel_normal_benchmark_fast_parallel(
		const _Complex float *sens, const float *stkern_mat, 
		_Complex float *dst, const _Complex float *src,
    		const unsigned long dim0,
    		const unsigned long dim1,
    		const unsigned long nmaps,
    		const unsigned long nimg,
		DFTI_DESCRIPTOR_HANDLE plan1d, DFTI_DESCRIPTOR_HANDLE plan1d_1,
		_Complex float *cfksp3,
		_Complex float *cfksp4);

void jtmodel_adjoint_benchmark_fast_parallel(
    const _Complex float * __restrict__ sens, 
    _Complex float * dst, const _Complex float * src,
    const unsigned long dim0,
    const unsigned long dim1,
    const unsigned long nmaps,
    const unsigned long nimg,
    DFTI_DESCRIPTOR_HANDLE plan2d,
    _Complex float * cfksp3);

#ifdef __cplusplus
}
#endif


