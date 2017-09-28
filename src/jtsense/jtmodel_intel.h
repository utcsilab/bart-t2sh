
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"
#include <fftw3.h>

void jtmodel_normal_benchmark_fast(const long cfimg_dims[DIMS], const long sens_dims[DIMS],
		const _Complex float *sens, const float *stkern_mat, 
		_Complex float *dst, const _Complex float *src,
		fftwf_plan plan1d, fftwf_plan plan1d_inv, fftwf_plan plan1d_1, fftwf_plan plan1d_inv_1,
		unsigned int M, unsigned int N);

#ifdef __cplusplus
}
#endif


