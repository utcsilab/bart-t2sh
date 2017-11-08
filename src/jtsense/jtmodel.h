/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#ifdef __cplusplus
extern "C" {
#endif

#include <fftw3.h>
#ifdef USE_MKL
#include <complex.h>
#include <mkl.h>
#endif

#include "misc/mri.h"


struct operator_s;
struct linop_s;

void create_stkern_mat(complex float* stkern_mat,
		const long pat_dims[DIMS], const complex float* pat,
		const long bas_dims[DIMS], const complex float* bas);

struct linop_s* jtmodel_init(const long max_dims[DIMS],
		const struct linop_s* sense_op,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long bas_dims[DIMS], const _Complex float* basis,
		const complex float* stkern_mat,
		bool use_gpu);

#ifdef USE_INTEL_KERNELS
struct linop_s* jtmodel_intel_init(const long max_dims[DIMS],
		const long cfimg_dims[DIMS],
		const struct linop_s* sense_op,
		const long sens_dims[DIMS], const _Complex float* sens,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long bas_dims[DIMS], const _Complex float* basis,
		const complex float* stkern_mat, bool use_gpu,
		DFTI_DESCRIPTOR_HANDLE plan1d_0, DFTI_DESCRIPTOR_HANDLE plan1d_1);
#endif

#ifdef __cplusplus
}
#endif


