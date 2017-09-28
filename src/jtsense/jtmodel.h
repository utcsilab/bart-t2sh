/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#ifdef __cplusplus
extern "C" {
#endif

#include <fftw3.h>

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

struct linop_s* jtmodel_intel_init(const long max_dims[DIMS],
		const long cfimg_dims[DIMS],
		const struct linop_s* sense_op,
		const long sens_dims[DIMS], const _Complex float* sens,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long bas_dims[DIMS], const _Complex float* basis,
		const complex float* stkern_mat, bool use_gpu,
		fftwf_plan plan1d_0, fftwf_plan plan1d_inv_0, fftwf_plan plan1d_1, fftwf_plan plan1d_inv_1);


#ifdef __cplusplus
}
#endif


