/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
*/

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "jtmodel.h"



struct jtmodel_data {

	long ksp_dims[DIMS];
	long cfksp_dims[DIMS];

	const struct linop_s* sense_op;
	const struct linop_s* sample_op;
	const struct linop_s* temporal_op;

	const struct operator_s* stkern_op;

	complex float* cfksp;

	bool use_cfksp;

};


struct stkern_data {

	long fake_cfksp_dims[DIMS];
	long cfksp_dims[DIMS];
	long stkern_dims[DIMS];

	const complex float* stkern_mat;
};




// consider moving this up, so that rjtsense only has to call this once
static void create_stkern_mat(complex float* stkern_mat,
		const long pat_dims[DIMS], const complex float* pat,
		const long bas_dims[DIMS], const complex float* bas)
{

#if 0
fmac mask_perm_16 bas_exp_16 tmp
transpose 6 16 tmp tmp2
tproj -K3 tmp2 bas_exp_16 tmp3
transpose 5 6 tmp3 tmp4
transpose 6 15 tmp4 tmp5
#endif

	// -----------------------------------------------------------
	// initialize dimensions and strides
	
	long max_dims[DIMS + 1];      // [X Y Z 1 1 T K 1 ... 1 1]
	long trp_dims[DIMS + 1];      // [X Y Z 1 1 T 1 1 ... 1 K]
	long tproj_dims[DIMS + 1];    // [X Y Z 1 1 1 K 1 ... 1 K] 
	long fake_bas_dims[DIMS + 1]; // [1 1 1 1 1 T K 1 ... 1 1]

	long max_strs[DIMS];
	long bas_strs[DIMS];
	long pat_strs[DIMS];


	for (unsigned int i = 0; i < DIMS; i++) {

		assert((pat_dims[i] == bas_dims[i]) || (1 == pat_dims[i]) || (1 == bas_dims[i]));
		max_dims[i] = (1 == pat_dims[i]) ? bas_dims[i] : pat_dims[i];
	}
	max_dims[DIMS] = 1;

	md_select_dims(DIMS + 1, ~COEFF_FLAG, trp_dims, max_dims);
	trp_dims[DIMS] = max_dims[COEFF_DIM];

	md_select_dims(DIMS + 1, ~TE_FLAG, tproj_dims, trp_dims);
	tproj_dims[COEFF_DIM] = bas_dims[COEFF_DIM];

	md_copy_dims(DIMS, fake_bas_dims, bas_dims);
	fake_bas_dims[DIMS] = 1;
	
	md_calc_strides(DIMS, max_strs, max_dims, CFL_SIZE);
	md_calc_strides(DIMS, bas_strs, bas_dims, CFL_SIZE);
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);


	// -----------------------------------------------------------
	// fmac pattern basis tmp

	complex float* tmp = md_alloc_sameplace(DIMS, max_dims, CFL_SIZE, bas);

	md_clear(DIMS, max_dims, tmp, CFL_SIZE);
	md_zfmac2(DIMS, max_dims, max_strs, tmp, bas_strs, bas, pat_strs, pat);


	// -----------------------------------------------------------
	// transpose 6 16 tmp tmp

	md_transpose(DIMS + 1, COEFF_DIM, DIMS, trp_dims, tmp, max_dims, tmp, CFL_SIZE);


	// -----------------------------------------------------------
	// tproj tmp bas stkern_mat

	md_zmatmulc(DIMS + 1, tproj_dims, stkern_mat, fake_bas_dims, bas, trp_dims, tmp);


	md_free(tmp);
}


static void stkern_apply(const void* _data, unsigned int N, void* args[N])
{
	const struct stkern_data* data = _data;

	long fake_cfksp_strs[DIMS];
	long stkern_strs[DIMS];
	long cfksp_strs[DIMS];

	md_calc_strides(DIMS, fake_cfksp_strs, data->fake_cfksp_dims, CFL_SIZE);
	md_calc_strides(DIMS, stkern_strs, data->stkern_dims, CFL_SIZE);
	md_calc_strides(DIMS, cfksp_strs, data->cfksp_dims, CFL_SIZE);

	assert(2 == N);

	complex float* dst = args[0];
	const complex float* src = args[1];

	md_zmatmul2(DIMS, data->fake_cfksp_dims, fake_cfksp_strs, dst, data->stkern_dims, stkern_strs, data->stkern_mat, data->cfksp_dims, cfksp_strs, src);

}

static void stkern_del(const void* _data)
{
	const struct stkern_data* data = _data;
	md_free((void*)data->stkern_mat);
	free((void*)data);
}

static const struct operator_s* stkern_init(const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis,
		long stkern_dims[DIMS], long cfksp_dims[DIMS])
{

	struct stkern_data* data = xmalloc(sizeof(struct stkern_data));

#if 0
	// FIXME this is very very slow on GPU
	complex float* stkern_mat = md_alloc_sameplace(DIMS, stkern_dims, CFL_SIZE, basis);
	create_stkern_mat(stkern_mat, pat_dims, pattern, bas_dims, basis);
#else
	// FIXME only do this if running on GPU
	complex float* stkern_mat = md_alloc_sameplace(DIMS, stkern_dims, CFL_SIZE, basis);
	complex float* stkern_mat_cpu = md_alloc(DIMS, stkern_dims, CFL_SIZE);
	complex float* bas_cpu = md_alloc(DIMS, bas_dims, CFL_SIZE);
	complex float* pat_cpu = md_alloc(DIMS, pat_dims, CFL_SIZE);

	md_copy(DIMS, bas_dims, bas_cpu, basis, CFL_SIZE);
	md_copy(DIMS, pat_dims, pat_cpu, pattern, CFL_SIZE);

	create_stkern_mat(stkern_mat_cpu, pat_dims, pat_cpu, bas_dims, bas_cpu);
	md_copy(DIMS, stkern_dims, stkern_mat, stkern_mat_cpu, CFL_SIZE);

	md_free(stkern_mat_cpu);
	md_free(bas_cpu);
	md_free(pat_cpu);
#endif

	data->stkern_mat = stkern_mat;

	md_copy_dims(DIMS, data->stkern_dims, stkern_dims);
	md_copy_dims(DIMS, data->cfksp_dims, cfksp_dims);

	long fake_cfksp_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, fake_cfksp_dims, cfksp_dims);
	fake_cfksp_dims[TE_DIM] = cfksp_dims[COEFF_DIM];
	md_copy_dims(DIMS, data->fake_cfksp_dims, fake_cfksp_dims);

	return operator_create(DIMS, cfksp_dims, DIMS, cfksp_dims, data, stkern_apply, stkern_del);
}


static void jtmodel_forward(const void* _data, complex float* dst, const complex float* src)
{
	const struct jtmodel_data* data = _data;

	if (data->use_cfksp) {

		UNUSED(src);
		UNUSED(dst);
		UNUSED(_data);
		error("use_cfksp is on - TODO: implement compact forward op\n");
	}
	else {

		md_clear(DIMS, data->cfksp_dims, data->cfksp, CFL_SIZE);

		linop_forward_unchecked(data->sense_op, data->cfksp, src);
		linop_forward_unchecked(data->temporal_op, dst, data->cfksp);
		linop_forward_unchecked(data->sample_op, dst, dst);
	}
}


static void jtmodel_adjoint(const void* _data, complex float* dst, const complex float* src)
{

	const struct jtmodel_data* data = _data;

	if (data->use_cfksp)
		linop_adjoint_unchecked(data->sense_op, dst, src);
	else {

		complex float* ksp = md_alloc_sameplace(DIMS, data->ksp_dims, CFL_SIZE, src);
		md_clear(DIMS, data->ksp_dims, ksp, CFL_SIZE);

		md_clear(DIMS, data->cfksp_dims, data->cfksp, CFL_SIZE);

		linop_adjoint_unchecked(data->sample_op, ksp, src);
		linop_adjoint_unchecked(data->temporal_op, data->cfksp, ksp);
		linop_adjoint_unchecked(data->sense_op, dst, data->cfksp);

		md_free(ksp);
	}
}
//#endif


static void jtmodel_normal(const void* _data, complex float* dst, const complex float* src)
{

	const struct jtmodel_data* data = _data;

	complex float* cfksp2 = md_alloc_sameplace(DIMS, data->cfksp_dims, CFL_SIZE, src);

	linop_forward_unchecked(data->sense_op, data->cfksp, src);
	operator_apply_unchecked(data->stkern_op, cfksp2, data->cfksp);
	linop_adjoint_unchecked(data->sense_op, dst, cfksp2);

	md_free(cfksp2);

}


static void jtmodel_del(const void* _data)
{

	const struct jtmodel_data* data = _data;
	operator_free(data->stkern_op);
	md_free(data->cfksp);
	free((void*)data);
}


/**
 * Create jtsense operator, y = P Phi F S a,
 * where P is the sampling operator, Phi is the basis,
 * F is the Fourier transform and S is the sensitivity maps
 *
 * @param max_dims maximal dimensions across all data structures
 * @param sense_op Fourier transform and sensitivity maps (F S)
 * @param temporal_op temporal operator (Phi)
 * @param sample_op sampling operator (P)
 */
struct linop_s* jtmodel_init(const long max_dims[DIMS],
		const struct linop_s* sense_op, const struct linop_s* temporal_op, const struct linop_s* sample_op,
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis,
		bool use_cfksp)
{

	struct jtmodel_data* data = xmalloc(sizeof(struct jtmodel_data));

	data->sense_op = sense_op;
	data->use_cfksp = use_cfksp; // true if writing compact op in terms of cfimg

	if (use_cfksp) {

		UNUSED(sample_op);
		UNUSED(temporal_op);

		data->sample_op = NULL;
		data->temporal_op = NULL;

		md_select_dims(DIMS, (FFT_FLAGS | COIL_FLAG | COEFF_FLAG), data->ksp_dims, max_dims);
	}
	else {

		data->sample_op = sample_op;
		data->temporal_op = temporal_op;

		md_select_dims(DIMS, (FFT_FLAGS | COIL_FLAG | TE_FLAG), data->ksp_dims, max_dims);
	}

	md_select_dims(DIMS, (FFT_FLAGS | COIL_FLAG | COEFF_FLAG), data->cfksp_dims, max_dims);

	data->cfksp = md_alloc_sameplace(DIMS, data->cfksp_dims, CFL_SIZE, basis);

	long stkern_dims[DIMS];
	md_select_dims(DIMS, (PHS1_FLAG | PHS2_FLAG | COEFF_FLAG), stkern_dims, max_dims);
	stkern_dims[TE_DIM] = stkern_dims[COEFF_DIM];

	const struct operator_s* stkern_op = stkern_init(pat_dims, pattern, bas_dims, basis, stkern_dims, data->cfksp_dims);
	data->stkern_op = stkern_op;

	if (use_cfksp)
		return linop_create(DIMS, data->cfksp_dims, linop_domain(sense_op)->N, linop_domain(sense_op)->dims, data, jtmodel_forward, jtmodel_adjoint, jtmodel_normal, NULL, jtmodel_del);
	else
		return linop_create(linop_codomain(sample_op)->N, linop_codomain(sample_op)->dims, linop_domain(sense_op)->N, linop_domain(sense_op)->dims, data, jtmodel_forward, jtmodel_adjoint, jtmodel_normal, NULL, jtmodel_del);

}



