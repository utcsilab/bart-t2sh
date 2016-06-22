/* Copyright 2013-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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

	linop_data_t base;

	long cfksp_dims[DIMS];

	const struct linop_s* sense_op;
	const struct operator_s* stkern_op;

	complex float* cfksp;
};


struct stkern_data {

	operator_data_t base;

	long fake_cfksp_dims[DIMS];
	long cfksp_dims[DIMS];
	long stkern_dims[DIMS];

	const complex float* stkern_mat;
};




/**
 * Create T2Sh kernel, Psi = Phi^H P Phi,
 * where Phi is the basis and P is the sampling pattern
 */
static void create_stkern_mat(complex float* stkern_mat,
		const long pat_dims[DIMS], const complex float* pat,
		const long bas_dims[DIMS], const complex float* bas)
{

#if 0
fmac mask bas tmp
transpose 6 15 tmp tmp2
t2sh_proj -K4 tmp2 bas tmp3
transpose 5 6 tmp3 tmp4
transpose 6 15 tmp4 tmp5
#endif

	// -----------------------------------------------------------
	// initialize dimensions and strides

	long max_dims[DIMS + 1];      // [X Y Z 1 1 T K A B C ... 1 1]
	long trp_dims[DIMS + 1];      // [X Y Z 1 1 T 1 A B C ... 1 K]

	long tproj_dims[DIMS + 1];    // [X Y Z 1 1 1 K A B C ... 1 K] 
	long tproj2_dims[DIMS + 1];   // [X Y Z 1 1 K 1 A B C ... 1 K] 
	long stkern_dims[DIMS + 1];   // [X Y Z 1 1 K K A B C ... 1 1] 

	long fake_bas_dims[DIMS + 1]; // [1 1 1 1 1 T K 1 1 1 ... 1 1]

	long max_strs[DIMS];
	long bas_strs[DIMS];
	long pat_strs[DIMS];


	for (unsigned int i = 0; i < DIMS; i++) {

		assert((pat_dims[i] == bas_dims[i]) || (1 == pat_dims[i]) || (1 == bas_dims[i]));
		max_dims[i] = (1 == pat_dims[i]) ? bas_dims[i] : pat_dims[i];
	}
	max_dims[DIMS] = 1;

	// stick the COEFF_DIM into the extra dummy dimension, so that it doesn't get overwritten by the projection
	md_select_dims(DIMS + 1, ~COEFF_FLAG, trp_dims, max_dims);
	trp_dims[DIMS] = max_dims[COEFF_DIM];

	// the t2sh_proj will squash the TE_DIM and set COEFF_DIM to K
	md_select_dims(DIMS + 1, ~TE_FLAG, tproj_dims, trp_dims);
	tproj_dims[COEFF_DIM] = bas_dims[COEFF_DIM];

	// also want tproj with TE_DIM in the right place
	md_transpose_dims(DIMS + 1, TE_DIM, COEFF_DIM, tproj2_dims, tproj_dims);

	// final result is a symmetric matrix with possibly higher-level dims
	md_transpose_dims(DIMS + 1, COEFF_DIM, DIMS, stkern_dims, tproj2_dims);

	// same as basis, but with the extra dummy dimension
	md_copy_dims(DIMS, fake_bas_dims, bas_dims);
	fake_bas_dims[DIMS] = 1;

	md_calc_strides(DIMS, max_strs, max_dims, CFL_SIZE);
	md_calc_strides(DIMS, bas_strs, bas_dims, CFL_SIZE);
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);


	// -----------------------------------------------------------
	// fmac pattern basis tmp

	complex float* tmp = md_alloc_sameplace(DIMS + 1, max_dims, CFL_SIZE, bas);

	md_clear(DIMS, max_dims, tmp, CFL_SIZE);
	md_zfmac2(DIMS, max_dims, max_strs, tmp, bas_strs, bas, pat_strs, pat);


	// -----------------------------------------------------------
	// transpose 6 16 tmp tmp2

	// cannot do in place because there may be higher-level dims in use
	complex float* tmp2 = md_alloc_sameplace(DIMS + 1, trp_dims, CFL_SIZE, bas);
	md_transpose(DIMS + 1, COEFF_DIM, DIMS, trp_dims, tmp2, max_dims, tmp, CFL_SIZE);
	md_free(tmp);


	// -----------------------------------------------------------
	// tproj tmp2 bas tmp3

	complex float* tmp3 = md_alloc_sameplace(DIMS + 1, tproj_dims, CFL_SIZE, bas);
	md_zmatmulc(DIMS + 1, tproj_dims, tmp3, fake_bas_dims, bas, trp_dims, tmp2);
	md_free(tmp2);


	// -----------------------------------------------------------
	// transpose 5 6 tmp3 tmp4

	complex float* tmp4 = md_alloc_sameplace(DIMS + 1, tproj2_dims, CFL_SIZE, bas);
	md_transpose(DIMS + 1, TE_DIM, COEFF_DIM, tproj2_dims, tmp4, tproj_dims, tmp3, CFL_SIZE);
	md_free(tmp3);


	// -----------------------------------------------------------
	// transpose 6 16 tmp4 stkern_mat
	
	md_transpose(DIMS + 1, COEFF_DIM, DIMS, stkern_dims, stkern_mat, tproj2_dims, tmp4, CFL_SIZE);
	md_free(tmp4);
}


static void stkern_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const struct stkern_data* data = CONTAINER_OF(_data, const struct stkern_data, base);

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

static void stkern_del(const operator_data_t* _data)
{
	const struct stkern_data* data = CONTAINER_OF(_data, const struct stkern_data, base);
	md_free((void*)data->stkern_mat);
	free((void*)data);
}

static const struct operator_s* stkern_init(const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis,
		long stkern_dims[DIMS], long cfksp_dims[DIMS],
		bool use_gpu)
{


	PTR_ALLOC(struct stkern_data, data);

	// FIXME this is very very slow on GPU
	// FIXME does not work if pattern differs across CSHIFT dim
	complex float* stkern_mat = md_alloc(DIMS, stkern_dims, CFL_SIZE);
	create_stkern_mat(stkern_mat, pat_dims, pattern, bas_dims, basis);

#ifdef USE_CUDA
	complex float* gpu_stkern_mat = NULL;
	if (use_gpu) {

		gpu_stkern_mat = md_gpu_move(DIMS, stkern_dims, stkern_mat, CFL_SIZE);
		md_free(stkern_mat);
		stkern_mat = gpu_stkern_mat;
	}
#else
	assert(!use_gpu);
#endif

	data->stkern_mat = stkern_mat;

	md_copy_dims(DIMS, data->stkern_dims, stkern_dims);
	md_copy_dims(DIMS, data->cfksp_dims, cfksp_dims);

	long fake_cfksp_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, fake_cfksp_dims, cfksp_dims);
	fake_cfksp_dims[TE_DIM] = cfksp_dims[COEFF_DIM];
	md_copy_dims(DIMS, data->fake_cfksp_dims, fake_cfksp_dims);

	return operator_create(DIMS, cfksp_dims, DIMS, cfksp_dims, &data->base, stkern_apply, stkern_del);
}


static void jtmodel_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	UNUSED(src);
	UNUSED(dst);
	UNUSED(_data);
	error("TODO: implement compact forward op\n");
}


static void jtmodel_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct jtmodel_data* data = CONTAINER_OF(_data, const struct jtmodel_data, base);

	linop_adjoint_unchecked(data->sense_op, dst, src);
}


static void jtmodel_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{

	const struct jtmodel_data* data = CONTAINER_OF(_data, const struct jtmodel_data, base);

	complex float* cfksp2 = md_alloc_sameplace(DIMS, data->cfksp_dims, CFL_SIZE, src);

	linop_forward_unchecked(data->sense_op, data->cfksp, src);
	operator_apply_unchecked(data->stkern_op, cfksp2, data->cfksp);
	linop_adjoint_unchecked(data->sense_op, dst, cfksp2);

	md_free(cfksp2);

}


static void jtmodel_del(const linop_data_t* _data)
{

	const struct jtmodel_data* data = CONTAINER_OF(_data, const struct jtmodel_data, base);

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
		const struct linop_s* sense_op,
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis,
		bool use_gpu)
{

	PTR_ALLOC(struct jtmodel_data, data);

	data->sense_op = sense_op;

	md_select_dims(DIMS, (FFT_FLAGS | COIL_FLAG | COEFF_FLAG | CSHIFT_FLAG), data->cfksp_dims, max_dims);

#ifdef USE_CUDA
	data->cfksp = (use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->cfksp_dims, CFL_SIZE);
#else
	assert(!use_gpu);
	data->cfksp = md_alloc(DIMS, data->cfksp_dims, CFL_SIZE);
#endif

	long stkern_dims[DIMS];
	md_select_dims(DIMS, (PHS1_FLAG | PHS2_FLAG | COEFF_FLAG | CSHIFT_FLAG), stkern_dims, max_dims);
	stkern_dims[TE_DIM] = stkern_dims[COEFF_DIM];

	const struct operator_s* stkern_op = stkern_init(pat_dims, pattern, bas_dims, basis, stkern_dims, data->cfksp_dims, use_gpu);
	data->stkern_op = stkern_op;

	return linop_create(DIMS, data->cfksp_dims, linop_domain(sense_op)->N, linop_domain(sense_op)->dims, &data->base, jtmodel_forward, jtmodel_adjoint, jtmodel_normal, NULL, jtmodel_del);

}



