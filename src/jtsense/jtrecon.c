/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2013-2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
*/

#include <complex.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include "iter/iter2.h"
#include "iter/prox.h"
#include "iter/lsqr.h"
#include "iter/thresh.h"

#include "num/vecops.h"
#include "num/multind.h"
#include "num/gpuops.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/rand.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/rvc.h"
#include "linops/sampling.h"
#include "linops/optest.h"
#include "linops/grad.h"

#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "wavelet2/wavelet.h"

#include "sense/model.h"
#include "sense/optcom.h"

#include "lowrank/lrthresh.h"

#include "jtsense/jtmodel.h"

#include "jtrecon.h"



const struct jtsense_conf jtsense_defaults = {
	.K = 3,
	.jsparse = false,
	.use_ist = false,
	.crop = false,
	.randshift = true,
	.zmean = false,

	.modelerr = 0.,
	.Kmodelerr = 0,

	.positive = false,

	.use_l2 = false,
	.l2lambda = 0.,

	.num_l1wav_lam = 0,
	.l1wav_lambdas = NULL,

	.use_l1wav = false,
	.lambda_l1wav = 0.,
	.l1wav_dim = -1,

	.use_llr = false,
	.lambda_llr = 0.,
	.llrblk = 8,
	.llr_dim = -1,

	.use_tv = false,
	.lambda_tv = 0.,

	.use_odict = false,
	.lambda_odict = 0.,

	.fast = false,
};


struct data {

	const complex float* pattern;
  
	const struct linop_s* jtsense_op;
	const struct linop_s* temporal_op;
	const struct operator_p_s* l1wavthresh_op;
	const struct operator_p_s* lrthresh_op;
	const struct operator_p_s* l1thresh_op;
	const struct operator_p_s* tvthresh_op;
	const struct operator_p_s* l2mean_op;
	const struct linop_s* odict_op;

	const complex float* kspace;

	struct jtsense_conf* conf;

	long ksp_dims[DIMS];
	long img_dims[DIMS];
	long cfimg_dims[DIMS];
	long bfimg_dims[DIMS];
	long wave_dims[DIMS];
	long x_dims[DIMS];

	struct wavelet_plan_s* wdata;

	float* tmp_coeff;

	_Bool crop;

};


struct data2 {

	const struct jtsense_conf* conf;
	unsigned int num_prox_funs;
	const struct operator_p_s** prox_funs;
	const struct linop_s** G_ops;
	const complex float* kspace;
	const struct linop_s* A_op;
	const obj_fun_t* obj_funs;
};


struct modelerr_data {
	int K;
	long cfimg_dims[DIMS];
};


static void modelerr_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct modelerr_data* data = _data;

	long strs[DIMS];
	md_calc_strides(DIMS, strs, data->cfimg_dims, CFL_SIZE);

	long dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, dims, data->cfimg_dims);
	dims[COEFF_DIM] = data->K;

	// zero out the first K components
	md_copy(DIMS, data->cfimg_dims, dst, src, CFL_SIZE);
	md_clear2(DIMS, dims, strs, dst, CFL_SIZE);
}


static void modelerr_free(const void* data)
{
	free((void*)data);
}

#if 0
static void modelerr_normal(const void* _data, complex float* dst, const complex float* src)
{
	const struct modelerr_data* data = _data;

	modelerr_apply(_data, dst, src);
	md_zsmul(DIMS, data->img_dims, dst, dst, -1.);
}
#endif

static struct linop_s* modelerr_linop_create(const long cfimg_dims[DIMS], const int K)
{
	struct modelerr_data* data = xmalloc(sizeof(struct modelerr_data));

	assert(K <= cfimg_dims[COEFF_DIM]);
	data->K = K;

	md_copy_dims(DIMS, data->cfimg_dims, cfimg_dims);

	return linop_create(DIMS, cfimg_dims, DIMS, cfimg_dims, data, modelerr_apply, modelerr_apply, modelerr_apply, NULL, modelerr_free);

}


float jt_estimate_scaling(const long dims[DIMS], const _Complex float* sens, const _Complex float* data)
{
	assert(1 == dims[MAPS_DIM]);

	long te_img_dims[DIMS];
	md_select_dims(DIMS, ~(COIL_FLAG), te_img_dims, dims);

	long str[DIMS];
	md_calc_strides(DIMS, str, te_img_dims, sizeof(complex float));

	complex float* tmp = md_alloc(DIMS, te_img_dims, sizeof(complex float));

	if (NULL == sens)
		rss_combine(dims, tmp, data);
	 else
		optimal_combine(dims, 0., tmp, sens, data);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~(TE_FLAG), img_dims, te_img_dims);

	complex float* tmpnorm = md_alloc(DIMS, img_dims, sizeof(complex float));
	md_zrss(DIMS, te_img_dims, TE_FLAG, tmpnorm, tmp);

	size_t imsize = (size_t)md_calc_size(DIMS, img_dims);

	float scale = estimate_scaling_norm(1., imsize, tmpnorm, false);

	md_free(tmp);
	md_free(tmpnorm);

	return scale;
}



struct multithresh_data {
	unsigned int P;
	struct operator_p_s** thresh_ops;
	const struct linop_s* unitary_op;
	long loop_dim;
	long single_dims[DIMS];
};


static void multithresh_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	const struct multithresh_data* mdata = _data;

	const long* full_dims = linop_codomain(mdata->unitary_op)->dims;
	long N = linop_codomain(mdata->unitary_op)->N;

	complex float* coeff = md_alloc_sameplace(N, full_dims, CFL_SIZE, dst);
	linop_forward_unchecked(mdata->unitary_op, coeff, src);

	complex float* loop_coeff = md_alloc_sameplace(DIMS, mdata->single_dims, CFL_SIZE, dst);

	long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

	for (unsigned int i = 0; i < mdata->P; i++)
	{
		pos[mdata->loop_dim] = i;

		md_copy_block(DIMS, pos, mdata->single_dims, loop_coeff, full_dims, coeff, CFL_SIZE);

		operator_p_apply_unchecked(mdata->thresh_ops[i], mu, loop_coeff, loop_coeff);

		md_copy_block(DIMS, pos, full_dims, coeff, mdata->single_dims, loop_coeff, CFL_SIZE);
	}

	linop_adjoint_unchecked(mdata->unitary_op, dst, coeff);

	md_free(coeff);
	md_free(loop_coeff);
}


static void multithresh_del(const void* _data)
{
	const struct multithresh_data* mdata = _data;
	for (unsigned int i = 0; i < mdata->P; i++)
		operator_p_free(mdata->thresh_ops[i]);
	free(mdata->thresh_ops);
	free((void*)mdata);
}


static const struct operator_p_s* prox_multithresh_create(unsigned int P, const float lambdas[P], const struct linop_s* unitary_op, const long dim, bool use_gpu)
{

	const long* full_dims = linop_codomain(unitary_op)->dims;

	// check compatibility -- need a lambda for each outer dimension
	assert(full_dims[dim] == P);

	struct multithresh_data* mdata = xmalloc( sizeof(struct multithresh_data) );

	mdata->P = P;
	mdata->unitary_op = unitary_op;
	mdata->thresh_ops = xmalloc( P * sizeof(struct operator_p_s*) );
	mdata->loop_dim = dim;

	md_select_dims(DIMS, ~MD_BIT(dim), mdata->single_dims, full_dims);

	for (unsigned int i = 0; i < P; i++)
		mdata->thresh_ops[i] = (struct operator_p_s*)prox_thresh_create(DIMS, mdata->single_dims, lambdas[i], 0u, use_gpu);

	return operator_p_create(DIMS, linop_domain(unitary_op)->dims, DIMS, linop_domain(unitary_op)->dims,
			(void*)mdata, multithresh_apply, multithresh_del);
}


static float datacon_l2norm(const void* _data, const float* _x)
{
	const struct data2* data = _data;
	const long* dims = linop_codomain(data->A_op)->dims;
	const complex float* x = (const complex float*)_x;

	complex float* tmp = md_alloc_sameplace(DIMS, dims, CFL_SIZE, _x);
	linop_forward_unchecked(data->A_op, tmp, x);
	md_zsub(DIMS, dims, tmp, data->kspace, tmp); // y - Ax

	float t = md_zscalar_real(DIMS, dims, tmp, tmp);

	md_free(tmp);

	return 0.5 * t;
}

static float wavelet_l1norm(const struct linop_s* wav_op, const struct operator_p_s* thresh_op, const complex float* x)
{
	// FIXME: use multiple lambdas if available: multithresh
	float lambda = get_thresh_lambda(thresh_op);
	const long* dims = linop_codomain(wav_op)->dims;

	complex float* tmp = md_alloc_sameplace(DIMS, dims, CFL_SIZE, x);

	linop_forward_unchecked(wav_op, tmp, x);

	float t =  lambda *  md_z1norm(DIMS, dims, tmp);

	md_free(tmp);

	//debug_printf(DP_DEBUG1, "wavl1norm = %f\n", t);

	return t;
}

static float llr_nucnorm(const struct linop_s* G_op, const struct operator_p_s* lrthresh_op, const complex float* x)
{
	float lambda = get_lrthresh_lambda(lrthresh_op);
	const long* dims = linop_codomain(G_op)->dims;

	complex float* tmp = md_alloc_sameplace(DIMS, dims, CFL_SIZE, x);

	linop_forward_unchecked(G_op, tmp, x);

	float t = lambda * lrnucnorm(lrthresh_op, tmp);

	md_free(tmp);

	//debug_printf(DP_DEBUG1, "llr_nucnorm = %f\n", t);
	return t;
}

static float odict_l1norm(const struct linop_s* G_op, const struct operator_p_s* thresh_op, const complex float* x)
{
	float lambda = get_thresh_lambda(thresh_op);
	const long* dims = linop_codomain(G_op)->dims; // G_op should be identity
	
	float t =  lambda * md_z1norm(DIMS, dims, x);
	//debug_printf(DP_DEBUG1, "odict_l1norm = %f\n", t);
	return t;
}

static float l2mean_norm(const struct linop_s* G_op, const struct operator_p_s* l2mean_op, const complex float* x)
{
	UNUSED(G_op);
	UNUSED(l2mean_op);
	UNUSED(x);
	return 0.;
}

static float jtsense_objective(const void* _data, const float* _x)
{
	const struct data2* data = _data;
	const complex float* x = (const complex float*)_x;

	float t1 = datacon_l2norm(_data, _x);

	for (unsigned int i = 0; i < data->num_prox_funs; i++)
		t1 += data->obj_funs[i](data->G_ops[i], data->prox_funs[i], x);

	return t1;
}


void jtsense_recon2(const struct jtsense_conf* conf, _Complex float* x_img,
		italgo_fun2_t italgo, void* iconf,
		const struct linop_s* E_op,
		const struct linop_s* T_op,
		unsigned int num_prox_funs,
		const struct operator_p_s** prox_funs,
		const struct linop_s** G_ops,
		const obj_fun_t* obj_funs,
		const _Complex float* kspace,
		const _Complex float* x_truth)
{


	// -----------------------------------------------------------
	// initialize data: struct to hold all data and operators
	struct data2* data = xmalloc(sizeof(struct data));

	data->conf = conf;
	data->kspace = kspace;


	// -----------------------------------------------------------
	// create forward model

	if (NULL != T_op)
		data->A_op = linop_chain(T_op, E_op);
	else
		data->A_op = linop_clone(E_op);

	const struct lsqr_conf lsqr_conf = { .lambda = conf->use_l2 ? conf->l2lambda : 0. };


	// -----------------------------------------------------------
	// create prox operators
	
	data->num_prox_funs = num_prox_funs;
	data->prox_funs = prox_funs;
	data->G_ops = G_ops;
	data->obj_funs = obj_funs;


	// -----------------------------------------------------------
	// call iterative algorithm interface
	
	if (!conf->fast) {
		float objval = jtsense_objective((void*)data, (const float*)x_img);
		debug_printf(DP_DEBUG3, "OBJVAL = %f\n", objval);
	}

	lsqr2(DIMS, &lsqr_conf, italgo, iconf, data->A_op, data->num_prox_funs, data->prox_funs, data->G_ops, linop_domain(data->A_op)->dims, x_img, linop_codomain(data->A_op)->dims, kspace, NULL, x_truth, conf->fast ? NULL : data, conf->fast ? NULL : jtsense_objective);
	

	// -----------------------------------------------------------
	// cleanup

	linop_free(data->A_op);
	free(data);
}


struct zmean_data {
	
	const struct linop_s* transform_op;
	long mean_dims[DIMS];
	long mean_strs[DIMS];
	long offset;
	long L;
	long T;

	complex float* y;
};

static void zmean_init(struct zmean_data* data)
{
	if (NULL != data->y)
		return;

	data->y = md_alloc(DIMS, linop_codomain(data->transform_op)->dims, CFL_SIZE);
}


static void zmean_forward(const void* _data, complex float* dst, const complex float* src)
{
	const struct zmean_data* data = _data;

	const struct linop_s* t_op = data->transform_op;
	//const long* mean_strs = data->mean_strs;
	const long* mean_dims = data->mean_dims;
	long offset = data->offset;
	long L = data->L;
	long T = data->T;


	linop_forward_unchecked(t_op, dst, src);

#if 0
	complex float* tmp = md_alloc_sameplace(DIMS, mean_dims, CFL_SIZE, src);
#else
	complex float* tmp = md_alloc(DIMS, mean_dims, CFL_SIZE);
#endif

	//double tic = timestamp();
	md_copy(1, MD_DIMS(T), tmp, src + offset, CFL_SIZE);
#if 0
	md_zaxpy2(DIMS, linop_codomain(t_op)->dims, linop_codomain(t_op)->strs, dst, 1., mean_strs, tmp);
#else
	zmean_init((struct zmean_data*)data);
	md_copy(DIMS, linop_codomain(t_op)->dims, data->y, dst, CFL_SIZE);
	for (long i = 0; i < L; i++) {
		for (long j = 0; j < T; j++) {
			data->y[i + L*j] += tmp[j] / sqrtf(L);
		}
	}
	md_copy(DIMS, linop_codomain(t_op)->dims, dst, data->y, CFL_SIZE);
#endif
	//double toc = timestamp();
	//debug_printf(DP_DEBUG2, "forward time = %f\n", toc - tic);
	

	md_free(tmp);
}

static void zmean_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct zmean_data* data = _data;

	const struct linop_s* t_op = data->transform_op;
	//const long* mean_strs = data->mean_strs;
	const long* mean_dims = data->mean_dims;
	long offset = data->offset;
	long T = data->T;
	long L = data->L;


	linop_adjoint_unchecked(t_op, dst, src);

#if 0
	complex float* tmp = md_alloc_sameplace(DIMS, mean_dims, CFL_SIZE, src);
#else
	complex float* tmp = md_alloc(DIMS, mean_dims, CFL_SIZE);
#endif

	//double tic = timestamp();
	md_clear(DIMS, mean_dims, tmp, CFL_SIZE);
#if 0
	md_zaxpy2(DIMS, linop_codomain(t_op)->dims, mean_strs, tmp, 1., linop_codomain(t_op)->strs, src);
#else
	zmean_init((struct zmean_data*)data);
	md_copy(DIMS, linop_codomain(t_op)->dims, data->y, src, CFL_SIZE);
	for (long i = 0; i < L; i++) {
		for (long j = 0; j < T; j++) {
			tmp[j] += data->y[i + j*L] / sqrtf(L);
		}
	}
#endif
	md_copy(1, MD_DIMS(T), dst + offset, tmp, CFL_SIZE);
	//double toc = timestamp();
	//debug_printf(DP_DEBUG2, "adj time = %f\n", toc - tic);

	md_free(tmp);
}

static void zmean_normal(const void* _data, complex float* dst, const complex float* src)
{
	const struct zmean_data* data = _data;
	const struct linop_s* t_op = data->transform_op;

	complex float* tmp = md_alloc_sameplace(DIMS, linop_codomain(t_op)->dims, CFL_SIZE, src);

	zmean_forward(_data, tmp, src);
	zmean_adjoint(_data, dst, tmp);

	md_free(tmp);
}

static void zmean_del(const void* _data)
{
	const struct zmean_data* data = _data;
	md_free(data->y);
	free((void*)data);
}


static struct linop_s* zmean_create(const struct linop_s* transform_op)
{
	struct zmean_data* data = xmalloc( sizeof(struct zmean_data) );

	data->transform_op = transform_op;
	data->T = linop_codomain(transform_op)->dims[TE_DIM];

	long dimsL[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, dimsL, linop_codomain(transform_op)->dims);
	data->L = md_calc_size(DIMS, dimsL);

	md_singleton_dims(DIMS, data->mean_dims);
	data->mean_dims[TE_DIM] = data->T;
	md_calc_strides(DIMS, data->mean_strs, data->mean_dims, CFL_SIZE);

	data->offset = md_calc_size(DIMS, linop_domain(transform_op)->dims); // * CFL_SIZE;

	data->y = NULL;

	long dim[DIMS];
	md_singleton_dims(DIMS, dim);
	dim[0] = md_calc_size(DIMS, linop_domain(transform_op)->dims) + data->T;
	return linop_create(DIMS, linop_codomain(transform_op)->dims, DIMS, dim, data, zmean_forward, zmean_adjoint, zmean_normal, NULL, zmean_del);
}


#if 0
struct prox_zmean_data {
	long offset;
	const struct operator_p_s* prox_op;
	long T;
	long dim[DIMS];
};

static void prox_zmean_apply(const void* _data, float mu, complex float* dst, const complex float* src)
{
	const struct prox_zmean_data* data = _data;
	operator_p_apply_unchecked(data->prox_op, mu, dst, src);
#if 0
	md_copy(1, MD_DIMS(data->T), dst + data->offset, src + data->offset, CFL_SIZE);
#else
	md_clear(1, MD_DIMS(data->T), dst + data->offset, CFL_SIZE);
#endif
}

static void prox_zmean_del(const void* _data)
{
	const struct prox_zmean_data* data = _data;
	operator_p_free(data->prox_op);
}

static const struct operator_p_s* prox_zmean_wrapper(const struct operator_p_s* prox_op, const long T)
{

	long dim[DIMS];
	md_singleton_dims(DIMS, dim);
	dim[0] = md_calc_size(operator_p_domain(prox_op)->N, operator_p_domain(prox_op)->dims) + T;

	struct prox_zmean_data* data = xmalloc( sizeof(struct prox_zmean_data) );
	data->offset = dim[0] - T;
	data->T = T;
	data->prox_op = prox_op;
	md_copy_dims(DIMS, data->dim, dim);

	return operator_p_create(DIMS, dim, DIMS, dim, data, prox_zmean_apply, prox_zmean_del);
}
#endif

struct Gop_a_data {
	long offset;
	const struct linop_s* G_op;
	long T;
	long dim[DIMS];
};

static void Gop_a_forward(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_a_data* data = _data;
	linop_forward_unchecked(data->G_op, dst, src);
	//md_clear(1, MD_DIMS(data->T), dst + data->offset, CFL_SIZE);
}

static void Gop_a_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_a_data* data = _data;
	linop_adjoint_unchecked(data->G_op, dst, src);
	md_clear(1, MD_DIMS(data->T), dst + data->offset, CFL_SIZE);
}

static void Gop_a_normal(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_a_data* data = _data;
	linop_normal_unchecked(data->G_op, dst, src);
	md_clear(1, MD_DIMS(data->T), dst + data->offset, CFL_SIZE);
}

static void Gop_a_del(const void* _data)
{
	const struct Gop_a_data* data = _data;
	linop_free(data->G_op);
}

static const struct linop_s* Gop_a_wrapper(const struct linop_s* G_op, const long T)
{

	long dim[DIMS];
	md_singleton_dims(DIMS, dim);
	dim[0] = md_calc_size(linop_domain(G_op)->N, linop_domain(G_op)->dims) + T;

	struct Gop_a_data* data = xmalloc( sizeof(struct Gop_a_data) );
	data->offset = dim[0] - T;
	data->T = T;
	data->G_op = G_op;
	md_copy_dims(DIMS, data->dim, dim);

	return linop_create(linop_codomain(G_op)->N, linop_codomain(G_op)->dims, DIMS, dim, data, Gop_a_forward, Gop_a_adjoint, Gop_a_normal, NULL, Gop_a_del);
}

#if 1
struct Gop_x_data {
	long offset;
	const struct linop_s* G_op;
	long T;
	long dim[DIMS];

	complex float* tmp;
};

static void Gop_x_init(struct Gop_x_data* data, const complex float* src)
{
	if (NULL != data->tmp)
		return;

	data->tmp = md_alloc_sameplace(1, MD_DIMS(data->T), CFL_SIZE, src);
}


static void Gop_x_forward(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_x_data* data = _data;
	Gop_x_init((struct Gop_x_data*)data, src);

	md_copy(1, MD_DIMS(data->T), data->tmp, src + data->offset, CFL_SIZE);
	linop_forward_unchecked(data->G_op, dst, data->tmp);
}

static void Gop_x_adjoint(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_x_data* data = _data;
	Gop_x_init((struct Gop_x_data*)data, src);
	md_clear(DIMS, data->dim, dst, CFL_SIZE);

	linop_adjoint_unchecked(data->G_op, data->tmp, src);
	md_copy(1, MD_DIMS(data->T), dst + data->offset, data->tmp, CFL_SIZE);
}

static void Gop_x_normal(const void* _data, complex float* dst, const complex float* src)
{
	const struct Gop_x_data* data = _data;
	Gop_x_init((struct Gop_x_data*)data, src);
	md_clear(DIMS, data->dim, dst, CFL_SIZE);
	md_copy(1, MD_DIMS(data->T), data->tmp, src + data->offset, CFL_SIZE);
	linop_normal_unchecked(data->G_op, data->tmp, data->tmp);
	md_copy(1, MD_DIMS(data->T), dst + data->offset, data->tmp, CFL_SIZE);
}

static void Gop_x_del(const void* _data)
{
	const struct Gop_x_data* data = _data;
	linop_free(data->G_op);
	md_free(data->tmp);
	free((void*)data);
}


static const struct linop_s* Gop_x_wrapper(const struct linop_s* G_op, const long T, const long x_dims[DIMS])
{

	long dim[DIMS];
	md_singleton_dims(DIMS, dim);
	dim[0] = md_calc_size(DIMS, x_dims) + T;

	struct Gop_x_data* data = xmalloc( sizeof(struct Gop_x_data) );
	data->offset = dim[0] - T;
	data->T = T;
	data->G_op = G_op;
	md_copy_dims(DIMS, data->dim, dim);
	data->tmp = NULL;

	return linop_create(linop_codomain(G_op)->N, linop_codomain(G_op)->dims, DIMS, dim, data, Gop_x_forward, Gop_x_adjoint, Gop_x_normal, NULL, Gop_x_del);
}
#endif



void jtsense_recon(struct jtsense_conf* conf,
		italgo_fun2_t italgo, void* iconf,
		_Complex float* img, _Complex float* cfimg, _Complex float* bfimg, const _Complex float* kspace,
		const long crop_dims[DIMS],
		const long map_dims[DIMS], const _Complex float* maps,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long basis_dims[DIMS], const _Complex float* basis,
		const long odict_dims[DIMS], const _Complex float* odict, 
		const _Complex float* x_truth,
		bool use_cfksp,
		const long phase_dims[DIMS], const _Complex float* phase_ref)
{

#ifdef USE_CUDA
	bool use_gpu = cuda_ondevice(kspace);
#else
	bool use_gpu = false;
#endif

	bool local_cfimg = false;
	bool local_bfimg = false;

	bool save_img = true;

	if (NULL == img)
		save_img = false;

	// -----------------------------------------------------------
	// data storage

	struct data* data = xmalloc(sizeof(struct data));

	data->conf = conf;
	data->pattern = pattern;
	data->kspace = kspace;

	bool use_odict = data->conf->use_odict;


	// -----------------------------------------------------------
	// create dims arrays
	
	// max_dims = [X Y Z C M T T B]
	long max_dims[DIMS];
	md_select_dims(DIMS, ~0, max_dims, map_dims);
	max_dims[TE_DIM] = basis_dims[TE_DIM];
	max_dims[COEFF_DIM] = data->conf->K;

	if (use_odict)
		max_dims[COEFF2_DIM] = odict_dims[COEFF2_DIM];


	long max_dims_crop[DIMS];
	md_copy_dims(DIMS, max_dims_crop, max_dims);
	md_min_dims(DIMS, FFT_FLAGS, max_dims_crop, max_dims, crop_dims);

	md_select_dims(DIMS, ~(MAPS_FLAG | COEFF_FLAG | COEFF2_FLAG), data->ksp_dims, max_dims);
	md_select_dims(DIMS, ~(COIL_FLAG | COEFF_FLAG | COEFF2_FLAG), data->img_dims, max_dims_crop);
	md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG | COEFF2_FLAG), data->cfimg_dims, max_dims_crop);

	if (use_odict)
		md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG | COEFF_FLAG), data->bfimg_dims, max_dims_crop);

	debug_printf(DP_DEBUG3, "img_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, data->img_dims);
	debug_printf(DP_DEBUG3, "cfimg_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, data->cfimg_dims);
	if (use_odict) {
		debug_printf(DP_DEBUG3, "bfimg_dims =\t");
		debug_print_dims(DP_DEBUG3, DIMS, data->bfimg_dims);
	}
	debug_printf(DP_DEBUG3, "ksp_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, data->ksp_dims);

	if (use_odict)
		md_copy_dims(DIMS, data->x_dims, data->bfimg_dims);
	else
		md_copy_dims(DIMS, data->x_dims, data->cfimg_dims);

	debug_printf(DP_DEBUG3, "x_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, data->x_dims);

	long img_full_dims[DIMS];
	md_select_dims(DIMS, ~(FFT_FLAGS), img_full_dims, data->img_dims);
	md_copy_dims(3, img_full_dims, max_dims);

	long cfimg_full_dims[DIMS];
	md_select_dims(DIMS, ~FFT_FLAGS, cfimg_full_dims, data->cfimg_dims);
	md_copy_dims(3, cfimg_full_dims, max_dims);

	//size_t x_size = md_calc_size(DIMS, data->x_dims);


	// -----------------------------------------------------------
	// sense operator: F S M R -- F.T., maps, pfov mask, real value constraint

	long max_dims_sens[DIMS];
	md_select_dims(DIMS, (FFT_FLAGS | SENS_FLAGS | COEFF_FLAG), max_dims_sens, max_dims);

	struct linop_s* sense_op = sense_op = sense_init(max_dims_sens, (FFT_FLAGS | SENS_FLAGS), maps, use_gpu);

	if (data->conf->crop) {

		struct linop_s* pfov_op = linop_resize_create(DIMS, cfimg_full_dims, data->cfimg_dims);
		struct linop_s* tmp_op = linop_chain(pfov_op, sense_op);

		linop_free(pfov_op);
		linop_free(sense_op);

		sense_op = tmp_op;
	}

	struct linop_s* rvc_op = NULL;
	if (data->conf->sconf.rvc) {

#if 0
		rvc_op = rvc_create(DIMS, data->x_dims);
		struct linop_s* tmp_op = linop_chain(rvc_op, sense_op);
#else
		long rvc_dims[DIMS];
		md_select_dims(DIMS, ~MAPS_FLAG, rvc_dims, max_dims_sens);
		rvc_op = rvc_create(DIMS, rvc_dims);
		struct linop_s* tmp_op = linop_chain(sense_op, rvc_op);
#endif

		linop_free(sense_op);
		sense_op = tmp_op;
	}


	struct linop_s* phase_op = NULL;
	if (NULL != phase_ref) {

		unsigned int phase_flags = 0;

		for (unsigned int i = 0; i < DIMS; i++) {

			if (phase_dims[i] > 1)
				phase_flags = MD_SET(phase_flags, i);
		}

#if 0
		phase_op = linop_cdiag_create(DIMS, data->x_dims, phase_flags, phase_ref); 
		struct linop_s* tmp_op = linop_chain(phase_op, sense_op);
#else
		long max_dims_phs_sens[DIMS];
		md_select_dims(DIMS, ~MAPS_FLAG, max_dims_phs_sens, max_dims_sens);
		phase_op = linop_cdiag_create(DIMS, max_dims_phs_sens, phase_flags, phase_ref); 
		struct linop_s* tmp_op = linop_chain(sense_op, phase_op);
#endif

		linop_free(sense_op);
		sense_op = tmp_op;
	}


	const struct linop_s* sample_op = NULL;
	long max_dims_pat[DIMS];

	if (!use_cfksp) {

		md_select_dims(DIMS, (FFT_FLAGS | SENS_FLAGS | TE_FLAG), max_dims_pat, max_dims);
		sample_op = sampling_create(max_dims_pat, pat_dims, pattern);
	}

	//const struct linop_s* tmp_op = linop_chain(sense_op, sample_op);
	//linop_free(sense_op);
	//linop_free(sample_op);

	//data->sense_op = sense_op;


	// -----------------------------------------------------------
	// temporal and overcomplete dictionary ops
	
	long phi_dims[DIMS];

	md_select_dims(DIMS, TE_FLAG, phi_dims, basis_dims);
	phi_dims[COEFF_DIM] = data->conf->K;

	complex float* phi = md_alloc_sameplace(DIMS, phi_dims, CFL_SIZE, basis);

	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };
	md_copy_block(DIMS, pos2, phi_dims, phi, basis_dims, basis, CFL_SIZE);

	long cfksp_dims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, cfksp_dims, data->ksp_dims);
	cfksp_dims[COEFF_DIM] = phi_dims[COEFF_DIM];

	if (NULL != basis)
		data->temporal_op = linop_matrix_altcreate(DIMS, data->img_dims, data->cfimg_dims, TE_DIM, COEFF_DIM, basis);
	else
		data->temporal_op = NULL;

	if (use_odict)
		data->odict_op = linop_matrix_altcreate(DIMS, data->cfimg_dims, data->bfimg_dims, COEFF_DIM, COEFF2_DIM, odict);
	else
		data->odict_op = NULL;

	const struct linop_s* transform_op = data->odict_op;

	struct linop_s* ktemporal_op = NULL;
	if (!use_cfksp)
		ktemporal_op = linop_matrix_altcreate(DIMS, data->ksp_dims, cfksp_dims, TE_DIM, COEFF_DIM, basis);

	data->jtsense_op = jtmodel_init(max_dims, sense_op, ktemporal_op, sample_op, pat_dims, pattern, phi_dims, phi, use_cfksp);
	md_free(phi);


	// -----------------------------------------------------------
	// initialize coefficient images

	if (NULL == cfimg) {
		local_cfimg = true;
		cfimg = md_alloc_sameplace(DIMS, data->cfimg_dims, CFL_SIZE, basis);
		md_clear(DIMS, data->cfimg_dims, cfimg, CFL_SIZE);
	}

	if (NULL == bfimg && use_odict) {
		local_bfimg = true;
		bfimg = md_alloc_sameplace(DIMS, data->bfimg_dims, CFL_SIZE, basis);
		md_clear(DIMS, data->bfimg_dims, bfimg, CFL_SIZE);
	}

	complex float* x_img = use_odict ? bfimg : cfimg;

	if (data->conf->sconf.rvc)
		linop_forward_unchecked(rvc_op, x_img, x_img);


	// -----------------------------------------------------------
	// set up proximal functions and operators

	const struct linop_s* eye = linop_identity_create(DIMS, data->x_dims);


	// -----------------------------------------------------------
	// l1-wavelet operator

	const struct linop_s* wav_op = NULL;
	const struct linop_s* l1wav_linop = NULL;

	if (data->conf->use_l1wav) {

		unsigned long flags = 0u;
		long x_l1wav_dims[DIMS];

		switch(data->conf->l1wav_dim) {

			case TE_DIM:
				md_copy_dims(DIMS, x_l1wav_dims, data->img_dims);
				flags = TE_FLAG;
				l1wav_linop = transform_op;
				break;

			case COEFF_DIM:
				md_copy_dims(DIMS, x_l1wav_dims, data->cfimg_dims);
				flags = COEFF_FLAG;
				l1wav_linop = (use_odict ? data->odict_op : eye);
				break;

			case COEFF2_DIM:
				assert(use_odict);
				md_copy_dims(DIMS, x_l1wav_dims, data->bfimg_dims);
				flags = COEFF2_FLAG;
				l1wav_linop = eye;
				break;

			default:
				error("Incorrect l1wav transform\n");
		}

		long minsize[DIMS] = MD_INIT_ARRAY(DIMS, 1);
		long course_scale[3] = MD_INIT_ARRAY(3, 16);
		md_min_dims(3, ~0u, minsize, x_l1wav_dims, course_scale);

		wav_op = wavelet_create(DIMS, x_l1wav_dims, FFT_FLAGS, minsize, data->conf->randshift, use_gpu);
		//wav_op = grad_init(DIMS, x_l1wav_dims, COEFF_FLAG);

		if (0 == data->conf->num_l1wav_lam)
			data->l1wavthresh_op = prox_unithresh_create(DIMS, wav_op, data->conf->lambda_l1wav, data->conf->jsparse ? flags : 0u, use_gpu);
		//data->l1wavthresh_op = prox_thresh_create(DIMS, linop_codomain(wav_op)->dims, data->conf->lambda_l1wav, 0u, use_gpu);
		else
		{
			assert(false == data->conf->jsparse); // can't do joint thresholding if using multiple lambdas
			data->l1wavthresh_op = prox_multithresh_create(data->conf->num_l1wav_lam, data->conf->l1wav_lambdas, wav_op, data->conf->l1wav_dim, use_gpu);
		}

		//debug_print_iovec(DP_DEBUG1, operator_p_domain(data->l1wavthresh_op));
		//debug_print_iovec(DP_DEBUG1, operator_p_codomain(data->l1wavthresh_op));
	}
	else
		data->l1wavthresh_op = NULL;


	// -----------------------------------------------------------
	// locally low rank operator

	const struct linop_s* llr_linop = NULL;
	
	if (data->conf->use_llr) {

		long x_llr_dims[DIMS];

		switch(data->conf->llr_dim) {

			case TE_DIM:
				md_copy_dims(DIMS, x_llr_dims, data->img_dims);
				llr_linop = transform_op;
				break;

			case COEFF_DIM:
				md_copy_dims(DIMS, x_llr_dims, data->cfimg_dims);
				llr_linop = (use_odict ? data->odict_op : eye);
				break;

			case COEFF2_DIM:
				assert(use_odict);
				md_copy_dims(DIMS, x_llr_dims, data->bfimg_dims);
				llr_linop = eye;
				break;

			default:
				error("Incorrect llr transform\n");
		}

		long blkdims[MAX_LEV][DIMS];
		int levels = llr_blkdims(blkdims, FFT_FLAGS, x_llr_dims, data->conf->llrblk);
		blkdims[0][MAPS_DIM] = 1;
		UNUSED(levels);
#if 1
		data->lrthresh_op = lrthresh_create(x_llr_dims, data->conf->randshift, FFT_FLAGS, (const long (*)[])blkdims, data->conf->lambda_llr, false, false, false);
#else
		data->lrthresh_op = lrthresh_create(x_llr_dims, data->conf->randshift, FFT_FLAGS, (const long (*)[])blkdims, data->conf->lambda_llr, false, false, use_gpu);
#endif
	}
	else
		data->lrthresh_op = NULL;


	// -----------------------------------------------------------
	// TV operator

	const struct linop_s* tv_linop = NULL;
	
	if (data->conf->use_tv) {

		struct linop_s* tmp_op = grad_init(DIMS, data->img_dims, TE_FLAG);
		tv_linop = linop_chain(data->temporal_op, tmp_op);
		linop_free(tmp_op);

		data->tvthresh_op	= prox_thresh_create(DIMS + 1, linop_codomain(tv_linop)->dims, data->conf->lambda_tv, MD_BIT(DIMS), use_gpu);
	}
	else
		data->tvthresh_op = NULL;


	// -----------------------------------------------------------
	// l1 operator

	const struct linop_s* l1_linop = NULL;

	if (use_odict && data->conf->lambda_odict > 0.) {
		data->l1thresh_op = prox_thresh_create(DIMS, data->bfimg_dims, data->conf->lambda_odict, 0u, use_gpu);
		l1_linop = eye;
	}
	else
		data->l1thresh_op = NULL;


	// -----------------------------------------------------------
	// zero-mean PCA: solve for "dummy variable"
	
	const struct linop_s* tz_op = NULL;
	const struct linop_s* l2z_linop = NULL;
	complex float* x = NULL;
	data->l2mean_op = NULL;

	if (data->conf->zmean)
	{
		long x_img_size = md_calc_size(DIMS, data->x_dims);
		long T = basis_dims[TE_DIM];
	
		long dim[DIMS];
		md_singleton_dims(DIMS, dim);
		dim[0] = x_img_size + T;
		x = md_alloc_sameplace(1, dim, CFL_SIZE, x_img);
#if 0
		md_copy(DIMS, data->x_dims, x, x_img, CFL_SIZE);
#else
		md_clear(DIMS, dim, x, CFL_SIZE);
#endif
#if 0
		long mean_dims[DIMS];
		complex float* xbar = load_cfl("xbar", DIMS, mean_dims);
		complex float* txbar = md_alloc(DIMS, mean_dims, CFL_SIZE);
		md_clear(DIMS, mean_dims, txbar, CFL_SIZE);
		//md_zaxpy(DIMS, mean_dims, txbar, 1., xbar);
		md_zaxpy(DIMS, mean_dims, txbar, 1. / sqrt(md_calc_size(5, data->x_dims)), xbar);
		md_copy(1, MD_DIMS(T), x + x_img_size, txbar, CFL_SIZE);
		unmap_cfl(DIMS, mean_dims, xbar);
		md_free(txbar);
#else
		md_clear(1, MD_DIMS(T), x + md_calc_size(5, data->x_dims), CFL_SIZE);
#endif
		tz_op = zmean_create(transform_op);
		//linop_free(transform_op);
		//transform_op = tmp_op;
	
		//double tic = timestamp();
		//for (int i = 0; i < 10; i++) {
		//linop_forward_unchecked(tz_op, img, x);
		//linop_adjoint_unchecked(tz_op, x, img);
		//}
		//double toc = timestamp();
		//debug_printf(DP_DEBUG1, "time: %f\n", toc - tic);
		////abort();
		//bool test_op = test_adjoint_linop(tz_op, use_gpu);
		//debug_printf(DP_DEBUG1, "test zmean_op: %s\n", test_op ? "PASS" : "FAIL");
		//linop_free(eye);
		//eye = linop_identity_create(DIMS, dim);

		long dim2[DIMS];
		md_singleton_dims(DIMS, dim2);
		dim2[TE_DIM] = T;
		data->l2mean_op = prox_leastsquares_create(DIMS, dim2, .001, NULL);
		const struct linop_s* eye2 = linop_identity_create(DIMS, dim2);
		l2z_linop = Gop_x_wrapper(eye2, T, data->x_dims);
		//UNUSED(Gop_x_wrapper);

		if (NULL != data->l1wavthresh_op) {
			l1wav_linop = Gop_a_wrapper(l1wav_linop, T);
			//test_adjoint_linop(l1wav_linop, use_gpu);
			//data->l1wavthresh_op = prox_zmean_wrapper(data->l1wavthresh_op, T);
			//l1wav_linop = eye;
		}

		if (NULL != data->lrthresh_op) {
			llr_linop = Gop_a_wrapper(llr_linop, T);
			//data->lrthresh_op = prox_zmean_wrapper(data->lrthresh_op, T);
			//llr_linop = eye;
		}

		if (NULL != data->l1thresh_op) {
			l1_linop = Gop_a_wrapper(l1_linop, T);
			//data->l1thresh_op = prox_zmean_wrapper(data->l1thresh_op, T);
			//l1_linop = eye;
		}
	}
	else {
		UNUSED(tz_op);
	}


	// -----------------------------------------------------------
	// model error operator

	const struct linop_s* modelerr_linop = NULL;
	const struct operator_p_s* l2ball_op = NULL;
	if (data->conf->modelerr > 0.) {

		modelerr_linop = modelerr_linop_create(data->cfimg_dims, data->conf->Kmodelerr);
		l2ball_op = prox_l2ball_create(DIMS, data->cfimg_dims, data->conf->modelerr, NULL);
	}


	// -----------------------------------------------------------
	// positivity constraint
	const struct linop_s* pos_linop = NULL;
	const struct operator_p_s* pos_op = NULL;
	const struct operator_p_s* rvc_pop = NULL;
	if (data->conf->positive) {

		if (use_odict)
			error("TODO: implement for odict");

		pos_op = prox_greq_create(DIMS, data->img_dims, NULL);
		pos_linop = data->temporal_op;

		rvc_pop = prox_rvc_create(DIMS, data->x_dims);
	}

	// -----------------------------------------------------------
	// set up iterative algorithm interface

#if 0
	bool regs[4] = { data->conf->use_l1wav, data->conf->use_llr, (use_odict && data->conf->lambda_odict > 0.), true };
	const struct operator_p_s* admm_funs[4] = { data->l1wavthresh_op, data->lrthresh_op, data->l1thresh_op, data->l2mean_op };
	const struct linop_s* admm_linops[4] = { l1wav_linop, llr_linop, l1_linop, l2z_linop };
	const obj_fun_t obj_funs[4] = { wavelet_l1norm, llr_nucnorm, odict_l1norm, l2mean_norm };

	unsigned int num_funs = 0;
	const struct operator_p_s* prox_ops[4];
	const struct linop_s* linops[4];
	obj_fun_t obj_funs2[4];

	for (int i = 0; i < 4; i++) {
#else

	UNUSED(l2z_linop);
	UNUSED(l2mean_norm);

	bool regs[7] = { data->conf->use_l1wav, data->conf->use_llr, (use_odict && data->conf->lambda_odict > 0.), data->conf->use_tv, data->conf->modelerr > 0., data->conf->positive, false };
	const struct operator_p_s* admm_funs[7] = { data->l1wavthresh_op, data->lrthresh_op, data->l1thresh_op, data->tvthresh_op, l2ball_op, pos_op, rvc_pop };
	const struct linop_s* admm_linops[7] = { l1wav_linop, llr_linop, l1_linop, tv_linop, modelerr_linop, pos_linop, eye };
	const obj_fun_t obj_funs[7] = { wavelet_l1norm, llr_nucnorm, odict_l1norm, NULL, NULL, NULL, NULL };

	unsigned int num_funs = 0;
	const struct operator_p_s* prox_ops[7];
	const struct linop_s* linops[7];
	obj_fun_t obj_funs2[7];

	for (int i = 0; i < 7; i++) {
#endif

		if (regs[i]) {

			prox_ops[num_funs] = admm_funs[i];
			linops[num_funs] = admm_linops[i];
			obj_funs2[num_funs] = obj_funs[i];

			debug_printf(DP_DEBUG3, "using function %d\n", i);
			//debug_print_iovec(DP_DEBUG3, linop_domain(linops[num_funs]));
			//debug_print_iovec(DP_DEBUG3, linop_codomain(linops[num_funs]));

			num_funs++;
		}
	}

	// whipe out ops and prox_funs if using conjgrad
	const struct operator_p_s** prox_ops2 = NULL;
	const struct linop_s** linops2 = NULL;

	if (num_funs > 0) {
		prox_ops2 = prox_ops;
		linops2 = linops;
	}




	// -----------------------------------------------------------
	// perform recon
	
#if 0
	if (data->conf->zmean) {
		jtsense_recon2(conf, x, italgo, iconf, data->sense_op, tz_op, num_funs, prox_ops2, linops2, obj_funs2, kspace, x_truth);
		md_copy(DIMS, data->cfimg_dims, cfimg, x, CFL_SIZE);
		dump_cfl("x_raw", linop_domain(tz_op)->N, linop_domain(tz_op)->dims, x); 
		linop_forward_unchecked(tz_op, img, x);
	}
	else
#endif
		jtsense_recon2(conf, x_img, italgo, iconf, data->jtsense_op, transform_op, num_funs, prox_ops2, linops2, obj_funs2, kspace, x_truth);

	// project back onto time series
	if (use_odict)
		linop_forward_unchecked(data->odict_op, cfimg, bfimg);

	if (save_img) {
		linop_forward_unchecked(data->temporal_op, img, cfimg);
	}


	// -----------------------------------------------------------
	// cleanup

	if (local_bfimg)
		md_free(bfimg);

	if (local_cfimg)
		md_free(cfimg);

	if (NULL != data->odict_op && NULL != data->temporal_op) {
		linop_free(data->temporal_op);
		linop_free(data->odict_op);

	}
	else
		linop_free(data->temporal_op);

	if (data->conf->modelerr > 0.) {
		linop_free(modelerr_linop);
		operator_p_free(l2ball_op);
	}

	if (data->conf->zmean)
		linop_free(tz_op);

	if (data->conf->sconf.rvc)
		linop_free(rvc_op);

	linop_free(data->jtsense_op);
	linop_free(sense_op);

	if (!use_cfksp) {
		linop_free(sample_op);
		linop_free(ktemporal_op);
	}

	linop_free(eye);

	if (data->conf->use_llr)
		operator_p_free(data->lrthresh_op);

	if (data->conf->use_l1wav) {
		operator_p_free(data->l1wavthresh_op);
		linop_free(wav_op);
	}

	if (data->conf->use_tv) {
		operator_p_free(data->tvthresh_op);
		linop_free(tv_linop);
	}

	if (use_odict && data->conf->lambda_odict > 0.)
		operator_p_free(data->l1thresh_op);

	if(data->conf->positive) {
		operator_p_free(pos_op);
		operator_p_free(rvc_pop);
	}

}


#ifdef USE_CUDA
void jtsense_recon_gpu(struct jtsense_conf* conf,
		italgo_fun2_t italgo, void* iconf,
		_Complex float* img, _Complex float* cfimg, _Complex float* bfimg, const _Complex float* kspace,
		const long crop_dims[DIMS],
		const long map_dims[DIMS], const _Complex float* maps,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long basis_dims[DIMS], const _Complex float* basis,
		const long odict_dims[DIMS], const _Complex float* odict, 
		const _Complex float* x_truth,
		bool use_cfksp,
		const long phase_dims[DIMS], const _Complex float* phase_ref)
{

	// -----------------------------------------------------------
	// create dims arrays
	
	long ksp_dims[DIMS];
	long img_dims[DIMS];
	long cfimg_dims[DIMS];
	long bfimg_dims[DIMS];

	// max_dims = [X Y Z C M T T B]
	long max_dims[DIMS];
	md_select_dims(DIMS, ~0, max_dims, map_dims);
	max_dims[TE_DIM] = basis_dims[TE_DIM];
	max_dims[COEFF_DIM] = conf->K;

	if (conf->use_odict)
		max_dims[COEFF2_DIM] = odict_dims[COEFF2_DIM];

	long max_dims_crop[DIMS];
	md_copy_dims(DIMS, max_dims_crop, max_dims);
	md_min_dims(DIMS, FFT_FLAGS, max_dims_crop, max_dims, crop_dims);

	if (use_cfksp)
		md_select_dims(DIMS, ~(MAPS_FLAG | TE_FLAG | COEFF2_FLAG), ksp_dims, max_dims);
	else
		md_select_dims(DIMS, ~(MAPS_FLAG | COEFF_FLAG | COEFF2_FLAG), ksp_dims, max_dims);

	md_select_dims(DIMS, ~(COIL_FLAG | COEFF_FLAG | COEFF2_FLAG), img_dims, max_dims_crop);
	md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG | COEFF2_FLAG), cfimg_dims, max_dims_crop);

	if (conf->use_odict)
		md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG | COEFF_FLAG), bfimg_dims, max_dims_crop);

	long x_dims[DIMS];
	md_copy_dims(DIMS, x_dims, conf->use_odict ? bfimg_dims : cfimg_dims);


	// -----------------------------------------------------------
	// allocate and copy gpu memory
	
	complex float* gpu_maps = md_gpu_move(DIMS, map_dims, maps, CFL_SIZE);
	complex float* gpu_pat = md_gpu_move(DIMS, pat_dims, pattern, CFL_SIZE);
	complex float* gpu_ksp = md_gpu_move(DIMS, ksp_dims, kspace, CFL_SIZE);

	complex float* gpu_img = md_gpu_move(DIMS, img_dims, img, CFL_SIZE);
	complex float* gpu_cfimg = md_gpu_move(DIMS, cfimg_dims, cfimg, CFL_SIZE);
	complex float* gpu_bfimg = md_gpu_move(DIMS, bfimg_dims, bfimg, CFL_SIZE);

	complex float* gpu_basis = md_gpu_move(DIMS, basis_dims, basis, CFL_SIZE);
	complex float* gpu_odict = md_gpu_move(DIMS, odict_dims, odict, CFL_SIZE);

	complex float* gpu_x_truth = md_gpu_move(DIMS, x_dims, x_truth, CFL_SIZE);

	complex float* gpu_pref = md_gpu_move(DIMS, phase_dims, phase_ref, CFL_SIZE);


	// -----------------------------------------------------------
	// run recon

	jtsense_recon(conf, italgo, iconf, gpu_img, gpu_cfimg, gpu_bfimg, gpu_ksp, crop_dims, map_dims, gpu_maps, pat_dims, gpu_pat, basis_dims, gpu_basis, odict_dims, gpu_odict, gpu_x_truth, use_cfksp, phase_dims, gpu_pref);

	if (NULL != img)
		md_copy(DIMS, img_dims, img, gpu_img, CFL_SIZE);

	if (NULL != cfimg)
		md_copy(DIMS, cfimg_dims, cfimg, gpu_cfimg, CFL_SIZE);

	if (NULL != bfimg)
		md_copy(DIMS, bfimg_dims, bfimg, gpu_bfimg, CFL_SIZE);


	// -----------------------------------------------------------
	// cleanup

	md_free((void*)gpu_maps);
	md_free((void*)gpu_pat);
	md_free((void*)gpu_ksp);

	md_free((void*)gpu_img);
	md_free((void*)gpu_cfimg);
	md_free((void*)gpu_bfimg);

	md_free((void*)gpu_basis);
	md_free((void*)gpu_odict);

	md_free((void*)gpu_x_truth);

	md_free(gpu_pref);
}
#endif


