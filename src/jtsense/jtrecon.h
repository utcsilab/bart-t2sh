

#include <stdbool.h>
#include "sense/recon.h"
#include "iter/iter2.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DIMS
#define DIMS 16
#endif


/**
 * configuration parameters for jtsense reconstruction
 *
 * @param sconf sense_conf struct for sense reconstruction
 * @param K number of temporal basis elements 
 * @param jsparse TRUE for joint-sparsity
 * @param pd TRUE for non-positive derivative projection
 */
struct jtsense_conf {

	struct sense_conf sconf;

	int K;
	_Bool jsparse;
	_Bool use_ist;
	_Bool crop;
	_Bool randshift;
	_Bool zmean;

	float modelerr;
	int Kmodelerr;

	_Bool positive;

	_Bool use_l2;
	float l2lambda;

	int num_l1wav_lam;
	float* l1wav_lambdas;

	_Bool use_l1wav;
	float lambda_l1wav;
	long l1wav_dim;

	_Bool use_llr;
	float lambda_llr;
	int llrblk;
	long llr_dim;

	_Bool use_tv;
	float lambda_tv;

	_Bool use_odict;
	float lambda_odict;

	_Bool fast;
};


typedef float (*obj_fun_t)(const struct linop_s*, const struct operator_p_s*, const complex float*);

extern const struct jtsense_conf jtsense_defaults;

extern float jt_estimate_scaling(const long dims[DIMS], const _Complex float* sens, const _Complex float* data);


void jtsense_recon(struct jtsense_conf* conf,
		italgo_fun2_t italgo, void* iconf,
		_Complex float* img, _Complex float* cfimg, _Complex float* bfimg, const _Complex float* kspace,
		const long crop_dims[DIMS],
		const long map_dims[DIMS], const _Complex float* maps,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long basis_dims[DIMS], const _Complex float* basis,
		const long odict_dims[DIMS], const _Complex float* odict, 
		const _Complex float* x_truth,
		_Bool use_cfksp,
		const long phase_dims[DIMS], const _Complex float* phase_ref);

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
		_Bool use_cfksp,
		const long phase_dims[DIMS], const _Complex float* phase_ref);
#endif


struct linop_s;
struct operator_p_s;

extern void jtsense_recon2(const struct jtsense_conf* conf, _Complex float* x_img,
		italgo_fun2_t italgo, void* iconf,
		const struct linop_s* E_op,
		const struct linop_s* T_op,
		unsigned int num_prox_funs,
		const struct operator_p_s** prox_funs,
		const struct linop_s** G_ops,
		const obj_fun_t* obj_funs,
		const _Complex float* kspace,
		const _Complex float* x_truth);

#ifdef __cplusplus
}
#endif

