

#include <stdbool.h>
#include "sense/recon.h"
#include "iter/iter2.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DIMS
#define DIMS 16
#endif

#define MAX_TRAINS 1000
#define MAX_ECHOES 500


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
	unsigned int K;
	_Bool fast;
};


typedef float (*obj_fun_t)(const struct linop_s*, const struct operator_p_s*, const complex float*);

extern const struct jtsense_conf jtsense_defaults;

extern float jt_estimate_scaling(const long dims[DIMS], const long flags, const _Complex float* sens, const _Complex float* data);


struct linop_s;
struct operator_p_s;
struct operator_s;

extern const struct operator_s* operator_t2sh_pics_create(struct jtsense_conf* conf,
		italgo_fun2_t italgo, iter_conf* iconf,
		const struct linop_s* E_op,
		unsigned int num_prox_funs,
		const struct operator_p_s* prox_funs[num_prox_funs],
		const struct linop_s* G_ops[num_prox_funs],
		const obj_fun_t obj_funs[num_prox_funs],
		const _Complex float* cfimg_truth,
		_Bool use_gpu);


extern int vieworder_preprocess(const char* filename, _Bool header, unsigned int echoes2skip, long dims[3], long* views);

extern int TR_vals_preprocess(const char* filename, const _Bool header, const long Nmax, long* TR_vals, long* TR_idx);

extern int ksp_varTR_from_view_files(unsigned int D, const long ksp_cat_dims[D], const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, unsigned int echoes2skip, unsigned int skips_start, _Bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file, const char* TR_vals_file);

extern void ksp_from_views(unsigned int D, unsigned int skips_start, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, long view_dims[3], const long* ksp_views, const long* dab_views, const long* TR_idx);

extern int ksp_from_view_files(unsigned int D, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, unsigned int echoes2skip, unsigned int skips_start, _Bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file, const char* TR_vals_file);

extern int avg_ksp_from_view_files(unsigned int D, _Bool wavg, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, unsigned int echoes2skip, _Bool header, long Nmax, long Tmax, long T, const char* ksp_views_file, const char* dab_views_file, const char* TR_vals_file);

extern void cfksp_from_views(unsigned int D, unsigned int skips_start, const long cfksp_dims[D], _Complex float* cfksp, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, long view_dims[3], const long* ksp_views, const long* dab_views, const long* TR_idx);

extern int cfksp_from_view_files(unsigned int D, const long cfksp_dims[D], _Complex float* cfksp, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file, const char* TR_vals_file);

extern int cfksp_pat_from_view_files(unsigned int D, const long cfksp_dims[D], _Complex float* cfksp, const long pat_dims[D], _Complex float* pattern, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, unsigned int K, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file);

extern void dat_from_views(unsigned int D, const long dat_dims[D], _Complex float* dat, const long ksp_dims[D], const _Complex float* ksp, long view_dims[3], const long* ksp_views, const long* dab_views);

extern int dat_from_view_files(unsigned int D, const long dat_dims[D], _Complex float* dat, const long ksp_dims[D], const _Complex float* ksp, _Bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file);



#ifdef __cplusplus
}
#endif

