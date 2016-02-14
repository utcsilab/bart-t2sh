/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

#define MAX_TRAINS 1000
#define MAX_ECHOES 500


extern void jtsense_usage(const char* name, FILE* fd);

extern void jtsense_options(const char* name, FILE* fd);

struct jtsense_conf;
extern void debug_print_jtsense_conf(int level, struct jtsense_conf* conf);

struct iter_admm_conf;
extern struct iter_admm_conf* jtsense_mmconf(unsigned int maxiter, float rho, _Bool cold_start, _Bool fast);

struct iter_conjgrad_conf;
extern struct iter_conjgrad_conf* jtsense_cgconf(unsigned int maxiter);

struct iter_fista_conf;
extern struct iter_fista_conf* jtsense_fsconf(unsigned int maxiter, float step, float continuation);

struct iter_ist_conf;
extern struct iter_ist_conf* jtsense_isconf(unsigned int maxiter, float step, float continuation);

extern int vieworder_preprocess(const char* filename, _Bool header, unsigned int echoes2skip, long dims[3], long* views);

extern void ksp_from_views(unsigned int D, unsigned int skips_start, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, long view_dims[3], const long* ksp_views, const long* dab_views);

extern int ksp_from_view_files(unsigned int D, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, unsigned int echoes2skip, unsigned int skips_start, _Bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file);

extern int wavg_ksp_from_view_files(unsigned int D, const long ksp_dims[D], _Complex float* ksp, const long dat_dims[D], const _Complex float* data, unsigned int echoes2skip, _Bool header, long Nmax, long Tmax, long T, const char* ksp_views_file, const char* dab_views_file);

extern void cfksp_from_views(unsigned int D, unsigned int skips_start, const long cfksp_dims[D], _Complex float* cfksp, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, long view_dims[3], const long* ksp_views, const long* dab_views);

extern int cfksp_from_view_files(unsigned int D, const long cfksp_dims[D], _Complex float* cfksp, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file);

extern int cfksp_pat_from_view_files(unsigned int D, const long cfksp_dims[D], _Complex float* cfksp, const long pat_dims[D], _Complex float* pattern, const long dat_dims[D], const _Complex float* data, const long bas_dims[D], const _Complex float* bas, unsigned int K, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file);

#ifdef __cplusplus
}
#endif
