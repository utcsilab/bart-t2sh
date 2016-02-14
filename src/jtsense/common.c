/*
 * 2014-2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "iter/iter.h"

#include "jtsense/jtrecon.h"

#include "common.h"



extern void jtsense_usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage 1: %s [options] <te_kspace> <sensitivities> <basis> <output>\n", name);
	fprintf(fd, "\t<te_kspace> dimensions are [X Y Z ncoils nmaps TEs]\n");
	fprintf(fd, "\n");
	fprintf(fd, "Usage 2: %s [options] <kspace> <te_mask> <sensitivities> <basis> <output>\n", name);
	fprintf(fd, "\t<kspace> dimensions are  [X Y Z ncoils nmaps]\n");
	fprintf(fd, "\t<te_mask> dimensions are [X Y Z 1 1 TEs]\n");
	fprintf(fd, "\n");
	fprintf(fd, "Usage determined by number of arguments following options.\n");
}

extern void jtsense_options(const char* name, FILE* fd)
{
	UNUSED(name);
	fprintf(fd,	"\nOptional arguments:\n"
		"\t-K <num_coeff>\t\tnumber of temporal coefficients\n"
		"\t-r <lambda>\t\tl1 wavelet regularization\n"
		"\t-j\t\t\tuse joint-l1 norm for wavelets\n"
		"\t-R <c>\t\t\tuse continuation: final lambda = c * lambda_start\n"
		"\t-M <blk>\t\tUse blocksize blk for locally low rank\n"
		"\t-L <lambda>\t\tlocally low rank regularization\n"
		"\t-T <lambda>\t\tTV regularization in time\n"
		"\t-q <l2lambda>\t\tl2 regularization\n"
		"\t-g\t\t\tuse gpu\n"
		"\t-c\t\t\treal-valued constraint\n"
		"\t-f <start_img>\t\tuse start_img as initial guess\n"
		"\t-F <truth_img>\t\tcompate to truth_img each iteration\n"
		"\t-A <alpha_cfl>\t\tstore cfimg in alpha_cfl\n"
		"\t-B <beta_cfl>\t\tstore bfimg in beta_cfl (if using odict)\n"
		"\t-O <odict>\t\tuse odict as overcomplete dictionary\n"
		"\t-o <lambda>\t\tl1 regularization of overcomplete dictionary\n"
		"\t-C <X:Y:Z>\t\tpartial fov of dimensions [X, Y, Z]\n"
		"\t-s <step>\t\tuse step size step for IST/FISTA\n"
		"\t-i <maxiter>\t\trun maxiter iterations\n"
		"\t-m\t\t\trun admm\n"
		"\t-u <rho>\t\tuse parameter rho for admm\n"
		"\t-C <X:Y:Z>\t\tpartial fov of dimensions [X, Y, Z]\n"
		"\t-t\t\t\t\"fast\" version: don't compute objval, etc.\n"
		"\t-h\t\t\tprint this help\n");
}


extern void debug_print_jtsense_conf(int level, struct jtsense_conf* conf) {
  debug_printf(level, "jtsense conf:\n");
  debug_printf(level, "\tK:             %d\n", conf->K);
  debug_printf(level, "\tjsparse:       %s\n", conf->jsparse ? "on" : "off");
  debug_printf(level, "\tuse_ist:       %s\n", conf->use_ist ? "on" : "off");
  debug_printf(level, "\trandshift:     %s\n", conf->randshift ? "on" : "off");
  debug_printf(level, "\n");
  debug_printf(level, "\tuse_l2:        %s\n", conf->use_l2 ? "on" : "off");
  debug_printf(level, "\tl2lambda:      %f\n", conf->l2lambda);
  debug_printf(level, "\n");
  debug_printf(level, "\tnum_l1wav_lam: %d\n", conf->num_l1wav_lam);
  if (conf->num_l1wav_lam > 0) {
	  for (int i = 0; i < conf->num_l1wav_lam; i++)
		  debug_printf(level, "\t lambda[%d]:   %f\n", i, conf->l1wav_lambdas[i]);
  }
  debug_printf(level, "\n");
  debug_printf(level, "\tuse_l1wav:     %s\n", conf->use_l1wav ? "on" : "off");
  debug_printf(level, "\tlambda_l1wav:  %f\n", conf->lambda_l1wav);
  debug_printf(level, "\tl1wav_dim:     %d\n", conf->l1wav_dim);
  debug_printf(level, "\n");
  debug_printf(level, "\tuse_llr:      %s\n", conf->use_llr ? "on" : "off");
  debug_printf(level, "\tlambda_llr:   %f\n", conf->lambda_llr);
  debug_printf(level, "\tllrblk:       %d\n", conf->llrblk);
  debug_printf(level, "\tllr_dim:      %d\n", conf->llr_dim);
  debug_printf(level, "\n");
  debug_printf(level, "\tuse_odict:    %s\n", conf->use_odict ? "on" : "off");
  debug_printf(level, "\tlambda_odict: %f\n", conf->lambda_odict);

  debug_printf(level, "\n");
  debug_print_sense_conf(level, &conf->sconf);
  debug_printf(level, "\n\n");
}


struct iter_admm_conf* jtsense_mmconf(unsigned int maxiter, float rho, bool cold_start, bool fast)
{
	struct iter_admm_conf* iconf = xmalloc( sizeof(struct iter_admm_conf) );
	memcpy(iconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));

	iconf->maxiter = maxiter;
	iconf->rho = rho;
	iconf->do_warmstart = !cold_start;
	iconf->fast = fast;
	iconf->hogwild = true;

	return iconf;
}

struct iter_conjgrad_conf* jtsense_cgconf(unsigned int maxiter)
{
	struct iter_conjgrad_conf* iconf = xmalloc( sizeof(struct iter_conjgrad_conf) );
	memcpy(iconf, &iter_conjgrad_defaults, sizeof(struct iter_conjgrad_conf));

	iconf->maxiter = maxiter;

	return iconf;
}

struct iter_fista_conf* jtsense_fsconf(unsigned int maxiter, float step, float continuation)
{
	struct iter_fista_conf* iconf = xmalloc( sizeof(struct iter_fista_conf) );
	memcpy(iconf, &iter_fista_defaults, sizeof(struct iter_fista_conf));

	iconf->maxiter = maxiter;
	iconf->step = step;
	iconf->continuation = continuation;
	iconf->hogwild = true;

	return iconf;
}

struct iter_ist_conf* jtsense_isconf(unsigned int maxiter, float step, float continuation)
{
	struct iter_ist_conf* iconf = xmalloc( sizeof(struct iter_ist_conf) );
	memcpy(iconf, &iter_ist_defaults, sizeof(struct iter_ist_conf));

	iconf->maxiter = maxiter;
	iconf->step = step;
	iconf->continuation = continuation;

	return iconf;
}


int vieworder_preprocess(const char* filename, bool header, unsigned int echoes2skip, long dims[3], long* views)
{

	long strs[3];
	md_calc_strides(3, strs, dims, sizeof(long));

	long pos[3];
	md_set_dims(3, pos, 0);


	FILE *fd;
	char line_buffer[BUFSIZ];
	long line_number = 0;

	fd = fopen(filename, "r");
	if (!fd)
		error("Couldn't open file %s for reading.\n", filename);

	if (0 == fgets(line_buffer, sizeof(line_buffer), fd))
		return -1;

	if (header) {
		if (0 != sscanf(line_buffer, "index train echo y z\n"))
			return -1;
	}

	for (int i = 0; i < md_calc_size(3, dims); i++)
		views[i] = -1;

	long trash;
	long i;
	long train;
	long echo;
	long ky;
	long kz;
	long idx;

	while (fgets(line_buffer, sizeof(line_buffer), fd)) {

		if (5 == (i = sscanf(line_buffer, "%ld %ld %ld %ld %ld \n", &trash, &train, &echo, &ky, &kz)) ){
			//debug_printf(DP_DEBUG3, "train=%ld\tte=%ld\tky=%ld\tkz=%d\n", train, te, ky, kz);
			if (ky != -1 && kz != -1 && echo >= echoes2skip) {

				echo -= echoes2skip;

				pos[0] = train;
				pos[1] = echo;

				pos[2] = 0;
				idx = md_calc_offset(3, strs, pos);
				views[idx / sizeof(long)] = ky;

				pos[2] = 1;
				idx = md_calc_offset(3, strs, pos);
				views[idx / sizeof(long)] = kz;
			}
		}
		else
			return -1;

		++line_number;
	}

	fclose(fd);
	return 0;
}


void ksp_from_views(unsigned int D, unsigned int skips_start, const long ksp_dims[D], complex float* ksp, const long dat_dims[D], const complex float* data, long view_dims[3], const long* ksp_views, const long* dab_views)
{

	//debug_print_dims(DP_DEBUG1, 3, view_dims);
	//debug_print_dims(DP_DEBUG1, D, ksp_dims);

#if 0
	assert(view_dims[1] == ksp_dims[TE_DIM]);
#endif
	assert(view_dims[2] == 2);

	assert(skips_start == 0 || skips_start == 1);

	long ksp_strs[D];
	long dat_strs[D];
	long view_strs[3];

	long ksp_pos[D];
	long dat_pos[D];
	long view_pos[3];

	md_calc_strides(D, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(D, dat_strs, dat_dims, CFL_SIZE);
	md_calc_strides(3, view_strs, view_dims, sizeof(long));

	md_set_dims(D, ksp_pos, 0);
	md_set_dims(D, dat_pos, 0);
	md_set_dims(3, view_pos, 0);

	long N = view_dims[0];
	long T = ksp_dims[TE_DIM];

	md_clear(D, ksp_dims, ksp, CFL_SIZE);

#if 0
	complex float* ksp_views_cfl = md_alloc(3, view_dims, CFL_SIZE);
	complex float* dab_views_cfl = md_alloc(3, view_dims, CFL_SIZE);
	for (int i = 0; i < md_calc_size(3, view_dims); i++) {
		ksp_views_cfl[i] = ksp_views[i];
		dab_views_cfl[i] = dab_views[i];
	}
	dump_cfl("ksp_views", 3, view_dims, (const complex float*)ksp_views_cfl);
	dump_cfl("dab_views", 3, view_dims, (const complex float*)dab_views_cfl);
	md_free(ksp_views_cfl);
	md_free(dab_views_cfl);
#endif

	for (long train = 0; train < N; train++) {

		for (long echo = 0; echo < T + skips_start; echo++) {

			view_pos[0] = train;
			view_pos[1] = echo;

			view_pos[2] = 0;
			long view_idx = md_calc_offset(3, view_strs, view_pos);

			if (dab_views[view_idx / sizeof(long)] >= 0 && ksp_views[view_idx / sizeof(long)]  >= 0) {

				dat_pos[PHS1_DIM] = dab_views[view_idx / sizeof(long)];
				ksp_pos[PHS1_DIM] = ksp_views[view_idx / sizeof(long)];

				view_pos[2] = 1;
				view_idx = md_calc_offset(3, view_strs, view_pos);

				dat_pos[PHS2_DIM] = dab_views[view_idx / sizeof(long)];
				ksp_pos[PHS2_DIM] = ksp_views[view_idx / sizeof(long)];

				ksp_pos[TE_DIM] = echo;

				if (skips_start == 1 && echo != 0)
					ksp_pos[TE_DIM]--;

				long copy_dims[D];

				md_select_dims(D, READ_FLAG | COIL_FLAG, copy_dims, ksp_dims);

				long ksp_idx = md_calc_offset(D, ksp_strs, ksp_pos);
				long dat_idx = md_calc_offset(D, dat_strs, dat_pos);

				md_copy2(D, copy_dims, ksp_strs, ksp + ksp_idx / sizeof(long), dat_strs, data + dat_idx / sizeof(long), CFL_SIZE);
			}
		}
	}
}

/**
 * ksp_dims: [1 Y Z C 1 T]
 * dat_dims: [1 Y Z C 1 1]
 */
int ksp_from_view_files(unsigned int D, const long ksp_dims[D], complex float* ksp, const long dat_dims[D], const complex float* data, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file)
{
	long view_dims[3] = { Nmax, Tmax, 2 };
	long* ksp_views = md_alloc(3, view_dims, sizeof(long));
	long* dab_views = md_alloc(3, view_dims, sizeof(long));

	assert(0 == skips_start || 1 == skips_start);

	if (0 != vieworder_preprocess(ksp_views_file, header, echoes2skip - skips_start, view_dims, ksp_views))
		return -1;

	if (0 != vieworder_preprocess(dab_views_file, header, echoes2skip - skips_start, view_dims, dab_views))
		return -1;

	ksp_from_views(D, skips_start, ksp_dims, ksp, dat_dims, data, view_dims, ksp_views, dab_views);

	return 0;

}


/**
 * ksp_dims: [X Y Z C 1 1]
 * dat_dims: [X Y Z C 1 1]
 */
int wavg_ksp_from_view_files(unsigned int D, const long ksp_dims[D], complex float* ksp, const long dat_dims[D], const complex float* data, unsigned int echoes2skip, bool header, long Nmax, long Tmax, long T, const char* ksp_views_file, const char* dab_views_file)
{
	long view_dims[3] = { Nmax, Tmax, 2 };
	long* ksp_views = md_alloc(3, view_dims, sizeof(long));
	long* dab_views = md_alloc(3, view_dims, sizeof(long));

	assert(md_check_compat(D, 0u, ksp_dims, dat_dims));

	if (0 != vieworder_preprocess(ksp_views_file, header, echoes2skip, view_dims, ksp_views))
		return -1;

	if (0 != vieworder_preprocess(dab_views_file, header, echoes2skip, view_dims, dab_views))
		return -1;

	// [1 Y Z C 1 T]
	long te_dims[D];
	md_select_dims(D, ~READ_FLAG, te_dims, ksp_dims);
	te_dims[TE_DIM] = T;

	// [1 Y Z C 1 1]
	long ksp1_dims[D];
	md_select_dims(D, ~READ_FLAG, ksp1_dims, ksp_dims);

	long pos[D];
	md_set_dims(D, pos, 0);

	long ksp1_str[D];
	md_calc_strides(D, ksp1_str, ksp1_dims, CFL_SIZE);

	long te_str[D];
	md_calc_strides(D, te_str, te_dims, CFL_SIZE);

	// manually copy the 0th slice to initialize weights
	complex float* tmp_te = md_alloc(D, te_dims, CFL_SIZE);
	complex float* tmp1 = md_alloc(D, ksp1_dims, CFL_SIZE);

	md_clear(D, te_dims, tmp_te, CFL_SIZE);
	md_copy_block(D, pos, ksp1_dims, tmp1, dat_dims, data, CFL_SIZE);

	ksp_from_views(D, 0, te_dims, tmp_te, ksp1_dims, tmp1, view_dims, ksp_views, dab_views);

	complex float* weights = md_alloc(D, ksp1_dims, CFL_SIZE);
	md_zwavg2_core1(D, te_dims, TE_FLAG, ksp1_str, weights, te_str, tmp_te);

	md_free(tmp_te);
	md_free(tmp1);

	int counter = 0;

#pragma omp parallel for
	for (long i = 0; i < ksp_dims[READ_DIM]; i++) {

		complex float* ksp_te = md_alloc(D, te_dims, CFL_SIZE);
		complex float* ksp1 = md_alloc(D, ksp1_dims, CFL_SIZE);

		long pos1[D];
		md_set_dims(D, pos1, 0);
		pos1[READ_DIM] = i;

		md_copy_block(D, pos1, ksp1_dims, ksp1, dat_dims, data, CFL_SIZE);

		ksp_from_views(D, 0, te_dims, ksp_te, ksp1_dims, ksp1, view_dims, ksp_views, dab_views);

		md_zwavg2_core2(D, te_dims, TE_FLAG, ksp1_str, ksp1, weights, te_str, ksp_te);

		md_copy_block(D, pos1, ksp_dims, ksp, ksp1_dims, ksp1, CFL_SIZE);

		md_free(ksp_te);
		md_free(ksp1);

#pragma omp critical
		{ debug_printf(DP_DEBUG4, "%04d/%04ld    \n", ++counter, ksp_dims[READ_DIM]); }
	}

	md_free(weights);

	return 0;
}


int cfksp_from_view_files(unsigned int D, const long cfksp_dims[D], complex float* cfksp, const long dat_dims[D], const complex float* data, const long bas_dims[D], const complex float* bas, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file)
{
	long view_dims[3] = { Nmax, Tmax, 2 };
	long* ksp_views = md_alloc(3, view_dims, sizeof(long));
	long* dab_views = md_alloc(3, view_dims, sizeof(long));

	if (0 != vieworder_preprocess(ksp_views_file, header, echoes2skip - skips_start, view_dims, ksp_views))
		return -1;

	if (0 != vieworder_preprocess(dab_views_file, header, echoes2skip - skips_start, view_dims, dab_views))
		return -1;

	assert(skips_start == 0 || skips_start == 1);

	if (skips_start == 1) { 
		// we want to merge the first echoes, but there may be duplicates. We need to zero out any duplicates (would be
		// better to average...)

		long view_strs[3];

		long view_pos[3];
		long view_pos2[3];

		md_calc_strides(3, view_strs, view_dims, sizeof(long));

		md_set_dims(3, view_pos, 0);
		md_set_dims(3, view_pos2, 0);

		long N = view_dims[0];

		long phs0y = -1;
		long phs0z = -1;
		long phs1y = -1;
		long phs1z = -1;

		for (long train1 = 0; train1 < N;  train1++) {

			view_pos[0] = train1;
			view_pos[1] = 0;

			view_pos[2] = 0;
			phs0y = md_calc_offset(3, view_strs, view_pos);

			view_pos[2] = 1;
			phs0z = md_calc_offset(3, view_strs, view_pos);

			if (ksp_views[phs0y / sizeof(long)] > -1 && ksp_views[phs0z / sizeof(long)] > -1) {

				for (long train2 = 0; train2 < N; train2++) {

					view_pos[0] = train2;
					view_pos[1] = 1;

					view_pos[2] = 0;
					phs1y = md_calc_offset(3, view_strs, view_pos);

					view_pos[2] = 1;
					phs1z = md_calc_offset(3, view_strs, view_pos);

					if ( ksp_views[phs1y / sizeof(long)] > -1 && ksp_views[phs1z / sizeof(long)] > -1 ) {

						if ( (ksp_views[phs0y / sizeof(long)] == ksp_views[phs1y / sizeof(long)]) && (ksp_views[phs0z / sizeof(long)] == ksp_views[phs1z / sizeof(long)]) ) {

							//debug_printf(DP_DEBUG1, "match: ky = %d, kz = %d\n", ksp_views[phs0y / sizeof(long)], ksp_views[phs0z / sizeof(long)]);
							// for now, keep the skipped echo and throw out the original TE sample...
							ksp_views[phs1y / sizeof(long)] = -1;
							ksp_views[phs1z / sizeof(long)] = -1;
						}
					}
				}
			}
		}
	}

	cfksp_from_views(D, skips_start, cfksp_dims, cfksp, dat_dims, data, bas_dims, bas, view_dims, ksp_views, dab_views);

	return 0;

}


/**
 * Implements Phi^H P y without first expanding (zero-filling) y into the time dimension
 * @param cfksp output coefficient kspace
 */
void cfksp_from_views(unsigned int D, unsigned int skips_start, const long cfksp_dims[D], complex float* cfksp, const long dat_dims[D], const complex float* data, const long bas_dims[D], const complex float* bas, long view_dims[3], const long* ksp_views, const long* dab_views)
{
	
#if 0
	assert(view_dims[1] == ksp_dims[TE_DIM]);
#endif
	assert(view_dims[2] == 2);

	long cfksp_strs[D];
	long dat_strs[D];
	long bas_strs[D];
	long view_strs[3];

	long cfksp_pos[D];
	long dat_pos[D];
	long bas_pos[D];
	long view_pos[3];
	long view_pos2[3];

	md_calc_strides(D, cfksp_strs, cfksp_dims, CFL_SIZE);
	md_calc_strides(D, dat_strs, dat_dims, CFL_SIZE);
	md_calc_strides(D, bas_strs, bas_dims, CFL_SIZE);
	md_calc_strides(3, view_strs, view_dims, sizeof(long));

	md_set_dims(D, cfksp_pos, 0);
	md_set_dims(D, dat_pos, 0);
	md_set_dims(D, bas_pos, 0);
	md_set_dims(3, view_pos, 0);
	md_set_dims(3, view_pos2, 0);

	long N = view_dims[0];
	long T = bas_dims[TE_DIM];

	assert(skips_start == 0 || skips_start == 1);

	md_clear(D, cfksp_dims, cfksp, CFL_SIZE);

	for (long train = 0; train < N; train++) {

		for (long echo = 0; echo < T + skips_start; echo++) {

			view_pos[0] = train;
			view_pos[1] = echo;

			view_pos[2] = 1;
			long view_idx = md_calc_offset(3, view_strs, view_pos);

			view_pos2[0] = train;
			view_pos2[1] = 0;
			view_pos2[2] = 1;


			view_pos[2] = 0;
			view_pos2[2] = 0;
			view_idx = md_calc_offset(3, view_strs, view_pos);


			if (dab_views[view_idx / sizeof(long)] >= 0 && ksp_views[view_idx / sizeof(long)]  >= 0) {

				bas_pos[TE_DIM] = echo;

				if (skips_start == 1 && echo != 0)
					bas_pos[TE_DIM]--;
		
				dat_pos[PHS1_DIM] = dab_views[view_idx / sizeof(long)];
				cfksp_pos[PHS1_DIM] = ksp_views[view_idx / sizeof(long)];

				view_pos[2] = 1;
				view_idx = md_calc_offset(3, view_strs, view_pos);

				dat_pos[PHS2_DIM] = dab_views[view_idx / sizeof(long)];
				cfksp_pos[PHS2_DIM] = ksp_views[view_idx / sizeof(long)];

				long copy_dims[D];

				md_select_dims(D, READ_FLAG | COIL_FLAG | COEFF_FLAG, copy_dims, cfksp_dims);

				long cfksp_idx = md_calc_offset(D, cfksp_strs, cfksp_pos);
				long dat_idx = md_calc_offset(D, dat_strs, dat_pos);
				long bas_idx = md_calc_offset(D, bas_strs, bas_pos);

				md_zfmacc2(D, copy_dims, cfksp_strs, cfksp + cfksp_idx / sizeof(long), dat_strs, data + dat_idx / sizeof(long), bas_strs, bas + bas_idx / sizeof(long));
			}
		}

	}

}



int cfksp_pat_from_view_files(unsigned int D, const long cfksp_dims[D], complex float* cfksp, const long pat_dims[D], complex float* pattern, const long dat_dims[D], const complex float* data, const long bas_dims[D], const complex float* bas, unsigned int K, unsigned int echoes2skip, unsigned int skips_start, bool header, long Nmax, long Tmax, const char* ksp_views_file, const char* dab_views_file)
{

	long pos2[D];
	long phi_dims[D];

	for (unsigned int i = 0; i < D; i++)
		pos2[i] = 0;

	md_select_dims(D, TE_FLAG, phi_dims, bas_dims);
	phi_dims[COEFF_DIM] = K;

	complex float* phi = md_alloc(D, phi_dims, CFL_SIZE);

	md_copy_block(D, pos2, phi_dims, phi, bas_dims, bas, CFL_SIZE);

	// 0 <= skips_start <= echoes2skip
	// can be used to merge the skipped echoes into the first TE that we keep.
	// for now can only be zero (usual thing) or one
	assert(skips_start == 0 || skips_start == 1);

	// compute compact cfksp directly from view files
	if ( 0 != cfksp_from_view_files(D, cfksp_dims, cfksp, dat_dims, data, phi_dims, phi, echoes2skip, skips_start, header, Nmax, Tmax, ksp_views_file, dab_views_file)) {
		error("Error executing cfksp_from_view_files\n");
	}

	md_free(phi);

	// expand one slice into time bins for the pattern
	long single_ksp_dims[D];
	long pat_flat_dims[D];

	md_select_dims(D, ~READ_FLAG, single_ksp_dims, dat_dims);
	md_select_dims(D, ~(TE_FLAG), pat_flat_dims, pat_dims);

	complex float* pattern2 = md_alloc(D, pat_dims, CFL_SIZE);
	complex float* pat_flat = md_alloc(D, pat_flat_dims, CFL_SIZE);
	complex float* single_ksp = md_alloc(D, single_ksp_dims, CFL_SIZE);

	md_copy_block(D, pos2, single_ksp_dims, single_ksp, dat_dims, data, CFL_SIZE);

	estimate_pattern(D, single_ksp_dims, COIL_DIM, pat_flat, single_ksp);

	if( 0 != ksp_from_view_files(D, pat_dims, pattern2, pat_flat_dims, pat_flat, echoes2skip, skips_start, header, Nmax, Tmax, ksp_views_file, dab_views_file)) {
		error("Error executing ksp_from_view_files\n");
	}

	estimate_pattern(D, pat_dims, COIL_DIM, pattern, pattern2);

	md_free(pattern2);
	md_free(single_ksp);
	md_free(pat_flat);

	return 0;
}

