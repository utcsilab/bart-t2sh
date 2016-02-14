/*
 * 2013-2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "jtsense/jtrecon.h"
#include "jtsense/common.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "misc/debug.h"


#if 1
#define CFKSP
#endif

#ifndef DIMS
#define DIMS KSPACE_DIMS
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "%s: Joint-Temporal SENSE reconstruction\n", name);
	fprintf(fd, "\n");
	jtsense_usage(name, fd);
}

static void help(const char* name, FILE* fd)
{
	usage(name, fd);
	jtsense_options(name, fd);
}



int main_jtsense(int argc, char* argv[])
{

	double start_time = timestamp();

	int mode = 1; // 0 for te_ksp, 1 for flat ksp

	bool use_gpu = false;

	bool save_img = true;

	bool save_cfimg = false;
	char cfimg_save_name[100];

	bool save_bfimg = false;
	char bfimg_save_name[100];

	bool cold_start = true;
	char x_start_fname[100];

	bool im_truth = false;
	char x_truth_fname[100];

	char odict_name[100];

	bool use_dab = false;
	char vieworder_sort_name[100];
	char vieworder_dab_name[100];
	unsigned int e2s = 0;
	unsigned int skips_start = 0;

	long crop_dims[3] = MD_INIT_ARRAY(3, 0);

	bool use_admm = false;
	int maxiter = 50;
	float admm_rho = iter_admm_defaults.rho;
	float ist_step = iter_fista_defaults.step;
	float continuation = iter_fista_defaults.continuation;

	struct jtsense_conf conf;
	struct sense_conf sconf;
	memcpy(&sconf, &sense_defaults, sizeof(struct sense_conf));
	memcpy(&conf, &jtsense_defaults, sizeof(struct jtsense_conf));
	conf.sconf = sconf;

	int pos = 0;
	int delta = 0;
	float lam = 0.f;

	bool use_kacq = false;
	int kacq_uid = 0;

	int c;
	while (-1 != (c = getopt(argc, argv, "NC:r:R:p:T:L:M:s:A:O:o:F:f:i:mhq:cu:gK:jB:tzd:D:k:"))) {
		switch(c) {
			case 'r':
				conf.use_l1wav = true;
				conf.lambda_l1wav = atof(optarg);
				break;

			case 'R':
				continuation = atof(optarg);
				break;

			case 'p':
				if (1 != sscanf(optarg, "%d:%n", &conf.num_l1wav_lam, &delta)) {
					help(argv[0], stdout);
					error("specify number of lambdas after -p\n");
				}
				conf.use_l1wav = true;
				pos += delta;
				conf.l1wav_lambdas = xmalloc(sizeof(float) * conf.num_l1wav_lam);
				for (int i = 0; i < conf.num_l1wav_lam; i++) {
					if (1 != sscanf(optarg + pos, "%f:%n", &lam, &delta)) {
						help(argv[0], stdout);
						error("incorrect format for lambdas\n");
					}
					pos += delta;
					conf.l1wav_lambdas[i] = lam;
				}
				break;

			case 'T':
				conf.use_tv = true;
				conf.lambda_tv = atof(optarg);
				break;

			case 'L':
				conf.use_llr = true;
				conf.lambda_llr = atof(optarg);
				break;

			case 'M':
				conf.llrblk = atoi(optarg);
				break;

			case 's':
				ist_step = atof(optarg);
				break;

			case 'A':
				save_cfimg = true;
				sprintf(cfimg_save_name, "%s", optarg);
				break;

			case 'C':
				conf.crop = true;
				sscanf(optarg, "%ld:%ld:%ld", &crop_dims[0], &crop_dims[1], &crop_dims[2]);
				break;

			case 'O':
				conf.use_odict = true;
				sprintf(odict_name, "%s", optarg);
				break;

			case 'o':
				conf.lambda_odict = atof(optarg);
				break;

			case 'f':
				cold_start = false;
				sprintf(x_start_fname, "%s", optarg);
				break;

			case 'F':
				im_truth = true;
				sprintf(x_truth_fname, "%s", optarg);
				break;

			case 'i':
				maxiter = atoi(optarg);
				break;

			case 'm':
				use_admm = true;
				break;

			case 'h':
				help(argv[0], stdout);
				exit(0);

			case 'q':
				conf.use_l2 = true;
				conf.l2lambda = atof(optarg);
				break;

			case 'c':
				conf.sconf.rvc = true;
				break;

			case 'u':
				admm_rho = atof(optarg);
				break;

			case 'g':
				use_gpu = true;
				break;

			case 'K':
				conf.K = atoi(optarg);
				break;

			case 'j':
				conf.jsparse = true;
				break;

			case 'B':
				save_bfimg = true;
				sprintf(bfimg_save_name, "%s", optarg);
				break;

			case 'N':
				save_img = false;
				break;

			case 't':
				conf.fast = true;
				break;

			case 'z':
				conf.zmean = true;
				break;

			case 'd':
				use_dab = true;
				e2s = atoi(optarg);
				break;

			case 'D':
				skips_start = atoi(optarg);
				assert(skips_start == 0 || skips_start == 1);
				break;

			case 'k':
				use_kacq = true;
				kacq_uid = atoi(optarg);
				break;

			default:
				usage(argv[0], stderr);
				exit(1);
		}
	}

	int nargs = argc - optind;
	if (nargs == 0) {
		usage(argv[0], stderr);
		exit(1);
	}
	else if (nargs == 4)
		mode = 0;
	else if (nargs != 5) {
		help(argv[0], stderr);
		exit(1);
	}

	if (continuation < 0. || continuation > 1.)
		error("Continuation scalar (-R) should be between 0. and 1.\n");

	if (NULL != conf.l1wav_lambdas) {
		debug_printf(DP_INFO, "Multiple (%d) lambdas specified: ", conf.num_l1wav_lam);
		for( int i = 0; i < conf.num_l1wav_lam; i++)
			debug_printf(DP_INFO, "%f ", conf.l1wav_lambdas[i]);
		debug_printf(DP_INFO, "\n");
	}

#if 1
	if (-1 == conf.l1wav_dim)
		conf.l1wav_dim = (conf.use_odict ? COEFF2_DIM : COEFF_DIM);
#else
	if (-1 == conf.l1wav_dim)
		conf.l1wav_dim = COEFF_DIM;
#endif

#if 0
	if (-1 == conf.llr_dim)
		conf.llr_dim = (conf.use_odict ? COEFF2_DIM : COEFF_DIM);
#else
	if (-1 == conf.llr_dim)
		conf.llr_dim = COEFF_DIM;
#endif

	debug_print_jtsense_conf(DP_INFO, &conf);

	long sens_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long x_seed_dims[DIMS];
	long x_truth_dims[DIMS];
	long ksp_dims[DIMS];
	long basis_dims[DIMS];
	long cfimg_dims[DIMS];
	long bfimg_dims[DIMS];
	long odict_dims[DIMS];


	// -----------------------------------------------------------
	// load data

	complex float* sens_maps = load_cfl(argv[optind + 1 + mode], DIMS, sens_dims);
	complex float* basis = load_cfl(argv[optind + 2 + mode], DIMS, basis_dims);

	complex float* pattern = NULL;
	complex float* ksp_flat = NULL;

	complex float* kspace_data = NULL;

	long ksp_flat_dims[DIMS];

	// expand kspace into time bins if flat
	if (1 == mode) {

		ksp_flat = load_cfl(argv[optind + 0], DIMS, ksp_flat_dims);

		md_copy_dims(DIMS, ksp_dims, ksp_flat_dims);
#ifdef CFKSP
		ksp_dims[COEFF_DIM] = conf.K;
#else
		ksp_dims[TE_DIM] = basis_dims[TE_DIM];
#endif

		kspace_data = md_alloc(DIMS, ksp_dims, CFL_SIZE);

		if (use_dab) {
			if (false == use_kacq) {
				sprintf(vieworder_sort_name, "%s.txt", argv[optind + 1]);
				sprintf(vieworder_dab_name, "%s_dab.txt", argv[optind + 1]);
			} else {
				sprintf(vieworder_sort_name, "%s.txt.%d", argv[optind + 1], kacq_uid);
				sprintf(vieworder_dab_name, "%s_dab.txt.%d", argv[optind + 1], kacq_uid);
			}

#ifdef CFKSP

		debug_printf(DP_DEBUG1, "Computing cfksp adjoint and estimating pattern...");


		md_select_dims(DIMS, (PHS1_FLAG | PHS2_FLAG), pat_dims, ksp_dims);
		pat_dims[TE_DIM] = basis_dims[TE_DIM];

		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);

		if (0 != cfksp_pat_from_view_files(DIMS, ksp_dims, kspace_data, pat_dims, pattern, ksp_flat_dims, ksp_flat, basis_dims, basis, conf.K, e2s, skips_start, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name)) {
				error("Error executing cfksp_pat_from_view_files\n");
			}

		debug_printf(DP_DEBUG1, " Done\n");

#else
		if( 0 != ksp_from_view_files(DIMS, ksp_dims, kspace_data, ksp_flat_dims, ksp_flat, e2s, skips_start, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name)) {
				error("Error executing ksp_from_view_files\n");
			}

			md_select_dims(DIMS, ~(SENS_FLAGS), pat_dims, ksp_dims); // copy the spatial and temporal ksp dimensions into the pattern
			pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
			estimate_pattern(DIMS, ksp_dims, COIL_DIM, pattern, kspace_data);
#endif

			fftmod(DIMS, ksp_dims, PHS2_FLAG, kspace_data, kspace_data);

		}
		else {

			pattern = load_cfl(argv[optind + 1], DIMS, pat_dims);

			long ksp_str[DIMS];
			long ksp_flat_str[DIMS];
			long pat_str[DIMS];

			md_calc_strides(DIMS, ksp_str, ksp_dims, CFL_SIZE);
			md_calc_strides(DIMS, ksp_flat_str, ksp_flat_dims, CFL_SIZE);
			md_calc_strides(DIMS, pat_str, pat_dims, CFL_SIZE);

			md_clear2(DIMS, ksp_dims, ksp_str, kspace_data, CFL_SIZE);
			md_zfmac2(DIMS, ksp_dims, ksp_str, kspace_data, pat_str, pattern, ksp_flat_str, ksp_flat);
		}

		unmap_cfl(DIMS, ksp_flat_dims, ksp_flat);
	}
	else {

		debug_printf(DP_INFO, "Legacy mode.\n");
		kspace_data = load_cfl(argv[optind + 0], DIMS, ksp_dims);

		// allocate memory and estimate kspace sampling pattern
		md_select_dims(DIMS, ~(SENS_FLAGS), pat_dims, ksp_dims); // copy the spatial and temporal ksp dimensions into the pattern
		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp_dims, COIL_DIM, pattern, kspace_data);
	}

	complex float* odict = NULL;

	md_select_dims(DIMS, ~(COIL_FLAG), img_dims, sens_dims);
	img_dims[TE_DIM] = basis_dims[TE_DIM];

	if (conf.crop)
		md_min_dims(DIMS, FFT_FLAGS, img_dims, img_dims, crop_dims);

	md_select_dims(DIMS, ~TE_FLAG, cfimg_dims, img_dims);
	cfimg_dims[COEFF_DIM] = conf.K;

	if (conf.use_odict) {
		odict = load_cfl(odict_name, DIMS, odict_dims);
		md_select_dims(DIMS, ~COEFF_FLAG, bfimg_dims, cfimg_dims);
		bfimg_dims[COEFF2_DIM] = odict_dims[COEFF2_DIM];
	}


	// -----------------------------------------------------------
	// dimensions error checking
	
	for (int i = 0; i < 4; i++) {
		if (ksp_dims[i] != sens_dims[i]) {
			error("Dimensions of kspace and sensitivities do not match!\n");
		}
	}

#ifdef CFKSP
	if (ksp_dims[COEFF_DIM] != conf.K) {
		error("Coefficient dimension of kspace and basis does not match!\n");
#else
	if (ksp_dims[TE_DIM] != basis_dims[TE_DIM]) {
		error("Temporal (TE) dimension of kspace and basis does not match!\n");
#endif
	}

	assert(1 == ksp_dims[MAPS_DIM]);

	if (conf.use_odict)
		assert(odict_dims[COEFF_DIM] == conf.K);


	// -----------------------------------------------------------
	// initialization and print info

	(use_gpu ? num_init_gpu_memopt : num_init)();

	if (sens_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", sens_dims[MAPS_DIM]);


	size_t T = md_calc_size(DIMS, pat_dims);
	long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);
	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T/(float)samples); 

	debug_printf(DP_DEBUG2, "img_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, img_dims);
	debug_printf(DP_DEBUG2, "cfimg_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, cfimg_dims);
	debug_printf(DP_DEBUG2, "ksp_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, ksp_dims);
	debug_printf(DP_DEBUG2, "sens_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, sens_dims);
	debug_printf(DP_DEBUG2, "basis_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, basis_dims);
	debug_printf(DP_DEBUG2, "pat_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, pat_dims);
	if (conf.use_odict)
	{
		debug_printf(DP_DEBUG2, "odict_dims =\t");
		debug_print_dims(DP_DEBUG2, DIMS, odict_dims);
		if (conf.use_odict) {
			debug_printf(DP_DEBUG2, "bfimg_dims =\t");
			debug_print_dims(DP_DEBUG2, DIMS, bfimg_dims);
		}
	}


	complex float* image = NULL;
	if (save_img)
		image = create_cfl(argv[optind + 3 + mode], DIMS, img_dims);

	complex float* cfimg = NULL;
	complex float* bfimg = NULL;

	if (save_cfimg)
		cfimg = create_cfl(cfimg_save_name, DIMS, cfimg_dims);
	else
		cfimg = md_alloc(DIMS, cfimg_dims, CFL_SIZE);

	if (conf.use_odict && save_bfimg)
		bfimg = create_cfl(bfimg_save_name, DIMS, bfimg_dims);
	else if (conf.use_odict)
		bfimg = md_alloc(DIMS, bfimg_dims, CFL_SIZE);


	// FFTW uses non-centered FFT, i.e. F_centered = fftmod F fftmod
	// incorporate the first mod with the sensitivities
	// incorporate the second mod with kspace
	fftmod(DIMS, sens_dims, FFT_FLAGS, sens_maps, sens_maps);
	fftmod(DIMS, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);


	float scaling = jt_estimate_scaling(ksp_dims, NULL, kspace_data);
	md_zsmul(DIMS, ksp_dims, kspace_data, kspace_data, 1. / scaling);


	// -----------------------------------------------------------
	// load initial and truth images if provided
	
	if (cold_start) {
		debug_printf(DP_INFO, "cold start\n");
		md_clear(DIMS, conf.use_odict ? bfimg_dims : cfimg_dims, conf.use_odict ? bfimg : cfimg, CFL_SIZE);
	}
	else {
		debug_printf(DP_INFO, "warm start: %s\n", x_start_fname);
		complex float* x_seed = load_cfl(x_start_fname, DIMS, x_seed_dims);

		assert(md_check_compat(DIMS, 0u, conf.use_odict ? bfimg_dims : cfimg_dims, x_seed_dims));
		md_zsmul(DIMS, conf.use_odict ? bfimg_dims : cfimg_dims, conf.use_odict ? bfimg : cfimg, x_seed, 1. / scaling);

		unmap_cfl(DIMS, x_seed_dims, x_seed);
	}


	complex float* x_truth = NULL;

	if (im_truth) {
		x_truth = load_cfl(x_truth_fname, DIMS, x_truth_dims);

		assert(md_check_compat(DIMS, 0u, conf.use_odict ? bfimg_dims : cfimg_dims, x_truth_dims));
		md_zsmul(DIMS, conf.use_odict ? bfimg_dims : cfimg_dims, x_truth, x_truth, 1. / scaling);

	}

	md_copy_dims(3, crop_dims, img_dims);


	// -----------------------------------------------------------
	// set up iter interface

	italgo_fun2_t italgo = NULL;
	void* iconf = NULL;

	if (!conf.use_l1wav && !conf.use_llr && conf.lambda_odict == 0. && !conf.use_tv) {
		italgo = iter2_conjgrad;
		iconf = jtsense_cgconf(maxiter);
	}
	else if (use_admm) {
		italgo = iter2_admm;
		iconf = jtsense_mmconf(maxiter, admm_rho, cold_start, conf.fast);
	}
	else if (conf.use_ist) {
		italgo = iter2_ist;
		iconf = jtsense_isconf(maxiter, ist_step, continuation);
	}
	else {
		italgo = iter2_fista;
		iconf = jtsense_fsconf(maxiter, ist_step, continuation);
	}


	// -----------------------------------------------------------
	// call recon
	
	if (use_gpu)
#ifdef USE_CUDA
		jtsense_recon_gpu(&conf, italgo, iconf, image, cfimg, bfimg, kspace_data, crop_dims, sens_dims, sens_maps, pat_dims, pattern, basis_dims, basis, odict_dims, odict, x_truth);
#else
		error("Recon code not compiled with CUDA.\n");
#endif
	else
		jtsense_recon(&conf, italgo, iconf, image, cfimg, bfimg, kspace_data, crop_dims, sens_dims, sens_maps, pat_dims, pattern, basis_dims, basis, odict_dims, odict, x_truth);
	

	// -----------------------------------------------------------
	// cool down

	if (save_img) {
		debug_printf(DP_INFO, "Rescaling: %f\n", scaling);
		md_zsmul(DIMS, img_dims, image, image, scaling);
		unmap_cfl(DIMS, img_dims, image);
	}

	unmap_cfl(DIMS, sens_dims, sens_maps);
	unmap_cfl(DIMS, basis_dims, basis);

	if (save_cfimg) {
		md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, scaling);
		unmap_cfl(DIMS, cfimg_dims, cfimg);
	}
	else
		md_free(cfimg);

	if (conf.use_odict && save_bfimg) {
		md_zsmul(DIMS, bfimg_dims, bfimg, bfimg, scaling);
		unmap_cfl(DIMS, bfimg_dims, bfimg);
	}
	else if (conf.use_odict)
		md_free(bfimg);

	if (0 == mode) {
		unmap_cfl(DIMS, ksp_dims, kspace_data);
		md_free(pattern);
	}
	else {
		md_free(kspace_data);
		if (! use_dab)
			unmap_cfl(DIMS, pat_dims, pattern);
	}

	if (NULL != conf.l1wav_lambdas)
		free(conf.l1wav_lambdas);

	free(iconf);

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


