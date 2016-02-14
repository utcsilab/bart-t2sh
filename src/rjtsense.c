/*
 * 2013-2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <tgmath.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/gpuops.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "misc/debug.h"

#include "jtsense/jtrecon.h"
#include "jtsense/common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef DIMS
#define DIMS KSPACE_DIMS
#endif

#ifdef USE_CUDA
#define MAX_CUDA_DEVICES 16
#ifdef _OPENMP
omp_lock_t gpulock[MAX_CUDA_DEVICES];
#endif
#endif

#if 0
#define SINGLE_SLICE
#endif

#if 0
#define GPU_CPU
#endif

#if 1
#define CFKSP
#endif


extern bool num_auto_parallelize; // FIXME

static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "%s: Joint-Temporal SENSE slice-by-slice reconstruction\n", name);
	fprintf(fd, "The read (0th) dimension is Fourier transformed and each section\n"
			"perpendicular to this dimension is reconstructed separately.\n");
	fprintf(fd, "\n");
	jtsense_usage(name, fd);
}


static void help(const char* name, FILE* fd)
{
	usage(name, fd);
	jtsense_options(name, fd);
}

int main_rjtsense(int argc, char* argv[])
{

#ifdef USE_CUDA
	cuda_memcache_off();
#endif

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

	bool load_ksp_wavg = false;
	char ksp_wavg_name[100];

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
	while (-1 != (c = getopt(argc, argv, "W:NC:r:R:p:T:L:M:s:A:O:o:F:f:i:mhq:cu:gK:jB:td:D:k:"))) {
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

			case 't':
				conf.fast = true;
				break;

			case 'd':
				use_dab = true;
				e2s = atoi(optarg);
				break;

			case 'D':
				skips_start = atoi(optarg);
				assert(skips_start == 0 || skips_start == 1);
				break;

			case 'N':
				save_img = false;
				break;

			case 'W':
				load_ksp_wavg = true;
				sprintf(ksp_wavg_name, "%s", optarg);
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

	if (-1 == conf.l1wav_dim)
		conf.l1wav_dim = (conf.use_odict ? COEFF2_DIM : COEFF_DIM);

	if (-1 == conf.llr_dim)
		conf.llr_dim = (conf.use_odict ? COEFF2_DIM : COEFF_DIM);

	debug_print_jtsense_conf(DP_INFO, &conf);

	long sens_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long x_seed_dims[DIMS];
	//long x_truth_dims[DIMS];
	long basis_dims[DIMS];
	long cfimg_dims[DIMS];
	long bfimg_dims[DIMS];
	long odict_dims[DIMS];
	long ksp_flat_dims[DIMS];
	long ksp_wavg_dims[DIMS];
	long ksp_full_dims[DIMS];

	long ksp_full_strs[DIMS];
	long ksp_flat_strs[DIMS];
	long img_strs[DIMS];
	long x_seed_strs[DIMS];
	long sens_strs[DIMS];
	long pat_strs[DIMS];
	long bas_strs[DIMS];
	long odict_strs[DIMS];
	long cfimg_strs[DIMS];
	long bfimg_strs[DIMS];


	// -----------------------------------------------------------
	// load data

	complex float* sens_maps = load_cfl(argv[optind + 1 + mode], DIMS, sens_dims);
	complex float* basis = load_cfl(argv[optind + 2 + mode], DIMS, basis_dims);

	complex float* ksp_full_data = NULL;
	complex float* ksp_flat_data = NULL;
	complex float* pattern = NULL;

	if (0 == mode) {

		debug_printf(DP_INFO, "Legacy mode.\n");

		ksp_full_data = load_cfl(argv[optind + 0], DIMS, ksp_full_dims);
		md_select_dims(DIMS, ~(READ_FLAG | SENS_FLAGS), pat_dims, ksp_full_dims);
		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		long tmp_dims[DIMS];
		md_select_dims(DIMS, ~READ_FLAG, tmp_dims, ksp_full_dims);
		complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE); 
		long pos[DIMS];
		md_set_dims(DIMS, pos, 0);
		md_zrss(DIMS, ksp_full_dims, READ_FLAG, tmp, ksp_full_data);
		estimate_pattern(DIMS, tmp_dims, COIL_DIM, pattern, tmp);
		md_free(tmp);

		md_select_dims(DIMS, ~TE_FLAG, ksp_flat_dims, ksp_full_dims);
		ksp_flat_data = md_alloc(DIMS, ksp_flat_dims, CFL_SIZE);
		md_zrss(DIMS, ksp_full_dims, TE_FLAG, ksp_flat_data, ksp_full_data);
	}
	else {

		ksp_flat_data = load_cfl(argv[optind + 0], DIMS, ksp_flat_dims);

		if (use_dab) {

			if (false == use_kacq) {
				sprintf(vieworder_sort_name, "%s.txt", argv[optind + 1]);
				sprintf(vieworder_dab_name, "%s_dab.txt", argv[optind + 1]);
			} else {
				sprintf(vieworder_sort_name, "%s.txt.%d", argv[optind + 1], kacq_uid);
				sprintf(vieworder_dab_name, "%s_dab.txt.%d", argv[optind + 1], kacq_uid);
			}

			md_select_dims(DIMS, ~(READ_FLAG | SENS_FLAGS), pat_dims, ksp_flat_dims);
			pat_dims[TE_DIM] = basis_dims[TE_DIM];
		}
		else
			pattern = load_cfl(argv[optind + 1], DIMS, pat_dims);
	}

	complex float* odict = NULL;

	md_select_dims(DIMS, ~(COIL_FLAG), img_dims, sens_dims);
	img_dims[TE_DIM] = basis_dims[TE_DIM];

	if (conf.crop)
		md_min_dims(DIMS, FFT_FLAGS, img_dims, img_dims, crop_dims);

	md_select_dims(DIMS, ~TE_FLAG, cfimg_dims, img_dims);
	cfimg_dims[COEFF_DIM] = conf.K;

	if (conf.use_odict)
		odict = load_cfl(odict_name, DIMS, odict_dims);

	if (conf.use_odict && save_bfimg) {
		md_select_dims(DIMS, ~COEFF_FLAG, bfimg_dims, cfimg_dims);
		bfimg_dims[COEFF2_DIM] = odict_dims[COEFF2_DIM];
	}

	// -----------------------------------------------------------
	// dimensions error checking

	for (int i = 0; i < 4; i++) {
		if (ksp_flat_dims[i] != sens_dims[i])
			error("Dimensions of kspace and sensitivities do not match!\n");
	}

	assert(1 == ksp_flat_dims[MAPS_DIM]);

	if (pat_dims[TE_DIM] != basis_dims[TE_DIM])
		error("Temporal (TE) dimension of pattern and basis does not match!\n");

	if (conf.use_odict)
		assert(odict_dims[COEFF_DIM] == conf.K);


	// -----------------------------------------------------------
	// initialization and print info

	(use_gpu ? num_init_gpu : num_init)();

	if (sens_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", sens_dims[MAPS_DIM]);

	debug_printf(DP_DEBUG3, "img_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, img_dims);
	debug_printf(DP_DEBUG3, "cfimg_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, cfimg_dims);
	debug_printf(DP_DEBUG3, "ksp_flat_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, ksp_flat_dims);
	debug_printf(DP_DEBUG3, "sens_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, sens_dims);
	debug_printf(DP_DEBUG3, "basis_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, basis_dims);
	debug_printf(DP_DEBUG3, "pat_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, pat_dims);
	if (conf.use_odict)
	{
		debug_printf(DP_DEBUG3, "odict_dims =\t");
		debug_print_dims(DP_DEBUG3, DIMS, odict_dims);
		if (save_bfimg) {
			debug_printf(DP_DEBUG3, "bfimg_dims =\t");
			debug_print_dims(DP_DEBUG3, DIMS, bfimg_dims);
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

		float scaling = 1.;

		complex float* ksp_wavg_data = NULL;
		if (use_dab) {

			if (false == load_ksp_wavg) {
				double tic = timestamp();
				debug_printf(DP_DEBUG1, "Averaging data for scaling\n");
				ksp_wavg_data = md_alloc(DIMS, ksp_flat_dims, CFL_SIZE);

				if( 0 != wavg_ksp_from_view_files(DIMS, ksp_flat_dims, ksp_wavg_data, ksp_flat_dims, ksp_flat_data, e2s, true, MAX_TRAINS, MAX_ECHOES, basis_dims[TE_DIM], vieworder_sort_name, vieworder_dab_name))
					error("Error executing wavg_ksp_from_view_files\n");

				debug_printf(DP_DEBUG1, "Done (%.2f seconds)\n", timestamp() - tic);
			}
			else
				ksp_wavg_data = load_cfl(ksp_wavg_name, DIMS, ksp_wavg_dims);

			scaling = jt_estimate_scaling(ksp_flat_dims, NULL, ksp_wavg_data);
			if (false == load_ksp_wavg)
				md_free(ksp_wavg_data);
			else
				unmap_cfl(DIMS, ksp_wavg_dims, ksp_wavg_data);
		}
		else 
			scaling = jt_estimate_scaling(ksp_flat_dims, NULL, ksp_flat_data);

		if (0 == mode)
			md_zsmul(DIMS, ksp_full_dims, ksp_full_data, ksp_full_data, 1. / scaling);
		else
			md_zsmul(DIMS, ksp_flat_dims, ksp_flat_data, ksp_flat_data, 1. / scaling);


		debug_printf(DP_INFO, "Readout FFT..\n");
		if (0 == mode) {
			fftscale(DIMS, ksp_full_dims, READ_FLAG, ksp_full_data, ksp_full_data);
			ifftc(DIMS, ksp_full_dims, READ_FLAG, ksp_full_data, ksp_full_data);
		}
		else {
			fftscale(DIMS, ksp_flat_dims, READ_FLAG, ksp_flat_data, ksp_flat_data);
			ifftc(DIMS, ksp_flat_dims, READ_FLAG, ksp_flat_data, ksp_flat_data);
		}
		debug_printf(DP_INFO, "Done.\n");


	// -----------------------------------------------------------
	// load initial and truth images if provided

	complex float* x_seed = NULL;

	if (cold_start)
		debug_printf(DP_INFO, "cold start\n");
	else {
		debug_printf(DP_INFO, "warm start: %s\n", x_start_fname);
		x_seed = load_cfl(x_start_fname, DIMS, x_seed_dims);

		assert(md_check_compat(DIMS, 0u, conf.use_odict ? bfimg_dims : cfimg_dims, x_seed_dims));
	}

	//complex float* x_truth = NULL;

	if (im_truth)
		debug_printf(DP_WARN, "Supplying a truth image in not supported in rjtsense.\n");

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
	// setup parslice dims and strs

	md_calc_strides(DIMS, ksp_full_strs, ksp_full_dims, CFL_SIZE);
	md_calc_strides(DIMS, ksp_flat_strs, ksp_flat_dims, CFL_SIZE);
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);
	md_calc_strides(DIMS, cfimg_strs, cfimg_dims, CFL_SIZE);
	md_calc_strides(DIMS, x_seed_strs, x_seed_dims, CFL_SIZE);
	md_calc_strides(DIMS, sens_strs, sens_dims, CFL_SIZE);
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);
	md_calc_strides(DIMS, bas_strs, basis_dims, CFL_SIZE);

	if (conf.use_odict) {
		md_calc_strides(DIMS, odict_strs, odict_dims, CFL_SIZE);
		md_calc_strides(DIMS, bfimg_strs, bfimg_dims, CFL_SIZE);
	}

	debug_printf(DP_INFO, "Reconstruction...\n");

	// dimensions and strides for one slice

	long ksp1_flat_dims[DIMS];
	long ksp1_dims[DIMS];
	long img1_dims[DIMS];
	long cfimg1_dims[DIMS];
	long bfimg1_dims[DIMS];
	long sens1_dims[DIMS];
	long x1_dims[DIMS];

	long ksp1_flat_strs[DIMS];
	long ksp1_strs[DIMS];
	long img1_strs[DIMS];
	long cfimg1_strs[DIMS];
	long bfimg1_strs[DIMS];
	long sens1_strs[DIMS];
	long x1_strs[DIMS];

	md_select_dims(DIMS, ~READ_FLAG, ksp1_flat_dims, ksp_flat_dims);
	md_calc_strides(DIMS, ksp1_flat_strs, ksp1_flat_dims, CFL_SIZE);
	
	md_select_dims(DIMS, ~READ_FLAG, ksp1_dims, ksp1_flat_dims);
#ifdef CFKSP
	ksp1_dims[COEFF_DIM] = conf.K;
#else
	ksp1_dims[TE_DIM] = pat_dims[TE_DIM];
#endif
	md_calc_strides(DIMS, ksp1_strs, ksp1_dims, CFL_SIZE);

	md_select_dims(DIMS, ~READ_FLAG, img1_dims, img_dims);
	md_calc_strides(DIMS, img1_strs, img1_dims, CFL_SIZE);
	
	md_select_dims(DIMS, ~READ_FLAG, sens1_dims, sens_dims);
	md_calc_strides(DIMS, sens1_strs, sens1_dims, CFL_SIZE);

	md_select_dims(DIMS, ~READ_FLAG, cfimg1_dims, cfimg_dims);
	md_calc_strides(DIMS, cfimg1_strs, cfimg1_dims, CFL_SIZE);
	

	if (conf.use_odict) {
		md_select_dims(DIMS, ~READ_FLAG, bfimg1_dims, bfimg_dims);
		md_calc_strides(DIMS, bfimg1_strs, bfimg1_dims, CFL_SIZE);
	}

	md_copy_dims(DIMS, x1_dims, conf.use_odict ? bfimg1_dims : cfimg1_dims);
	md_calc_strides(DIMS, x1_strs, x1_dims, CFL_SIZE);
	

	// -----------------------------------------------------------
	// call parslice recon
	
	bool ap_save = num_auto_parallelize;
	num_auto_parallelize = false;

#ifdef USE_CUDA
	int nr_cuda_devices = 0;
#endif

	if (use_gpu) {
#ifdef USE_CUDA
		nr_cuda_devices = MIN(cuda_devices(), MAX_CUDA_DEVICES);
		debug_printf(DP_INFO, "num gpus: %d\n", nr_cuda_devices);
#ifdef _OPENMP
#ifdef GPU_CPU
		//omp_set_num_threads(omp_get_max_threads()); // - 2 * nr_cuda_devices);
#else
		omp_set_num_threads(2 * nr_cuda_devices);
#endif
#else
		assert(0);
#endif
#else
		error("Recon code not compiled with CUDA\n.");
#endif
	} else {
		fft_set_num_threads(1);
	}

	int counter = 0;
#ifdef _OPENMP
#ifdef GPU_CPU
	int num_threads_loop = omp_get_max_threads();
#ifdef USE_CUDA
	num_threads_loop = MIN(omp_get_max_threads(), 3 * nr_cuda_devices);
#endif
		//#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_loop)
#pragma omp parallel for num_threads(num_threads_loop)
#else
#pragma omp parallel for
#endif
#endif
	for (int i = 0; i < ksp_flat_dims[READ_DIM]; i++) {

#ifdef SINGLE_SLICE
		if (i == 140 || i == 90) {
#endif

		complex float* image1 = NULL;
		if (save_img)
			image1 = md_alloc(DIMS, img1_dims, CFL_SIZE);

		complex float* ksp1_flat = md_alloc(DIMS, ksp1_flat_dims, CFL_SIZE);
		complex float* kspace1 = md_alloc(DIMS, ksp1_dims, CFL_SIZE);
		complex float* sens1 = md_alloc(DIMS, sens1_dims, CFL_SIZE);
		complex float* pattern1 = md_alloc(DIMS, pat_dims, CFL_SIZE);
		complex float* basis1 = md_alloc(DIMS, basis_dims, CFL_SIZE);
		complex float* cfimg1 = md_alloc(DIMS, cfimg1_dims, CFL_SIZE);
		complex float* bfimg1 = NULL;
		complex float* odict1 = NULL;

		if (conf.use_odict) {
			odict1 = md_alloc(DIMS, odict_dims, CFL_SIZE);
			bfimg1 = md_alloc(DIMS, bfimg1_dims, CFL_SIZE);
		}

		complex float* x_img1 = conf.use_odict ? bfimg1 : cfimg1;


		if (cold_start)
			md_clear(DIMS, x1_dims, x_img1, CFL_SIZE);
		else { // FIXME use a single call to md_zsmul2
			md_copy2(DIMS, x1_dims, x1_strs, x_img1, x_seed_strs, ((char*)x_seed) + i * x_seed_strs[0], CFL_SIZE);
			md_zsmul2(DIMS, x1_dims, x1_strs, x_img1, x1_strs, x_img1, 1. / scaling);
		}

		if (0 == mode)
			md_copy2(DIMS, ksp1_dims, ksp1_strs, kspace1, ksp_full_strs, ((char*)ksp_full_data) + i * ksp_full_strs[0], CFL_SIZE);
		else {
			md_copy2(DIMS, ksp1_flat_dims, ksp1_flat_strs, ksp1_flat, ksp_flat_strs, ((char*)ksp_flat_data) + i * ksp_flat_strs[0], CFL_SIZE);
		}

		md_copy2(DIMS, sens1_dims, sens1_strs, sens1, sens_strs, ((char*)sens_maps) + i * sens_strs[0], CFL_SIZE);
		md_copy2(DIMS, basis_dims, bas_strs, basis1, bas_strs, basis, CFL_SIZE);
		//md_copy2(DIMS, cfimg1_dims, cfimg1_strs, cfimg1, cfimg_strs, ((char*)cfimg) + i * cfimg_strs[0], CFL_SIZE); //should not be used

		if (0 == use_dab)
			md_copy2(DIMS, pat_dims, pat_strs, pattern1, pat_strs, pattern, CFL_SIZE);

		if (conf.use_odict) {
			md_copy2(DIMS, odict_dims, odict_strs, odict1, odict_strs, odict, CFL_SIZE);
			//md_copy2(DIMS, bfimg1_dims, bfimg1_strs, bfimg1, bfimg_strs, ((char*)bfimg) + i * bfimg_strs[0], CFL_SIZE);
		}

		if (1 == mode) {

			if (use_dab) {

#ifdef CFKSP

				if (0 != cfksp_pat_from_view_files(DIMS, ksp1_dims, kspace1, pat_dims, pattern1, ksp1_flat_dims, ksp1_flat, basis_dims, basis1, conf.K, e2s, skips_start, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name)) {
					error("Error executing cfksp_pat_from_view_files\n");
				}

#else
				if( 0 != ksp_from_view_files(DIMS, ksp1_dims, kspace1, ksp1_flat_dims, ksp1_flat, e2s, skips_start, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name))
					error("Error executing ksp_from_view_files\n");

				estimate_pattern(DIMS, ksp1_dims, COIL_DIM, pattern1, kspace1);
#endif
			}
			else {

				md_clear2(DIMS, ksp1_dims, ksp1_strs, kspace1, CFL_SIZE);
				md_zfmac2(DIMS, ksp1_dims, ksp1_strs, kspace1, pat_strs, pattern1, ksp1_flat_strs, ksp1_flat);
			}
		}

		// FFTW uses non-centered FFT, i.e. F_centered = mod F mod
		// incorporate the first mod with the sensitivities
		// incorporate the second mod with kspace
		fftmod(DIMS, sens1_dims, FFT_FLAGS, sens1, sens1);

		if (use_dab)
			fftmod(DIMS, ksp1_dims, READ_FLAG | PHS1_FLAG, kspace1, kspace1);
		else
			fftmod(DIMS, ksp1_dims, FFT_FLAGS, kspace1, kspace1);

#ifdef _OPENMP
		int omp_thread_num = omp_get_thread_num();
#else
		int omp_thread_num = 0;
#endif

		double slice_start_time = timestamp();
#ifdef USE_CUDA
		int gpun = 0;
#endif

		if (use_gpu) {
#ifdef USE_CUDA
			if (nr_cuda_devices == 0)
				error("No CUDA Devices\n");

			debug_printf(DP_DEBUG3, "thread num = %d\n", omp_thread_num);

#ifdef GPU_CPU
			if (omp_thread_num < nr_cuda_devices) {
				gpun = omp_thread_num;
				cuda_init(gpun);
#ifdef _OPENMP
				omp_set_lock(&gpulock[gpun]);
#endif

				debug_printf(DP_DEBUG1, "GPU recon\n");
				jtsense_recon_gpu(&conf, italgo, iconf, image1, cfimg1, bfimg1, kspace1, crop_dims, sens1_dims, sens1, pat_dims, pattern1, basis_dims, basis1, odict_dims, odict1, NULL);
#ifdef _OPENMP
				omp_unset_lock(&gpulock[gpun]);
#endif
			}
			else {
				gpun = -1;
				debug_printf(DP_DEBUG1, "CPU recon\n");
				jtsense_recon(&conf, italgo, iconf, image1, cfimg1, bfimg1, kspace1, crop_dims, sens1_dims, sens1, pat_dims, pattern1, basis_dims, basis1, odict_dims, odict1, NULL);
			}
#else
				gpun = omp_thread_num % nr_cuda_devices;
				cuda_init(gpun);
#ifdef _OPENMP
				omp_set_lock(&gpulock[gpun]);
#endif

				debug_printf(DP_DEBUG3, "GPU recon\n");
				jtsense_recon_gpu(&conf, italgo, iconf, image1, cfimg1, bfimg1, kspace1, crop_dims, sens1_dims, sens1, pat_dims, pattern1, basis_dims, basis1, odict_dims, odict1, NULL);
#ifdef _OPENMP
				omp_unset_lock(&gpulock[gpun]);
#endif
#endif
#else
			error("Recon code not compiled with CUDA.\n");
#endif
		} else {
			debug_printf(DP_DEBUG4, "CPU recon\n");
			jtsense_recon(&conf, italgo, iconf, image1, cfimg1, bfimg1, kspace1, crop_dims, sens1_dims, sens1, pat_dims, pattern1, basis_dims, basis1, odict_dims, odict1, NULL);
		}

		double slice_time = timestamp() - slice_start_time;

#ifdef USE_CUDA
#ifdef GPU_CPU
		if (use_gpu && omp_thread_num < nr_cuda_devices)
#else
		if (use_gpu)
#endif
			debug_printf(DP_DEBUG2, "done with slice %d on GPU %d after %f seconds (thread %d)\n", i, gpun, slice_time, omp_thread_num);
		else
#endif
			debug_printf(DP_DEBUG2, "done with slice %d after %f seconds (thread %d)\n", i, slice_time, omp_thread_num);

		double save_start_time = timestamp();
		if (save_img) {
			//md_copy2(DIMS, img1_dims, img_strs, ((char*)image) + i * img_strs[0], img1_strs, image1, CFL_SIZE);
			md_zsmul2(DIMS, img1_dims, img_strs, (complex float*)(((char*)image) + i * img_strs[0]), img1_strs, image1, scaling);
		}

		if (save_cfimg)
			md_zsmul2(DIMS, cfimg1_dims, cfimg_strs, (complex float*)(((char*)cfimg) + i * cfimg_strs[0]), cfimg1_strs, cfimg1, scaling);
				//md_copy2(DIMS, cfimg1_dims, cfimg_strs, ((char*)cfimg) + i * cfimg_strs[0], cfimg1_strs, cfimg1, CFL_SIZE);

		if (conf.use_odict && save_bfimg)
			md_zsmul2(DIMS, bfimg1_dims, bfimg_strs, (complex float*)(((char*)bfimg) + i * bfimg_strs[0]), bfimg1_strs, bfimg1, scaling);
				//md_copy2(DIMS, bfimg1_dims, bfimg_strs, ((char*)bfimg) + i * bfimg_strs[0], bfimg1_strs, bfimg1, CFL_SIZE);

		double save_time = timestamp() - save_start_time;
		debug_printf(DP_DEBUG2, "done copying slice %d to volume after %f seconds (threadh %d)\n", i, save_time, omp_thread_num);

		debug_printf(DP_DEBUG3, "freeing temporary memory %d\n", i);

		md_free(image1);
		md_free(ksp1_flat);
		md_free(kspace1);
		md_free(sens1);
		md_free(pattern1);
		md_free(basis1);
		md_free(odict1);
		md_free(cfimg1);
		md_free(bfimg1);

#ifdef SINGLE_SLICE
		}
#endif


#pragma omp critical
		{ debug_printf(DP_DEBUG2, "%04d/%04ld    \n", ++counter, ksp_flat_dims[0]); }
	}
	debug_printf(DP_DEBUG2, "\n");


	// -----------------------------------------------------------
	// cool down

	num_auto_parallelize = ap_save;

	//debug_printf(DP_INFO, "Rescaling: %f\n", scaling);
	//md_zsmul(DIMS, img_dims, image, image, scaling);

	unmap_cfl(DIMS, sens_dims, sens_maps);

	if (save_img)
		unmap_cfl(DIMS, img_dims, image);

	if (! cold_start)
		unmap_cfl(DIMS, x_seed_dims, x_seed);

	if (save_cfimg) {
		//md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, scaling);
		unmap_cfl(DIMS, cfimg_dims, cfimg);
	}
	else
		md_free(cfimg);

	if (conf.use_odict && save_bfimg) {
		//md_zsmul(DIMS, bfimg_dims, bfimg, bfimg, scaling);
		unmap_cfl(DIMS, bfimg_dims, bfimg);
	}
	else if (conf.use_odict)
		md_free(bfimg);

	if (0 == mode) {
		md_free(ksp_flat_data);
		md_free(pattern);
		unmap_cfl(DIMS, ksp_full_dims, ksp_full_data);
	}
	else {
		unmap_cfl(DIMS, ksp_flat_dims, ksp_flat_data);
		if (0 == use_dab)
			unmap_cfl(DIMS, pat_dims, pattern);
	}

	unmap_cfl(DIMS, basis_dims, basis);

	if (NULL != conf.l1wav_lambdas)
		free(conf.l1wav_lambdas);

	free(iconf);

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);

}


