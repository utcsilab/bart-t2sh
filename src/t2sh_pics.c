/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#ifdef USE_MKL
#include <mkl.h>
#endif

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/init.h"

#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/realval.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "lowrank/lrthresh.h"

#include "misc/types.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "grecon/optreg.h"

#include "jtsense/jtrecon.h"
#include "jtsense/jtmodel.h"



static const char usage_str[] = "<cfksp> <te_pattern> <sens> <basis> <output>";

static const char help_str[] = "T2 Shuffling reconstruction.\n"
"\t<cfksp> is k-space in subspace domain\n"
"\t<te_pattern> is samplling pattern over time\n"
"\t<sens> is coil sensitivities\n"
"\t<basis> is subspace basis\n"
"\t<output> is reconstructed image in subspace domain";



int main_t2sh_pics(int argc, char* argv[])
{
	double start_time = timestamp();


	// -----------------------------------------------------------
	// set up conf and option parser

	struct jtsense_conf conf = jtsense_defaults;
	struct sense_conf sconf = sense_defaults;
	conf.sconf = sconf;

	bool use_gpu = false;
	int gpun = -1;

	bool rvc = false;
	float scaling = 0.;

	unsigned int llr_blk = 8;

	const char* cfimg_truth_file = NULL;
	bool use_cfimg_truth = false;

	const char* cfimg_start_file = NULL;
	bool warm_start = false;

	const char* stkern_file = NULL;
	bool use_stkern_file = false;

	unsigned int maxiter = 50;
	bool randshift = true;
	bool hogwild = false;
	float ist_step = iter_fista_defaults.step;
	float ist_continuation = iter_fista_defaults.continuation;
	float admm_rho = iter_admm_defaults.rho;
	unsigned int admm_maxitercg = iter_admm_defaults.maxitercg;


	struct opt_reg_s ropts;
	assert(0 == opt_reg_init(&ropts));


	const struct opt_s opts[] = {

		{ 'R', true, opt_reg, &ropts, " <T>:A:B:C\tgeneralized regularization options [-Rh for help]" },
		OPT_SET('c', &rvc, "real-value constraint"),
		OPT_FLOAT('s', &ist_step, "step", "IST/FISTA iteration stepsize"),
		OPT_FLOAT('r', &ist_continuation, "val", "IST/FISTA continuation fraction"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_CLEAR('n', &randshift, "disable random shifting"),
		OPT_INT('g', &gpun, "gpun", "Use GPU device gpun"),
		OPT_SELECT('I', enum algo_t, &ropts.algo, IST, "\tselect IST"),
		OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('H', &hogwild, "hogwild"),
		OPT_SET('F', &conf.fast, "fast"),
		OPT_STRING('T', &cfimg_truth_file, "file", "truth file"),
		OPT_STRING('W', &cfimg_start_file, "<img>", "Warm start with <img>"),
		OPT_STRING('S', &stkern_file, "<stkern>", "Use precomputed stkern mat <stkern>"),
		OPT_FLOAT('u', &admm_rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm_maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_SELECT('m', enum algo_t, &ropts.algo, ADMM, "\tSelect ADMM"),
		OPT_FLOAT('w', &scaling, "val", "scaling"),
		OPT_UINT('K', &conf.K, "K", "Number of temporal coefficients"),
	};

	cmdline(&argc, argv, 5, 5, usage_str, help_str, ARRAY_SIZE(opts), opts);


	// -----------------------------------------------------------
	// check parameters

	if (ist_continuation < 0. || ist_continuation > 1.)
		error("Continuation scalar should be between 0. and 1.\n");

	if (ist_step <= 0.)
		error("Step size should be greater than 0.\n");

	if (NULL != cfimg_truth_file)
		use_cfimg_truth = true;

	if (NULL != cfimg_start_file)
		warm_start = true;

	if (NULL != stkern_file)
		use_stkern_file = true;

	if (-1 != gpun)
		use_gpu = true;


	// -----------------------------------------------------------
	// load data and get dimensions

	long max_dims[DIMS];
	long cfksp_dims[DIMS];
	long pat_dims[DIMS];
	long sens_dims[DIMS];
	long basis_dims[DIMS];
	long cfimg_dims[DIMS];

	complex float* cfksp = load_cfl(argv[1], DIMS, cfksp_dims);
	complex float* pattern = load_cfl(argv[2], DIMS, pat_dims);
	complex float* sens = load_cfl(argv[3], DIMS, sens_dims);
	complex float* basis = load_cfl(argv[4], DIMS, basis_dims);
	complex float* cfimg = NULL;

	// FIXME: check dimensions of cfksp to match conf.K

	// keep first K basis elements
	complex float* bas = NULL;
	if (basis_dims[COEFF_DIM] > conf.K) {

		long subs_dims[DIMS]; // basis truncated to K elements
		md_select_dims(DIMS, TE_FLAG, subs_dims, basis_dims);
		subs_dims[COEFF_DIM] = conf.K;

		long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

		bas = anon_cfl(NULL, DIMS, subs_dims);
		md_copy_block(DIMS, pos, subs_dims, bas, basis_dims, basis, CFL_SIZE);
		unmap_cfl(DIMS, basis_dims, basis);
		basis = bas;
		md_copy_dims(DIMS, basis_dims, subs_dims);
	}
	else if (basis_dims[COEFF_DIM] < conf.K) {

		debug_printf(DP_WARN, "Subspace size is larger than basis. Using subspace size %d\n", basis_dims[COEFF_DIM]);
		conf.K = basis_dims[COEFF_DIM];
	}

	// keep first K ksp coefficients
	complex float* cfkspK = NULL;
	if (cfksp_dims[COEFF_DIM] > conf.K) {

		long cfksp_trunc_dims[DIMS]; // cfksp truncated to K elements
		md_select_dims(DIMS, ~COEFF_FLAG, cfksp_trunc_dims, cfksp_dims);
		cfksp_trunc_dims[COEFF_DIM] = conf.K;

		long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

		cfkspK = anon_cfl(NULL, DIMS, cfksp_trunc_dims);
		md_copy_block(DIMS, pos, cfksp_trunc_dims, cfkspK, cfksp_dims, cfksp, CFL_SIZE);
		unmap_cfl(DIMS, cfksp_dims, cfksp);
		cfksp = cfkspK;
		md_copy_dims(DIMS, cfksp_dims, cfksp_trunc_dims);
	}
	else if (cfksp_dims[COEFF_DIM] < conf.K) {

		debug_printf(DP_WARN, "Subspace size is larger than cfksp. Using subspace size %d\n", cfksp_dims[COEFF_DIM]);
		conf.K = cfksp_dims[COEFF_DIM];
	}


	md_copy_dims(DIMS, max_dims, cfksp_dims);
	md_copy_dims(5, max_dims, sens_dims);
	max_dims[TE_DIM] = basis_dims[TE_DIM];

	md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG), cfimg_dims, max_dims);
	if (1 != cfksp_dims[MAPS_DIM])
		error("kspace should not have multiple sets of maps!\n");


#ifdef USE_CUDA
	if (use_gpu)
		num_init_gpu_device(gpun);
	else
#endif
		num_init();

	fftmod(DIMS, cfksp_dims, FFT_FLAGS, cfksp, cfksp);
	fftmod(DIMS, sens_dims, FFT_FLAGS, sens, sens);

	cfimg = create_cfl(argv[5], DIMS, cfimg_dims);
	md_clear(DIMS, cfimg_dims, cfimg, CFL_SIZE);


	if (scaling == 0.)
		scaling = jt_estimate_scaling(cfksp_dims, COEFF_FLAG, NULL, cfksp);
	else {

		if (scaling <= 0.)
			error("Scaling should be greater than 0.\n");

		debug_printf(DP_DEBUG1, "Scaling: %f\n", scaling);
	}

	if (scaling != 0.)
		md_zsmul(DIMS, cfksp_dims, cfksp, cfksp, 1. / scaling);




	long cfimg_truth_dims[DIMS];
	complex float* cfimg_truth = NULL;

	if (use_cfimg_truth) {

		cfimg_truth = load_cfl(cfimg_truth_file, DIMS, cfimg_truth_dims);

		if (!md_check_compat(DIMS, 0u, cfimg_dims, cfimg_truth_dims))
			error("Truth image dimensions not compatible with output!\n");

		if (scaling != 0.)
			md_zsmul(DIMS, cfimg_dims, cfimg_truth, cfimg_truth, 1. / scaling);
	}

	long cfimg_start_dims[DIMS];
	complex float* cfimg_start = NULL;

	if (warm_start) { 

		debug_printf(DP_DEBUG1, "Warm start: %s\n", cfimg_start_file);
		cfimg_start = load_cfl(cfimg_start_file, DIMS, cfimg_start_dims);

		if (!md_check_compat(DIMS, 0u, cfimg_dims, cfimg_start_dims))
			error("Initial (warm start) image dimensions not compatible with output!\n");

		md_copy(DIMS, cfimg_dims, cfimg, cfimg_start, CFL_SIZE);

		free((void*)cfimg_start_file);
		unmap_cfl(DIMS, cfimg_dims, cfimg_start);

		if (scaling != 0.)
			md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, 1. /  scaling);
	}


	long stkern_dims[DIMS];
	complex float* stkern_mat = NULL;

	if (use_stkern_file)
		stkern_mat = load_cfl(stkern_file, DIMS, stkern_dims);


	// -----------------------------------------------------------
	// print options and statistics

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction (device %d)\n", gpun);

	if (sens_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", sens_dims[MAPS_DIM]);

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (use_cfimg_truth)
		debug_printf(DP_INFO, "Compare to truth\n");


	long T = md_calc_size(DIMS, pat_dims);
	long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

	debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);

	debug_printf(DP_DEBUG2, "cfksp_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, cfksp_dims);

	debug_printf(DP_DEBUG2, "pat_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, pat_dims);

	debug_printf(DP_DEBUG2, "sens_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, sens_dims);

	debug_printf(DP_DEBUG2, "basis_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, basis_dims);

	debug_printf(DP_DEBUG2, "cfimg_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, cfimg_dims);


	// -----------------------------------------------------------
	// initialize forward op 

	long max_dims_sens[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, max_dims_sens, max_dims);

	debug_printf(DP_DEBUG2, "max_dims =\t");
	debug_print_dims(DP_DEBUG2, DIMS, max_dims);
	debug_printf(DP_DEBUG2, "max_dims_sens =\t");
	debug_print_dims(DP_DEBUG2, DIMS, max_dims_sens);


	long sens_flags = (FFT_FLAGS | SENS_FLAGS);
	if (max_dims_sens[CSHIFT_DIM] > 1)
		sens_flags = MD_SET(sens_flags, CSHIFT_DIM);

	double rescale_start = timestamp();
#ifndef USE_INTEL_KERNELS
	const struct linop_s* sense_op = sense_init(max_dims_sens, sens_flags, sens);
#else
	const struct linop_s* sense_op = NULL;
#endif
	double rescale_end = timestamp();
	debug_printf(DP_DEBUG3, "rescale Time: %f\n", rescale_end - rescale_start);

	if (rvc) {
		debug_printf(DP_INFO, "RVC\n");

		struct linop_s* rvc_op = linop_realval_create(DIMS, cfimg_dims);
		struct linop_s* tmp_op = linop_chain(rvc_op, sense_op);

		linop_forward(rvc_op, DIMS, cfimg_dims, cfimg, DIMS, cfimg_dims, cfimg);

		linop_free(rvc_op);
		linop_free(sense_op);
		sense_op = tmp_op;
	}

#ifdef USE_INTEL_KERNELS
	DFTI_DESCRIPTOR_HANDLE plan1d_0 = NULL;
	DFTI_DESCRIPTOR_HANDLE plan1d_1 = NULL;
	const struct linop_s* forward_op = jtmodel_intel_init(max_dims, cfimg_dims, sense_op, sens_dims, sens, pat_dims, pattern, basis_dims, basis, use_stkern_file ? stkern_mat : NULL, use_gpu, plan1d_0, plan1d_1);
#else
	const struct linop_s* forward_op = jtmodel_init(max_dims, sense_op, pat_dims, pattern, basis_dims, basis, use_stkern_file ? stkern_mat : NULL, use_gpu);
#endif

	if (use_stkern_file) {

		free((void*)stkern_file);
		unmap_cfl(DIMS, stkern_dims, stkern_mat);
	}



	// -----------------------------------------------------------
	// initialize prox functions and transform operators

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	opt_reg_configure(DIMS, cfimg_dims, &ropts, thresh_ops, trafos, llr_blk, randshift, use_gpu);

	int nr_penalties = ropts.r;
	struct reg_s* regs = ropts.regs;
	enum algo_t algo = ropts.algo;


	// -----------------------------------------------------------
	// initialize algorithm

	italgo_fun2_t italgo = iter2_call_iter;
	struct iter_call_s iter2_data;
	SET_TYPEID(iter_call_s, &iter2_data);

	iter_conf* iconf = CAST_UP(&iter2_data);

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;
	struct iter_ist_conf isconf;
	struct iter_admm_conf mmconf;

	if ((CG == algo) && (1 == nr_penalties) && (L2IMG != regs[0].xform))
		algo = FISTA;

	if (nr_penalties > 1)
		algo = ADMM;


	switch (algo) {

		case CG:

			debug_printf(DP_INFO, "conjugate gradients\n");

			assert((0 == nr_penalties) || ((1 == nr_penalties) && (L2IMG == regs[0].xform)));

			cgconf = iter_conjgrad_defaults;
			cgconf.maxiter = maxiter;
			cgconf.l2lambda = (0 == nr_penalties) ? 0. : regs[0].lambda;

			iter2_data.fun = iter_conjgrad;
			iter2_data._conf = CAST_UP(&cgconf);

			nr_penalties = 0;

			break;

		case IST:

			debug_printf(DP_INFO, "IST\n");

			assert(1 == nr_penalties);

			isconf = iter_ist_defaults;
			isconf.maxiter = maxiter;
			isconf.step = ist_step;
			isconf.hogwild = hogwild;
			isconf.continuation = ist_continuation;

			iter2_data.fun = iter_ist;
			iter2_data._conf = CAST_UP(&isconf);

			break;

		case ADMM:

			debug_printf(DP_INFO, "ADMM\n");

			mmconf = iter_admm_defaults;
			mmconf.maxiter = maxiter;
			mmconf.maxitercg = admm_maxitercg;
			mmconf.cg_eps = 1.E-5;
			mmconf.mu = 2;
			mmconf.rho = admm_rho;
			mmconf.hogwild = hogwild;
			mmconf.dynamic_rho = !hogwild;
			mmconf.fast = conf.fast;
			mmconf.do_warmstart = warm_start;
			mmconf.ABSTOL = 0.;
			mmconf.RELTOL = 0.;

			italgo = iter2_admm;
			iconf = CAST_UP(&mmconf);

			break;

		case FISTA:

			debug_printf(DP_INFO, "FISTA\n");

			assert(1 == nr_penalties);

			fsconf = iter_fista_defaults;
			fsconf.maxiter = maxiter;
			fsconf.step = ist_step;
			fsconf.hogwild = hogwild;
			fsconf.continuation = ist_continuation;

			iter2_data.fun = iter_fista;
			iter2_data._conf = CAST_UP(&fsconf);

			break;

		default:

			assert(0);
	}


	const struct operator_s* t2sh_pics_op = operator_t2sh_pics_create(&conf, italgo, iconf, forward_op, nr_penalties, thresh_ops,
				(ADMM == algo) ? trafos : NULL, NULL, cfimg_truth, use_gpu);

	double op_start = timestamp();
	operator_apply(t2sh_pics_op, DIMS, cfimg_dims, cfimg, DIMS, cfksp_dims, cfksp);
	double op_end = timestamp();
	debug_printf(DP_DEBUG3, "Solver Time: %f\n", op_end - op_start);

	operator_free(t2sh_pics_op);

	debug_printf(DP_INFO, "Rescaling: %f\n", scaling);
	md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, scaling);


	// -----------------------------------------------------------
	// clean up


	linop_free(forward_op);
#ifndef USE_INTEL_KERNELS
	linop_free(sense_op);
#endif

	unmap_cfl(DIMS, cfksp_dims, cfksp);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, basis_dims, basis);
	unmap_cfl(DIMS, cfimg_dims, cfimg);

	if (use_cfimg_truth) {

		free((void*)cfimg_truth_file);
		unmap_cfl(DIMS, cfimg_dims, cfimg_truth);
	}


	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	exit(0);
}


