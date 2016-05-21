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

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/rvc.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "lowrank/lrthresh.h"

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
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('F', &conf.fast, "fast"),
		OPT_STRING('T', &cfimg_truth_file, "file", "truth file"),
		OPT_STRING('W', &cfimg_start_file, "<img>", "Warm start with <img>"),
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

	if (-1 != gpun)
		use_gpu = true;


	// -----------------------------------------------------------
	// load data and get dimensions
	
	long max_dims[DIMS];
	long cfksp_dims[DIMS];
	long pat_dims[DIMS];
	long sens_dims[DIMS];
	long basis_dims[DIMS];
	long subs_dims[DIMS]; // basis truncated to K elements
	long cfimg_dims[DIMS];

	complex float* cfksp = load_cfl(argv[1], DIMS, cfksp_dims);
	complex float* pattern = load_cfl(argv[2], DIMS, pat_dims);
	complex float* sens = load_cfl(argv[3], DIMS, sens_dims);
	complex float* basis = load_cfl(argv[4], DIMS, basis_dims);
	complex float* cfimg = NULL;

	// FIXME: check dimensions of cfksp to match conf.K
	if (basis_dims[COEFF_DIM] != conf.K) {

		md_select_dims(DIMS, TE_FLAG, subs_dims, basis_dims);
		subs_dims[COEFF_DIM] = conf.K;

		long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

		complex float* bas = anon_cfl(NULL, DIMS, subs_dims);
		md_copy_block(DIMS, pos, subs_dims, bas, basis_dims, basis, CFL_SIZE);
		unmap_cfl(DIMS, basis_dims, basis);
		basis = bas;
		md_copy_dims(DIMS, basis_dims, subs_dims);
	}

	md_copy_dims(DIMS, max_dims, basis_dims);

	md_copy_dims(5, max_dims, sens_dims);

	md_select_dims(DIMS, ~(COIL_FLAG | TE_FLAG), cfimg_dims, max_dims);

	if (!md_check_compat(DIMS, ~(FFT_FLAGS | MAPS_FLAG), cfimg_dims, sens_dims))
		error("Dimensions of coefficent image and sensitivities do not match!\n");

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
		scaling = jt_estimate_scaling(cfksp_dims, NULL, cfksp);
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
		
		if (md_check_compat(DIMS, 0u, cfimg_dims, cfimg_truth_dims))
			error("Truth image dimensions not compatible with output!\n");

		if (scaling != 0.)
			md_zsmul(DIMS, cfimg_dims, cfimg_truth, cfimg_truth, 1. / scaling);
	}

	long cfimg_start_dims[DIMS];
	complex float* cfimg_start = NULL;

	if (warm_start) { 

		debug_printf(DP_DEBUG1, "Warm start: %s\n", cfimg_start_file);
		cfimg_start = load_cfl(cfimg_start_file, DIMS, cfimg_start_dims);

		if (md_check_compat(DIMS, 0u, cfimg_dims, cfimg_start_dims))
			error("Initial (warm start) image dimensions not compatible with output!\n");

		md_copy(DIMS, cfimg_dims, cfimg, cfimg_start, CFL_SIZE);

		free((void*)cfimg_start_file);
		unmap_cfl(DIMS, cfimg_dims, cfimg_start);

		if (scaling != 0.)
			md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, 1. /  scaling);
	}


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


	const struct linop_s* sense_op = sense_init(max_dims_sens, (FFT_FLAGS | SENS_FLAGS), sens);


	if (rvc) {

		struct linop_s* rvc_op = rvc_create(DIMS, cfimg_dims);
		struct linop_s* tmp_op = linop_chain(sense_op, rvc_op);

		linop_forward(rvc_op, DIMS, cfimg_dims, cfimg, DIMS, cfimg_dims, cfimg);

		linop_free(rvc_op);
		linop_free(sense_op);
		sense_op = tmp_op;
	}

	const struct linop_s* forward_op = jtmodel_init(max_dims, sense_op, pat_dims, pattern, basis_dims, basis);


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

	void* iconf = &iter2_data;

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
			iter2_data._conf = &cgconf.base;

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
			iter2_data._conf = &isconf.base;

			break;

		case ADMM:

			debug_printf(DP_INFO, "ADMM\n");

			mmconf = iter_admm_defaults;
			mmconf.maxiter = maxiter;
			mmconf.maxitercg = admm_maxitercg;
			mmconf.rho = admm_rho;
			mmconf.hogwild = hogwild;
			mmconf.fast = conf.fast;
			mmconf.do_warmstart = warm_start;
			//		mmconf.dynamic_rho = true;
			mmconf.ABSTOL = 0.;
			mmconf.RELTOL = 0.;

			italgo = iter2_admm;
			iconf = &mmconf;

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
			iter2_data._conf = &fsconf.base;

			break;

		default:

			assert(0);
	}


	if (use_gpu) 
#ifdef USE_CUDA
		jtsense_recon(&conf, cfimg, italgo, iconf, forward_op, nr_penalties, thresh_ops,
				(ADMM == algo) ? trafos : NULL, NULL, cfksp, cfimg_truth);
#else
	error("BART not compiled for GPU!\n");
#endif
	else
		jtsense_recon(&conf, cfimg, italgo, iconf, forward_op, nr_penalties, thresh_ops,
				(ADMM == algo) ? trafos : NULL, NULL, cfksp, cfimg_truth);

	debug_printf(DP_INFO, "Rescaling: %f\n", scaling);
	md_zsmul(DIMS, cfimg_dims, cfimg, cfimg, scaling);


	// -----------------------------------------------------------
	// clean up


	linop_free(forward_op);
	linop_free(sense_op);

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

