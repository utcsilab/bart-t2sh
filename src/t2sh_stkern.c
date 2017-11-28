/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "jtsense/jtmodel.h"


#ifndef DIMS
#define DIMS 16
#endif



static const char* usage_str = "<pattern> <basis> <stkern_out>";
static const char* help_str = "Create space-time t2shuffling kernel";



int main_t2sh_stkern(int argc, char* argv[])
{
	num_init();

	double start_time = timestamp();

// FIXME varTR support
#if 0
	bool varTR = false;
	const char* TR_vals_file = NULL;
	unsigned int R = 1;
#endif

	unsigned int K = 4;

	const struct opt_s opts[] = {
#if 0
		OPT_STRING('r', &TR_vals_file, "<file>", "Variable TRs <file>"),
		OPT_UINT('R', &R, "R", "Number of unique TR values [Default R=1]"),
#endif
		OPT_UINT('K', &K, "K", "Subspace size [Default K=4]"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


#if 0
	if (NULL != TR_vals_file)
		varTR = true;
#endif


	// -----------------------------------------------------------
	// load data

	long pat_dims[DIMS];
	long bas_dims[DIMS];
	long bas_full_dims[DIMS];
	long stk_dims[DIMS];

	complex float* pat = load_cfl(argv[1], DIMS, pat_dims);
	complex float* bas_full = load_cfl(argv[2], DIMS, bas_full_dims);

	// load subspace basis
	long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

	md_select_dims(DIMS, ~COEFF_FLAG, bas_dims, bas_full_dims);
	bas_dims[COEFF_DIM] = K;
	complex float* bas = anon_cfl(NULL, DIMS, bas_dims);

	md_copy_block(DIMS, pos, bas_dims, bas, bas_full_dims, bas_full, CFL_SIZE);
	unmap_cfl(DIMS, bas_full_dims, bas_full);

#if 0
	if (varTR)
		ksp_dims[TIME_DIM] = R;
#endif


	// -----------------------------------------------------------
	// create stkern
	
	md_select_dims(DIMS, (PHS1_FLAG | PHS2_FLAG | COEFF_FLAG | CSHIFT_FLAG | TIME_FLAG), stk_dims, pat_dims);
	stk_dims[COEFF_DIM] = bas_dims[COEFF_DIM];
	stk_dims[TE_DIM] = stk_dims[COEFF_DIM];

	debug_printf(DP_DEBUG3, "stkern_dims =\t");
	debug_print_dims(DP_DEBUG3, DIMS, stk_dims);

	complex float* stk = create_cfl(argv[3], DIMS, stk_dims);

	create_stkern_mat(stk, pat_dims, pat, bas_dims, bas);


	// -----------------------------------------------------------
	// cool down

	unmap_cfl(DIMS, pat_dims, pat);
	unmap_cfl(DIMS, bas_dims, bas);
	unmap_cfl(DIMS, stk_dims, stk);

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


