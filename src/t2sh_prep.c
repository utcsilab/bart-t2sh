/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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
#include "num/fft.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "jtsense/jtrecon.h"


#ifndef DIMS
#define DIMS 16
#endif



static const char* usage_str = "<echoes2skip> <num_echoes> <vieworder_sort> <vieworder_sort_dab> <data_in> <kspace_out> [<pat_out>]";
static const char* help_str = "Reorder data according to vieworder files and output corresponding kspace.\n";



int main_t2sh_prep(int argc, char* argv[])
{
	double start_time = timestamp();

	bool wavg = false;
	bool proj = false;
	bool pat = false;

	const char* basis_file = NULL;
	const char* vieworder_sort_file = NULL;
	const char* vieworder_dab_file = NULL;
	const char* pat_file = NULL;

	unsigned int echoes2skip = 0;
	unsigned int num_echoes = 0;
	unsigned int K = 4;


	const struct opt_s opts[] = {

		OPT_SET('w', &wavg, "Weighted average along time"),
		OPT_STRING('b', &basis_file, "<file>", "Project onto basis in <file>"),
		OPT_UINT('K', &K, "K", "Subspace size for -b (Default K=4)"),
	};

	cmdline(&argc, argv, 6, 7, usage_str, help_str, ARRAY_SIZE(opts), opts);


	if (NULL != basis_file)
		proj = true;

	if (proj && wavg)
		error("Cannot project and average\n");

	if (8 == argc) {

		pat = true;
		pat_file = argv[7];
		assert(proj);
	}


	// -----------------------------------------------------------
	// load data

	long dat_dims[DIMS];
	long ksp_dims[DIMS];
	long bas_dims[DIMS];
	long pat_dims[DIMS];

	echoes2skip = atoi(argv[1]);
	num_echoes = atoi(argv[2]);

	vieworder_sort_file = argv[3];
	vieworder_dab_file = argv[4];

	complex float* dat = load_cfl(argv[5], DIMS, dat_dims);

	if (!wavg && dat_dims[READ_DIM] > 1)
		debug_printf(DP_WARN, "Warning: 3D volume expanding into time!\n");

	md_copy_dims(DIMS, ksp_dims, dat_dims);

	complex float* basis = NULL;

	if (proj) {

		ksp_dims[COEFF_DIM] = K;
		basis = load_cfl(basis_file, DIMS, bas_dims);
	}
	else if (!wavg)
		ksp_dims[TE_DIM] = num_echoes;

	md_select_dims(DIMS, ~(COIL_FLAG), pat_dims, dat_dims);
	pat_dims[TE_DIM] = num_echoes;

	complex float* ksp = create_cfl(argv[6], DIMS, ksp_dims);
	complex float* pattern = NULL;


	if (proj) {

		// expand kspace into subspace, reorder data
		pattern = (pat ? create_cfl : anon_cfl)(pat_file, DIMS, pat_dims);
		if( 0 != cfksp_pat_from_view_files(DIMS, ksp_dims, ksp, pat_dims, pattern, dat_dims, dat, bas_dims, basis, K, echoes2skip, 0, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_file, vieworder_dab_file))
			error("Error executing cfksp_pat_from_view_files\n");
	}
	else if (wavg) {

		// expand kspace into time bins, reorder data, and average
		if( 0 != wavg_ksp_from_view_files(DIMS, ksp_dims, ksp, dat_dims, dat, echoes2skip, true, MAX_TRAINS, MAX_ECHOES, num_echoes, vieworder_sort_file, vieworder_dab_file))
			error("Error executing wavg_ksp_from_view_files\n");
	}
	else {

		// expand kspace into time bins and reorder data
		if( 0 != ksp_from_view_files(DIMS, ksp_dims, ksp, dat_dims, dat, echoes2skip, 0, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_file, vieworder_dab_file))
			error("Error executing ksp_from_view_files\n");
	}

	fftmod(DIMS, ksp_dims, PHS2_FLAG, ksp, ksp);


	// -----------------------------------------------------------
	// cool down

	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, dat_dims, dat);

	if (proj) {

		free((void*)basis_file);
		unmap_cfl(DIMS, pat_dims, pattern);
		unmap_cfl(DIMS, bas_dims, basis);
	}

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


