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



static const char* usage_str = "<echoes2skip> <num_echoes> <vieworder_sort> <vieworder_sort_dab> <data_in> <kspace_out>";
static const char* help_str = "Reorder data according to vieworder files and output corresponding kspace.";



int main_t2sh_prep(int argc, char* argv[])
{
	num_init();

	double start_time = timestamp();

	bool wavg = false;
	bool avg = false;
	bool proj = false;

	const char* basis_file = NULL;
	const char* vieworder_sort_file = NULL;
	const char* vieworder_dab_file = NULL;

	unsigned int echoes2skip = 0;
	unsigned int num_echoes = 0;
	unsigned int K = 4;


	const struct opt_s opts[] = {

		OPT_SET('w', &wavg, "Weighted average along time"),
		OPT_SET('a', &avg, "average along time"),
		OPT_STRING('b', &basis_file, "<file>", "Project onto basis in <file>"),
		OPT_UINT('K', &K, "K", "Subspace size for -b (Default K=4)"),
	};

	cmdline(&argc, argv, 6, 6, usage_str, help_str, ARRAY_SIZE(opts), opts);


	if (NULL != basis_file)
		proj = true;

	if (proj && (wavg || avg))
		error("Cannot project and average\n");


	// -----------------------------------------------------------
	// load data

	long dat_dims[DIMS];
	long ksp_dims[DIMS];
	long bas_dims[DIMS];

	echoes2skip = atoi(argv[1]);
	num_echoes = atoi(argv[2]);

	vieworder_sort_file = argv[3];
	vieworder_dab_file = argv[4];

	complex float* dat = load_cfl(argv[5], DIMS, dat_dims);

	if (!(wavg || avg) && !proj && dat_dims[READ_DIM] > 1)
		debug_printf(DP_WARN, "Warning: 3D volume expanding into time!\n");

	md_copy_dims(DIMS, ksp_dims, dat_dims);

	complex float* basis = NULL;

	if (proj) {

		// set kspace subspace dimensions
		ksp_dims[COEFF_DIM] = K;

		// load subspace basis
		long bas_full_dims[DIMS];
		long pos[DIMS] = MD_INIT_ARRAY(DIMS, 0);

		complex float* tmp = load_cfl(basis_file, DIMS, bas_full_dims);

		md_select_dims(DIMS, ~COEFF_FLAG, bas_dims, bas_full_dims);
		bas_dims[COEFF_DIM] = K;
		basis = md_alloc(DIMS, bas_dims, CFL_SIZE);

		md_copy_block(DIMS, pos, bas_dims, basis, bas_full_dims, tmp, CFL_SIZE);
		unmap_cfl(DIMS, bas_full_dims, tmp);
	}
	else if (!(wavg || avg))
		ksp_dims[TE_DIM] = num_echoes;

	complex float* ksp = create_cfl(argv[6], DIMS, ksp_dims);


	if (proj) {

		// expand kspace into subspace, reorder data
		debug_printf(DP_INFO, "calling cfksp \n");
		if( 0 != cfksp_from_view_files(DIMS, ksp_dims, ksp, dat_dims, dat, bas_dims, basis, echoes2skip, 0, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_file, vieworder_dab_file))
			error("Error executing cfksp_from_view_files\n");
		debug_printf(DP_INFO, "done\n");

	}
	else if (wavg || avg) {

		// expand kspace into time bins, reorder data, and average
		if( 0 != avg_ksp_from_view_files(DIMS, wavg, ksp_dims, ksp, dat_dims, dat, echoes2skip, true, MAX_TRAINS, MAX_ECHOES, num_echoes, vieworder_sort_file, vieworder_dab_file))
			error("Error executing avg_ksp_from_view_files\n");
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
		md_free(basis);
	}

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


