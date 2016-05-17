/*
 * 2014	Jonathan Tamir <jtamir@eecs.berkeley.edu>
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

#include "jtsense/common.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"

#include "misc/debug.h"


#ifndef DIMS
#define DIMS KSPACE_DIMS
#endif



static const char* usage_str = "[-b] <echoes2skip> <num_echoes> <vieworder_sort> <vieworder_sort_dab> <data_in> <kspace_out>";
static const char* help_str = "Reorder data according to vieworder files and output corresponding kspace.\n\t-b: perform weighted average along time\n";



int main_t2sh_prep(int argc, char* argv[])
{
	double start_time = timestamp();

	bool wavg = mini_cmdline_bool(argc, argv, 'b', 6, usage_str, help_str);

	char vieworder_sort_name[100];
	char vieworder_dab_name[100];

	unsigned int echoes2skip = 0;
	unsigned int num_echoes = 0;

	long dat_dims[DIMS];
	long ksp_dims[DIMS];


	// -----------------------------------------------------------
	// load data

	echoes2skip = atoi(argv[1]);
	num_echoes = atoi(argv[2]);
	sprintf(vieworder_sort_name, "%s", argv[3]);
	sprintf(vieworder_dab_name, "%s", argv[4]);

	complex float* dat = load_cfl(argv[5], DIMS, dat_dims);

	if (wavg == 0 && dat_dims[READ_DIM] > 1)
		debug_printf(DP_WARN, "Warning: 3D volume expanding into time!\n");

	md_copy_dims(DIMS, ksp_dims, dat_dims);

	if (0 == wavg)
		ksp_dims[TE_DIM] = num_echoes;

	complex float* ksp = create_cfl(argv[6], DIMS, ksp_dims);

	if (wavg) {

		// expand kspace into time bins, reorder data, and average
		if( 0 != wavg_ksp_from_view_files(DIMS, ksp_dims, ksp, dat_dims, dat, echoes2skip, true, MAX_TRAINS, MAX_ECHOES, num_echoes, vieworder_sort_name, vieworder_dab_name))
			error("Error executing wavg_ksp_from_view_files\n");
	}
	else {

		// expand kspace into time bins and reorder data
		if( 0 != ksp_from_view_files(DIMS, ksp_dims, ksp, dat_dims, dat, echoes2skip, 0, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name))
			error("Error executing ksp_from_view_files\n");
	}

	fftmod(DIMS, ksp_dims, PHS2_FLAG, ksp, ksp);


	// -----------------------------------------------------------
	// cool down

	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, dat_dims, dat);

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


