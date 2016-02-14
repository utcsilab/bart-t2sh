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



static const char* usage_str = "<vieworder_sort> <vieworder_sort_dab> <ksp_in> <data_out>";
static const char* help_str = "Reorder data according to vieworder files and output corresponding kspace.\n"
"Designed so that data -> te_prep -> te_unprep ==  data\n"
"and ksp -> te_unprep -> te_prep ==  ksp\n"
"\t-b: perform weighted average along time\n";



int main(int argc, char* argv[])
{
	double start_time = timestamp();

	mini_cmdline(argc, argv, 4, usage_str, help_str);

	char vieworder_sort_name[100];
	char vieworder_dab_name[100];

	long dat_dims[DIMS];
	long ksp_dims[DIMS];


	// -----------------------------------------------------------
	// load data

	sprintf(vieworder_sort_name, "%s", argv[1]);
	sprintf(vieworder_dab_name, "%s", argv[2]);

	complex float* ksp = load_cfl(argv[3], DIMS, ksp_dims);

	fftmod(DIMS, ksp_dims, PHS2_FLAG, ksp, ksp);

	md_select_dims(DIMS, ~TE_FLAG, dat_dims, ksp_dims);

	complex float* dat = create_cfl(argv[4], DIMS, dat_dims);

		if( 0 != dat_from_view_files(DIMS, dat_dims, dat, ksp_dims, ksp, true, MAX_TRAINS, MAX_ECHOES, vieworder_sort_name, vieworder_dab_name))
			error("Error executing ksp_from_view_files\n");



	// -----------------------------------------------------------
	// cool down

	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, dat_dims, dat);

	double run_time = timestamp() - start_time;

	debug_printf(DP_INFO, "Total run time: %f seconds\n", run_time);

	exit(0);
}


