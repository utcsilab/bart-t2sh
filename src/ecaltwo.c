/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/fft.h"

#include "calib/cc.h"
#include "calib/calib.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char usage_str[] = "x y z <input> <sensitivities> [<ev_maps>]";
static const char help_str[] =
		"Second part of ESPIRiT calibration.\n"
		"Optionally outputs the eigenvalue maps.";





int main_ecaltwo(int argc, char* argv[])
{
	long maps = 2; // channels;
	struct ecalib_conf conf = ecalib_defaults;

	const char* cal_data_file = NULL;

	const struct opt_s opts[] = {

		OPT_FLOAT('c', &conf.crop, "crop_value", "Crop the sensitivities if the eigenvalue is smaller than {crop_value}."),
		OPT_LONG('m', &maps, "maps", "Number of maps to compute."),
		OPT_SET('S', &conf.softcrop, "Create maps with smooth transitions (Soft-SENSE)."),
		OPT_CLEAR('O', &conf.orthiter, "()"),
		OPT_SET('g', &conf.usegpu, "()"),
		OPT_CLEAR('P', &conf.rotphase, "If <cal_data> is supplied, do not rotate the phase with respect to the first principal component"),
		OPT_FLOAT('v', &conf.var, "variance", "Variance of noise in data."),
		OPT_SET('a', &conf.automate, "Automatically pick crop (second) threshold (must supply <cal_data>."),
		OPT_STRING('i', &cal_data_file, "<cal_data>", "(use calibration data in <file>)"),
	};

	cmdline(&argc, argv, 5, 6, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (conf.automate)
		conf.orthiter = false;


	long in_dims[DIMS];
	long cal_dims[DIMS];

	complex float* in_data = load_cfl(argv[4], DIMS, in_dims);

	complex float* cal_data = (NULL == cal_data_file ) ? NULL : load_cfl(cal_data_file, DIMS, cal_dims);

	long channels = 0;

	while (in_dims[3] != (channels * (channels + 1) / 2))
		channels++;

	debug_printf(DP_INFO, "Channels: %d\n", channels);

	assert(maps <= channels);

	if (NULL != cal_data)
		assert(cal_dims[3] == channels);


	long out_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	long map_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	
	out_dims[0] = atoi(argv[1]);
	out_dims[1] = atoi(argv[2]);
	out_dims[2] = atoi(argv[3]);
	out_dims[3] = channels;
	out_dims[4] = maps;

	assert((out_dims[0] >= in_dims[0]));
	assert((out_dims[1] >= in_dims[1]));
	assert((out_dims[2] >= in_dims[2]));


	for (int i = 0; i < 3; i++)
		map_dims[i] = out_dims[i];

	map_dims[3] = 1;
	map_dims[4] = maps;


	complex float* out_data = create_cfl(argv[5], DIMS, out_dims);
	complex float* emaps;

	if (7 == argc)
		emaps = create_cfl(argv[6], DIMS, map_dims);
	else
		emaps = md_alloc(DIMS, map_dims, CFL_SIZE);

	caltwo(&conf, out_dims, out_data, emaps, in_dims, in_data, NULL, NULL);

	if (conf.intensity) {

		debug_printf(DP_DEBUG1, "Normalize...\n");

		normalizel1(DIMS, COIL_FLAG, out_dims, out_data);
	}

	float c = conf.crop;

	if (NULL != cal_data && conf.automate) {

		debug_printf(DP_DEBUG2, "SURE Crop... (var = %.2f)\n", conf.var);
		c = sure_crop(conf.var, out_dims, out_data, emaps, cal_dims, cal_data);
	}

	debug_printf(DP_DEBUG1, "Crop maps... (%.2f)\n", c);

	crop_sens(out_dims, out_data, conf.softcrop, c, emaps);

	debug_printf(DP_DEBUG1, "Fix phase...\n");

	complex float rot[channels][channels];

	if (NULL != cal_data && conf.rotphase) {

		// rotate the the phase with respect to the first principle component
		long scc_dims[DIMS] = MD_INIT_ARRAY(DIMS, 1);
		scc_dims[COIL_DIM] = channels;
		scc_dims[MAPS_DIM] = channels;
		scc(scc_dims, &rot[0][0], cal_dims, cal_data);

	} else {

		for (unsigned int i = 0; i < channels; i++)
			for (unsigned int j = 0; j < channels; j++)
				rot[i][j] = (i == j) ? 1. : 0.;
	}

	fixphase2(DIMS, out_dims, COIL_DIM, rot[0], out_data, out_data);

	debug_printf(DP_INFO, "Done.\n");

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	if (NULL != cal_data)
		unmap_cfl(DIMS, cal_dims, cal_data);

	if (7 == argc)
		unmap_cfl(DIMS, map_dims, emaps);
	else
		md_free(emaps);

	exit(0);
}


