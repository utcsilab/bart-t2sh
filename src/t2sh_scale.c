/* Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>


#include "misc/debug.h"
#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "sense/optcom.h"
#include "jtsense/jtrecon.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char* usage_str = "<input> [<scale>|<output>]";
static const char* help_str = "Compute scale factor. If <scake> specified, store scale factor in <scale>.\n"
	"If <output> specified, scale <input> and store in <output>.";


int main_t2sh_scale(int argc, char* argv[])
{
	cmdline(&argc, argv, 1, 3, usage_str, help_str, 0, NULL);

	num_init();

	bool do_scale = false;
	bool do_output = false;

	long dims[DIMS];

	complex float* in_data = load_cfl(argv[1], DIMS, dims);

	complex float* scale_data = NULL;
	complex float* out_data = NULL;

	if (3 <= argc)
		do_scale = true;

	if (4 == argc)
		do_output = true;

	float scaling = jt_estimate_scaling(dims, NULL, in_data);

	if (do_scale) {

		scale_data = create_cfl(argv[2], 1, MD_DIMS(1));
		scale_data[0] = scaling;
		unmap_cfl(1, MD_DIMS(1), scale_data);
	}

	if (do_output && scaling > 0.) {

		out_data = create_cfl(argv[3], DIMS, dims);
		md_zsmul(DIMS, dims, out_data, in_data, 1. / scaling);
		unmap_cfl(DIMS, dims, out_data);
	}

	unmap_cfl(DIMS, dims, in_data);

	exit(0);
}


