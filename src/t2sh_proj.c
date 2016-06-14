/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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

#ifndef DIMS
#define DIMS 16
#endif


static const char* usage_str = "<input> <basis> <output>";
static const char* help_str = "Compute temporal projection or backprojection.";


int main_t2sh_proj(int argc, char* argv[])
{
	bool forward = false;
	unsigned int K = 0;
	int single_TE = -1;

	const struct opt_s opts[] = {

		OPT_UINT('K', &K, "K", "Subspace size"),
		OPT_SET('f', &forward, "Compute forward: x = Phi alpha"),
		OPT_INT('t', &single_TE, "TE", "Compute single TE"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long in_dims[DIMS];
	long bas_dims[DIMS];
	long out_dims[DIMS];
	long max_dims[DIMS];

	complex float* in_data = load_cfl(argv[1], DIMS, in_dims);
	complex float* bas_data = load_cfl(argv[2], DIMS, bas_dims);


	// FIXME: if the basis is truncated, then the forward operator will not work.
	// might be better to use bas_dims[TE_DIM] for forward
	long num_coeffs = bas_dims[COEFF_DIM];

	if (0 == K)
		K = num_coeffs;

	if (-1 != single_TE) {
		assert(single_TE >= 0);
		assert(single_TE < num_coeffs);
	}

	if (forward) {

		md_select_dims(DIMS, ~COEFF_FLAG, out_dims, in_dims);

		if (-1 != single_TE)
			out_dims[TE_DIM] = 1;
		else
			out_dims[TE_DIM] = num_coeffs;
	}
	else {

		md_select_dims(DIMS, ~TE_FLAG, out_dims, in_dims);
		out_dims[COEFF_DIM] = K;
	}

	complex float* out_data = create_cfl(argv[3], DIMS, out_dims);
	md_clear(DIMS, out_dims, out_data, CFL_SIZE);

	long in_strs[DIMS];
	long bas_strs[DIMS];
	long out_strs[DIMS];

	complex float* bas = bas_data;
	long single_bas_dims[DIMS];

	if (-1 != single_TE) {
		md_select_dims(DIMS, ~TE_FLAG, single_bas_dims, bas_dims);
		bas = md_alloc(DIMS, single_bas_dims, CFL_SIZE);

		long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };
		pos2[TE_DIM] = single_TE;
		md_slice(DIMS, TE_FLAG, pos2, bas_dims, bas, bas_data, CFL_SIZE);
	}
	else
		md_copy_dims(DIMS, single_bas_dims, bas_dims);


	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);
	md_calc_strides(DIMS, bas_strs, single_bas_dims, CFL_SIZE);
	md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

	for (unsigned int i = 0; i < DIMS; i++)
		max_dims[i] = (in_dims[i] < out_dims[i] ? out_dims[i] : in_dims[i]);

	if (forward)
		max_dims[COEFF_DIM] = K;

	(forward ? md_zfmac2 : md_zfmacc2)(DIMS, max_dims, out_strs, out_data, in_strs, in_data, bas_strs, bas);

	if (-1 != single_TE)
		md_free(bas);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, bas_dims, bas_data);
	unmap_cfl(DIMS, out_dims, out_data);

	exit(0);
}


