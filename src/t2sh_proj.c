/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd,	"Usage: %s [options] <input> <basis> <output>\n", name);
}


static void help(const char* name, FILE *fd)
{
	usage(name, fd);
	fprintf(fd,	"Compute temporal projection or backprojection.\n"
		"\t-f\tCompute x = Phi alpha\tDEFAULT: compute alpha = Phi^H x\n"
		"\t-K\tNumber of coefficients\tDEFAULT: all\n"
		"\t-z <xbar>\tUse zero-mean PCA with mean xbar \tDEFAULT: off\n"
		"\t-t <TE>\tCompute single TE \tDEFAULT: off\n"
		"\t-h\thelp\n");
}

int main_t2sh_proj(int argc, char* argv[])
{
	bool forward = false;
	bool zmean = false;
	long K = 0;
	char xbar_name[100];

	int single_TE = -1;

	int c;
	while (-1 != (c = getopt(argc, argv, "t:K:z:fh"))) {

		switch (c) {

		case 'K':
			K = atoi(optarg);
			break;

		case 'z':
			zmean = true;
			sprintf(xbar_name, "%s", optarg);
			break;

		case 'f':
			forward = true;
			break;

		case 't':
			single_TE = atoi(optarg);
			break;

		case 'h':
			help(argv[0], stdout);
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind == 0) {
		usage(argv[0], stderr);
		exit(1);
	}

	if (argc - optind != 3) {
		fprintf(stderr,"Input arguments do not match expected format.\n");
		help(argv[0], stderr);
		exit(1);
	}
		

	long in_dims[DIMS];
	long bas_dims[DIMS];
	long out_dims[DIMS];
	long max_dims[DIMS];
	long xbar_dims[DIMS];

	complex float* in_data = load_cfl(argv[optind + 0], DIMS, in_dims);
	complex float* bas_data = load_cfl(argv[optind + 1], DIMS, bas_dims);
	complex float* xbar_data = NULL;

	if (zmean) {
		xbar_data = load_cfl(xbar_name, DIMS, xbar_dims);
		assert(xbar_dims[TE_DIM] == bas_dims[TE_DIM]);
	}

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

	out_dims[ITER_DIM] = bas_dims[ITER_DIM];

	complex float* out_data = create_cfl(argv[optind + 2], DIMS, out_dims);
	md_clear(DIMS, out_dims, out_data, CFL_SIZE);

	long in_strs[DIMS];
	long bas_strs[DIMS];
	long out_strs[DIMS];
	long xbar_strs[DIMS];

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

	if (zmean)
		md_calc_strides(DIMS, xbar_strs, xbar_dims, CFL_SIZE);

	for (unsigned int i = 0; i < DIMS; i++)
		max_dims[i] = (in_dims[i] < out_dims[i] ? out_dims[i] : in_dims[i]);

	if (forward)
		max_dims[COEFF_DIM] = K;

	complex float* tmp = in_data;

	if (zmean) {
		tmp = md_alloc(DIMS, in_dims, CFL_SIZE);
		md_copy(DIMS, in_dims, tmp, in_data, CFL_SIZE);
	}

	if (zmean && !forward)
		md_zaxpy2(DIMS, in_dims, in_strs, tmp, -1., xbar_strs, xbar_data);

	(forward ? md_zfmac2 : md_zfmacc2)(DIMS, max_dims, out_strs, out_data, in_strs, tmp, bas_strs, bas);

	if (zmean && forward)
		md_zaxpy2(DIMS, out_dims, out_strs, out_data, 1., xbar_strs, xbar_data);

	if (zmean)
		md_free(tmp);

	if (-1 != single_TE)
		md_free(bas);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, bas_dims, bas_data);
	unmap_cfl(DIMS, out_dims, out_data);

	exit(0);
}


