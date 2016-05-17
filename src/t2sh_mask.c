/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */


#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(_Complex float)
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-s] <e2s> <ETL> <kspace> <vieworder_file> <vieworder_mask>\n", name);
}

static void help(const char* name, FILE *fd)
{
	usage(name, fd);
	fprintf(fd,	"Compute temporal mask from vieworder file and match dimensions of kspace\n"
			"\t<e2s>\techoes2skip\n"
			"\t<ETL>\techo train length\n"
			"\t-s\tfile does not contain the first (header) line\n" 
		"\t-h\thelp\n");
}

static int read_vieworder_file(char* filename, _Bool skip, unsigned int D, long dims[D], unsigned int echoes2skip, complex float* te_mask)
{
	FILE *fd;
	char line_buffer[BUFSIZ];
	long line_number = 0;

	fd = fopen(filename, "r");
	if (!fd)
		error("Couldn't open file %s for reading.\n", filename);

	if (0 == fgets(line_buffer, sizeof(line_buffer), fd))
		return -1;

	if (skip == false) {
		if (0 != sscanf(line_buffer, "index train echo y z\n"))
			return -1;
	}

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	long mask_pos[D];
	md_set_dims(D, mask_pos, 0);

	long trash;
	int i;

	debug_printf(DP_DEBUG3, "dims = \n");
	debug_print_dims(DP_DEBUG3, D, dims);
	while (fgets(line_buffer, sizeof(line_buffer), fd)) {

		if (5 == (i = sscanf(line_buffer, "%ld %ld %ld %ld %ld \n", &trash, &trash, &mask_pos[TE_DIM], &mask_pos[PHS1_DIM], &mask_pos[PHS2_DIM])) ){
			//debug_printf(DP_DEBUG3, "te = %ld\tky=%ld\tkz=%d\n", mask_pos[TE_DIM], mask_pos[PHS1_DIM], mask_pos[PHS2_DIM]);
			if (mask_pos[PHS1_DIM] != -1 && mask_pos[PHS2_DIM] != -1) {
				mask_pos[TE_DIM] -= echoes2skip;
				//debug_print_dims(DP_DEBUG3, D, mask_pos);
				long idx = md_calc_offset(D, strs, mask_pos);
				te_mask[idx / CFL_SIZE] = 1.;
			}
			md_set_dims(D, mask_pos, 0);
		}

		++line_number;
	}

	fclose(fd);
	return 0;
}

int main_t2sh_mask(int argc, char* argv[])
{
	int c;
	_Bool skip = false;

	while (-1 != (c = getopt(argc, argv, "hs"))) {

		switch (c) {

		case 's':
			skip = true;
			break;

		case 'h':
			help(argv[0], stdout);
			exit(0);

		default:
			help(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 5) {

		fprintf(stderr, "Input arguments do not match expected format.\n");
		usage(argv[0], stderr);
		exit(1);
	}

	unsigned int D = DIMS;
	long in_dims[D];
	long out_dims[D];

	int echoes2skip = atoi(argv[optind + 0]);
	int T = atoi(argv[optind + 1]);
	
	complex float* kspace = load_cfl(argv[optind + 2], D, in_dims);

	md_select_dims(D, (PHS1_FLAG | PHS2_FLAG), out_dims, in_dims);
	out_dims[TE_DIM] = T;
	
	complex float* te_mask = create_cfl(argv[optind + 4], D, out_dims);
	md_clear(D, out_dims, te_mask, CFL_SIZE);

	if (0 != read_vieworder_file(argv[optind + 3], skip, D, out_dims, echoes2skip, te_mask))
		error("read_vieworder_file failed\n");

	unmap_cfl(D, in_dims, kspace);
	unmap_cfl(D, out_dims, te_mask);
	exit(0);
}




