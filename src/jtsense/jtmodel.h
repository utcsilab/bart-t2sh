/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"


struct operator_s;
struct linop_s;

struct linop_s* jtmodel_init(const long max_dims[DIMS],
		const struct linop_s* sense_op, const struct linop_s* temporal_op, const struct linop_s* sample_op,
		const long pat_dims[DIMS], const _Complex float* pattern,
		const long bas_dims[DIMS], const _Complex float* basis,
		_Bool use_cfksp);


#ifdef __cplusplus
}
#endif


