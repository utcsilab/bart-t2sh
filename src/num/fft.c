/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <frankong@berkeley.edu>
 *
 * 
 * FFT. It uses FFTW or CUFFT internally.
 *
 *
 * Gauss, Carl F. 1805. "Nachlass: Theoria Interpolationis Methodo Nova
 * Tractata." Werke 3, pp. 265-327, Königliche Gesellschaft der
 * Wissenschaften, Göttingen, 1866
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include <fftw3.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "fft.h"
#undef fft_plan_s

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "fft-cuda.h"
#define LAZY_CUDA
#endif


void fftscale2(unsigned int N, const long dimensions[N], unsigned long flags, const long ostrides[N], complex float* dst, const long istrides[N], const complex float* src)
{
	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dimensions);

	float scale = 1. / sqrtf((float)md_calc_size(N, fft_dims));

	md_zsmul2(N, dimensions, ostrides, dst, istrides, src, scale);
}

void fftscale(unsigned int N, const long dims[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	fftscale2(N, dims, flags, strs, dst, strs, src);
}


static double fftmod_phase(long length, int j)
{
	long center1 = length / 2;
	double shift = (double)center1 / (double)length;
	return ((double)j - (double)center1 / 2.) * shift;
}

static void fftmod2_r(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src, bool inv, double phase)
{
	if (0 == flags) {

		md_zsmul2(N, dims, ostrs, dst, istrs, src, cexp(M_PI * 2.i * (inv ? -phase : phase)));
		return;
	}


	/* this will also currently be slow on the GPU because we do not
	 * support strides there on the lowest level */

	unsigned int i = N - 1;
	while (!MD_IS_SET(flags, i))
		i--;

#if 1
	// If there is only one dimensions left and it is the innermost
	// which is contiguous optimize using md_zfftmod2

	if ((0u == MD_CLEAR(flags, i)) && (1 == md_calc_size(i, dims))
		&& (CFL_SIZE == ostrs[i]) && (CFL_SIZE == istrs[i])) {

		md_zfftmod2(N - i, dims + i, ostrs + i, dst, istrs + i, src, inv, phase);
		return;
	}
#endif

	long tdims[N];
	md_select_dims(N, ~MD_BIT(i), tdims, dims);

	#pragma omp parallel for
	for (int j = 0; j < dims[i]; j++)
		fftmod2_r(N, tdims, MD_CLEAR(flags, i),
			ostrs, (void*)dst + j * ostrs[i], istrs, (void*)src + j * istrs[i],
			inv, phase + fftmod_phase(dims[i], j));
}


static unsigned int md_find_first_case_element(unsigned int N, const long dims[N])
{
	unsigned int i;
	// Find the first non-zero dimension
	for (i = 0; i < N; ++i) {
		if (dims[i] != 1)
			return i;
	}

	return N-1; // All dimensions are singletons.
}

static unsigned int md_find_last_case_element(unsigned int N, unsigned long flags, unsigned int first)
{
	// Find the last dimension of the contiguous group
	unsigned int j = MD_IS_SET(flags, first) != 0; // This is the case
	unsigned int k;
	for (k = first + 1; k < N; ++k)
		if ((MD_IS_SET(flags, k) != 0) != j)
			break;
	return --k;
}


static void fftmod_even(unsigned int N, const long dims[N], unsigned long flags, complex float* dst, const complex float* src)
{
	// Get the total number of elements
	long relevant_dims[N];
	md_select_dims(N, flags, relevant_dims, dims);

	unsigned int first_case_element = md_find_first_case_element(N, dims);
	unsigned int case_element = md_find_last_case_element(N, flags, first_case_element);
	long total_elements = md_calc_size(N, dims);
	long inner_elements = md_calc_size(first_case_element+1, dims);
	long outer_loop_size = total_elements / inner_elements;
	
	// Compute whether the first element is 1 or -1 depending on the array sizes.
        double tmp_phase = .0;
        for (int p =  0; p < N ; p++) {
                if (relevant_dims[p] > 1)
                        tmp_phase += fftmod_phase(relevant_dims[p], 0);
        }

        double rem = tmp_phase - floor(tmp_phase);
        float initial_phase = .0;
        if (rem == 0.)
                initial_phase = 1.;
        else if (rem == 0.5)
                initial_phase = -1.;

	// Check what case is this by looking at the innermost dimensions
	if (!MD_IS_SET(flags, case_element)) {
		// Case #1: the innermost dimensions are not flagged, thus elements are multiplied by the same value
		#pragma omp parallel for
		for (long ol = 0; ol < outer_loop_size; ol++) {
			// Capture the phase for the first element and the other will be the same

			long iter_i[N];
			long ii = ol * inner_elements;
			long phase_idx = 0;
		
			for (int p =  0; p < N && ii > 0; p++) {
				if (dims[p] <=1)
					continue;
				iter_i[p] = ii % dims[p];
				ii /= dims[p];
				// Compute also the phase and the position in the src and dst arrays
				phase_idx += iter_i[p] * (relevant_dims[p] > 1);
			}

			float phase = (phase_idx % 2 == 0) * initial_phase + (phase_idx % 2 == 1) * (-initial_phase);
			long last_element = (ol+1) * inner_elements;

			#pragma omp simd
			for (long il = ol * inner_elements; il < last_element; il++) {
				dst[il] = src[il] * phase;
			}
		}
	} else {
		// Case #2: the innermost dimensions are flagged, thus elements are multiplied by phases with alternating sign
		#pragma omp parallel for
		for (long ol = 0; ol < outer_loop_size; ol++) {
			// Capture the phase for the first element and the other will be alternating for the innermost block

			long iter_i[N];
			long ii = ol * inner_elements;
			long phase_idx = 0;
		
			for (int p =  0; p < N && ii > 0; p++) {
				if (dims[p] <=1)
					continue;
				iter_i[p] = ii % dims[p];
				ii /= dims[p];
				// Compute also the phase and the position in the src and dst arrays
				phase_idx += iter_i[p] * (relevant_dims[p] > 1);
			}
			float phase = (phase_idx % 2 == 0) * initial_phase + (phase_idx % 2 == 1) * (-initial_phase);
			long last_element = (ol+1) * inner_elements;

			#pragma omp simd
			for (long il = ol * inner_elements; il < last_element; il++) {
				float alt_phase = (il % 2 == 0) * phase + (il % 2 == 1) * (-phase);
				dst[il] = src[il] * alt_phase;
			}
		}
	}
}


static unsigned long clear_singletons(unsigned int N, const long dims[N], unsigned long flags)
{
       return (0 == N) ? flags : clear_singletons(N - 1, dims, (1 == dims[N - 1]) ? MD_CLEAR(flags, N - 1) : flags);
}


void fftmod2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	fftmod2_r(N, dims, clear_singletons(N, dims, flags), ostrs, dst, istrs, src, false, 0.);
}


/*
 *	The correct usage is fftmod before and after fft and
 *      ifftmod before and after ifft (this is different from
 *	how fftshift/ifftshift has to be used)
 */
void ifftmod2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	fftmod2_r(N, dims, clear_singletons(N, dims, flags), ostrs, dst, istrs, src, true, 0.);
}

void fftmod(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long relevant_dims[N];
	md_select_dims(N, flags, relevant_dims, dimensions);

	if (!md_calc_all_modulo(N, relevant_dims, 4)) {
		// If at least one active dimensions (size >1) is not modulo 4 then fall back to general code
		long strs[N];
		md_calc_strides(N, strs, dimensions, CFL_SIZE);
		fftmod2(N, dimensions, flags, strs, dst, strs, src);
	} else {
		// Otherwise run faster code case
		fftmod_even(N, dimensions, clear_singletons(N, dimensions, flags), dst, src);
	}
}

void ifftmod(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long relevant_dims[N];
	md_select_dims(N, flags, relevant_dims, dimensions);

	if (!md_calc_all_modulo(N, relevant_dims, 4)) {
		// If at least one active dimensions (size >1) is not modulo 4 then fall back to general code
		long strs[N];
		md_calc_strides(N, strs, dimensions, CFL_SIZE);
		ifftmod2(N, dimensions, flags, strs, dst, strs, src);
	} else {
		// Otherwise run faster code case
		fftmod_even(N, dimensions, clear_singletons(N, dimensions, flags), dst, src);
	}
}

void ifftshift2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long pos[N];
	md_set_dims(N, pos, 0);
	for (unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pos[i] = dims[i] - dims[i] / 2;

	md_circ_shift2(N, dims, pos, ostrs, dst, istrs, src, CFL_SIZE);
}

void ifftshift(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	ifftshift2(N, dimensions, flags, strs, dst, strs, src);
}

void fftshift2(unsigned int N, const long dims[N], unsigned long flags, const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	long pos[N];
	md_set_dims(N, pos, 0);
	for (unsigned int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			pos[i] = dims[i] / 2;

	md_circ_shift2(N, dims, pos, ostrs, dst, istrs, src, CFL_SIZE);
}

void fftshift(unsigned int N, const long dimensions[N], unsigned long flags, complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);
	fftshift2(N, dimensions, flags, strs, dst, strs, src);
}



struct fft_plan_s {

	INTERFACE(operator_data_t);

	fftwf_plan fftw;

#ifdef  USE_CUDA
#ifdef	LAZY_CUDA
	unsigned int D;
	unsigned long flags;
	bool backwards;
	const long* dims;
	const long* istrs;
	const long* ostrs;
#endif
	struct fft_cuda_plan_s* cuplan;
#endif
};

static DEF_TYPEID(fft_plan_s);



static fftwf_plan fft_fftwf_plan(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards, bool measure)
{
	unsigned int N = D;
	fftwf_iodim64 dims[N];
	fftwf_iodim64 hmdims[N];
	unsigned int k = 0;
	unsigned int l = 0;

	//FFTW seems to be fine with this
	//assert(0 != flags); 

	for (unsigned int i = 0; i < N; i++) {

		if (MD_IS_SET(flags, i)) {

			dims[k].n = dimensions[i];
			dims[k].is = istrides[i] / CFL_SIZE;
			dims[k].os = ostrides[i] / CFL_SIZE;
			k++;

		} else  {

			hmdims[l].n = dimensions[i];
			hmdims[l].is = istrides[i] / CFL_SIZE;
			hmdims[l].os = ostrides[i] / CFL_SIZE;
			l++;
		}
	}

	fftwf_plan fftwf;

	#pragma omp critical
	fftwf = fftwf_plan_guru64_dft(k, dims, l, hmdims, (complex float*)src, dst,
				backwards ? 1 : (-1), measure ? FFTW_MEASURE : FFTW_ESTIMATE);

	return fftwf;
}


static void fft_apply(const operator_data_t* _plan, unsigned int N, void* args[N])
{
	complex float* dst = args[0];
	const complex float* src = args[1];
	const auto plan = CAST_DOWN(fft_plan_s, _plan);

	assert(2 == N);

#ifdef  USE_CUDA
	if (cuda_ondevice(src)) {
#ifdef	LAZY_CUDA
          if (NULL == plan->cuplan)
		((struct fft_plan_s*)plan)->cuplan = fft_cuda_plan(plan->D, plan->dims, plan->flags, plan->ostrs, plan->istrs, plan->backwards);
#endif
		assert(NULL != plan->cuplan);
		fft_cuda_exec(plan->cuplan, dst, src);

	} else 
#endif
	{
		assert(NULL != plan->fftw);
		fftwf_execute_dft(plan->fftw, (complex float*)src, dst);
	}
}


static void fft_free_plan(const operator_data_t* _data)
{
	const auto plan = CAST_DOWN(fft_plan_s, _data);

	fftwf_destroy_plan(plan->fftw);
#ifdef	USE_CUDA
#ifdef	LAZY_CUDA
	xfree(plan->dims);
	xfree(plan->istrs);
	xfree(plan->ostrs);
#endif
	if (NULL != plan->cuplan)
		fft_cuda_free_plan(plan->cuplan);
#endif
	xfree(plan);
}


const struct operator_s* fft_measure_create(unsigned int D, const long dimensions[D], unsigned long flags, bool inplace, bool backwards)
{
	PTR_ALLOC(struct fft_plan_s, plan);
	SET_TYPEID(fft_plan_s, plan);

	complex float* src = md_alloc(D, dimensions, CFL_SIZE);
	complex float* dst = inplace ? src : md_alloc(D, dimensions, CFL_SIZE);

	long strides[D];
	md_calc_strides(D, strides, dimensions, CFL_SIZE);

	plan->fftw = fft_fftwf_plan(D, dimensions, flags, strides, dst, strides, src, backwards, true);

	md_free(src);

	if (!inplace)
		md_free(dst);

#ifdef  USE_CUDA
	plan->cuplan = NULL;
#ifndef LAZY_CUDA
	if (cuda_ondevice(src))
          plan->cuplan = fft_cuda_plan(D, dimensions, flags, strides, strides, backwards);
#else
	plan->D = D;
	plan->flags = flags;
	plan->backwards = backwards;

	PTR_ALLOC(long[D], dims);
	md_copy_dims(D, *dims, dimensions);
	plan->dims = *PTR_PASS(dims);

	PTR_ALLOC(long[D], istrs);
	md_copy_strides(D, *istrs, strides);
	plan->istrs = *PTR_PASS(istrs);

	PTR_ALLOC(long[D], ostrs);
	md_copy_strides(D, *ostrs, strides);
	plan->ostrs = *PTR_PASS(ostrs);
#endif
#endif
	return operator_create2(D, dimensions, strides, D, dimensions, strides, CAST_UP(PTR_PASS(plan)), fft_apply, fft_free_plan);
}


const struct operator_s* fft_create2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards)
{
	PTR_ALLOC(struct fft_plan_s, plan);
	SET_TYPEID(fft_plan_s, plan);

	plan->fftw = fft_fftwf_plan(D, dimensions, flags, ostrides, dst, istrides, src, backwards, false);

#ifdef  USE_CUDA
	plan->cuplan = NULL;
#ifndef LAZY_CUDA
	if (cuda_ondevice(src))
		plan->cuplan = fft_cuda_plan(D, dimensions, flags, ostrides, istrides, backwards);
#else
	plan->D = D;
	plan->flags = flags;
	plan->backwards = backwards;

	PTR_ALLOC(long[D], dims);
	md_copy_dims(D, *dims, dimensions);
	plan->dims = *PTR_PASS(dims);

	PTR_ALLOC(long[D], istrs);
	md_copy_strides(D, *istrs, istrides);
	plan->istrs = *PTR_PASS(istrs);

	PTR_ALLOC(long[D], ostrs);
	md_copy_strides(D, *ostrs, ostrides);
	plan->ostrs = *PTR_PASS(ostrs);
#endif
#endif

	return operator_create2(D, dimensions, ostrides, D, dimensions, istrides, CAST_UP(PTR_PASS(plan)), fft_apply, fft_free_plan);
}

const struct operator_s* fft_create(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src, bool backwards)
{
	long strides[D];
	md_calc_strides(D, strides, dimensions, CFL_SIZE);

	return fft_create2(D, dimensions, flags, strides, dst, strides, src, backwards);
}




void fft_exec(const struct operator_s* o, complex float* dst, const complex float* src)
{
	operator_apply_unchecked(o, dst, src);
}




void fft_free(const struct operator_s* o)
{
	operator_free(o);
}


void fft2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	const struct operator_s* plan = fft_create2(D, dimensions, flags, ostrides, dst, istrides, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fft(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, false);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void ifft(unsigned int D, const long dimensions[D], unsigned long flags, complex float* dst, const complex float* src)
{
	const struct operator_s* plan = fft_create(D, dimensions, flags, dst, src, true);
	fft_exec(plan, dst, src);
	fft_free(plan);
}

void fftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftmod(D, dimensions, flags, dst, src);
	fft(D, dimensions, flags, dst, dst);
	fftmod(D, dimensions, flags, dst, dst);
}

void ifftc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifftmod(D, dimensions, flags, dst, src);
	ifft(D, dimensions, flags, dst, dst);
	ifftmod(D, dimensions, flags, dst, dst);
}

void fftc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftmod2(D, dimensions, flags, ostrides, dst, istrides, src);
	fft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	fftmod2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifftmod2(D, dimensions, flags, ostrides, dst, istrides, src);
	ifft2(D, dimensions, flags, ostrides, dst, ostrides, dst);
	ifftmod2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftu(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifft(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftu2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftu2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifft2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void fftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	fftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void ifftuc(unsigned int D, const long dimensions[__VLA(D)], unsigned long flags, complex float* dst, const complex float* src)
{
	ifftc(D, dimensions, flags, dst, src);
	fftscale(D, dimensions, flags, dst, dst);
}

void fftuc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	fftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}

void ifftuc2(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src)
{
	ifftc2(D, dimensions, flags, ostrides, dst, istrides, src);
	fftscale2(D, dimensions, flags, ostrides, dst, ostrides, dst);
}


bool fft_threads_init = false;

void fft_set_num_threads(unsigned int n)
{
#ifdef FFTWTHREADS
	#pragma omp critical
	if (!fft_threads_init) {

		fft_threads_init = true;
		fftwf_init_threads();
	}

	#pragma omp critical
        fftwf_plan_with_nthreads(n);
#else
	UNUSED(n);
#endif
}



