
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "optest.h"

#define ADJOINT_EPS 1.E-4


static bool test_adjoint_fun_priv(long N, long M, void* data_frw, fun_t frw, void* data_adj, fun_t adj, bool gpu)
{

	complex float* x = NULL;
	if (gpu)
#ifdef USE_CUDA
		x = md_alloc_gpu(1, MD_DIMS(N), CFL_SIZE);
#else
	assert(0);
#endif
	else
		x = md_alloc(1, MD_DIMS(N), CFL_SIZE);

	complex float* AHy = md_alloc_sameplace(1, MD_DIMS(N), CFL_SIZE, x);
	complex float* y = md_alloc_sameplace(1, MD_DIMS(M), CFL_SIZE, x);
	complex float* Ax = md_alloc_sameplace(1, MD_DIMS(M), CFL_SIZE, x);

	md_gaussian_rand(1, MD_DIMS(N), x);
	md_gaussian_rand(1, MD_DIMS(M), y);

	frw(data_frw, Ax, x);
	adj(data_adj, AHy, y);

 	complex float sx = md_zscalar(1, MD_DIMS(N), x, AHy);
	complex float sy = md_zscalar(1, MD_DIMS(M), Ax, y);

	float result = cabsf(sx - sy);
	debug_printf(DP_DEBUG3, "Adjoint test: abs(%f+%fi - %f+%fi) = %f\n", crealf(sx), cimagf(sx), crealf(sy), cimagf(sy), result);

	md_free(x);
	md_free(AHy);
	md_free(y);
	md_free(Ax);

	return (result < ADJOINT_EPS);
}


bool test_adjoint_fun(long N, long M, void* data, fun_t frw, fun_t adj, bool gpu)
{
	return test_adjoint_fun_priv(N, M, data, frw, data, adj, gpu);
}


bool test_adjoint_op(const struct operator_s* frw, const struct operator_s* adj, bool gpu)
{
	long frw_N = md_calc_size(operator_domain(frw)->N, operator_domain(frw)->dims);
	long frw_M = md_calc_size(operator_codomain(frw)->N, operator_codomain(frw)->dims);

	long adj_N = md_calc_size(operator_codomain(adj)->N, operator_codomain(adj)->dims);
	long adj_M = md_calc_size(operator_domain(adj)->N, operator_domain(adj)->dims);

	assert( (frw_N == adj_N) && (frw_M == adj_M) ); 

	return test_adjoint_fun_priv(frw_N, frw_M, (void*)frw, (fun_t)operator_apply_unchecked, (void*)adj, (fun_t)operator_apply_unchecked, gpu);
}


bool test_adjoint_linop(const struct linop_s* linop, bool gpu)
{
	return test_adjoint_op(linop->forward, linop->adjoint, gpu);
}


// we test complex , but this might not be necessary
// for use with iterative algorithms
bool test_derivative_fun(long N, long M, void* data, fun_t frw, fun_t der)
{
	complex float* x0 = md_alloc(1, MD_DIMS(N), CFL_SIZE);
	complex float* xd = md_alloc(1, MD_DIMS(N), CFL_SIZE);
	complex float* x1 = md_alloc(1, MD_DIMS(N), CFL_SIZE);
	complex float* y0 = md_alloc(1, MD_DIMS(M), CFL_SIZE);
	complex float* yd = md_alloc(1, MD_DIMS(M), CFL_SIZE);
	complex float* y1a = md_alloc(1, MD_DIMS(M), CFL_SIZE);
	complex float* y1b = md_alloc(1, MD_DIMS(M), CFL_SIZE);

	md_gaussian_rand(1, MD_DIMS(N), x0);
	md_gaussian_rand(1, MD_DIMS(N), xd);

	float nrmse = 0.;

	for (int i = 0; i < 15; i++) {

		frw(data, y0, x0);
		der(data, yd, xd);

		md_zadd(1, MD_DIMS(N), x1, x0, xd);
		frw(data, y1a, x1);

		md_zadd(1, MD_DIMS(M), y1b, y0, yd);

		nrmse = md_znrmse(1, MD_DIMS(M), y1a, y1b);

		debug_printf(DP_DEBUG2, "Derivative test: %d %f\n", i, log2f(nrmse));

		md_zsmul(1, MD_DIMS(N), xd, xd, 1. / 2.);
	}

	md_free(x0);
	md_free(xd);
	md_free(x1);
	md_free(y0);
	md_free(yd);
	md_free(y1a);
	md_free(y1b);

	return true; // FIXME
}


	
