#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "jtmodel_intel.h"

static void panel_forward(const complex float *restrict map0, const complex float *restrict map1,
		const complex float *restrict img0, const complex float *restrict img1,
		complex float *restrict tmppanel0,
		complex float *restrict tmppanel1,
		complex float *restrict cor_out, fftwf_plan plan, float sc,
		int m, int dim0, int dim1) {

	for (int row = 0; row < m; row++) {
		const complex float *_map0 = map0 + dim0 * row;
		const complex float *_map1 = map1 + dim0 * row;
		const complex float *_img0 = img0 + dim0 * row;
		const complex float *_img1 = img1 + dim0 * row;
		complex float *_tmppanel0 = tmppanel0 + dim0 * row;
		complex float *_tmppanel1 = tmppanel1 + dim0 * row;

		//#pragma simd
		for (int i = 0; i < dim0; i++) {
			_tmppanel0[i] = (_map0[i] * _img0[i] + _map1[i] * _img1[i]) * sc;
		}
		fftwf_execute_dft(plan, _tmppanel0, _tmppanel1);
	}
	// Transpose
	for (int col = 0; col < dim0; col++) {
		//#pragma simd
		for (int row = 0; row < m; row++) {
			cor_out[row + col * dim1] = tmppanel1[col + row * dim0];
		}
	}
}

static void panel_forward2(complex float *a, complex float *b, fftwf_plan plan, int m, int dim0, int dim1) {
	UNUSED(dim0);
	for (int col = 0; col < m; col++) {
		complex float *_a = a + col * dim1;
		complex float *_b = b + col * dim1;
		fftwf_execute_dft(plan, _a, _b);
	}
}

static void panel_forward3(complex float *a, complex float *b, complex float *cor_out,
		fftwf_plan plan, int m, int dim0, int dim1) {
	for (int col = 0; col < m; col++) {
		complex float *_a = a + col * dim1;
		complex float *_b = b + col * dim1;
		fftwf_execute_dft(plan, _a, _b);
	}
	// Transpose
	for (int row = 0; row < dim1; row++) {
		//#pragma simd
		for (int col = 0; col < m; col++) {
			cor_out[col + row * dim0] = b[row + col * dim1];
		}
	}
}

static void panel_forward4(const complex float *map0, const complex float *map1,
		complex float *tmpimg0, 
		complex float *tmpimg1,
		complex float *cor0,
		complex float *cor1, float sc,
		fftwf_plan plan, int m, int dim0, int dim1) {

	UNUSED(dim1);

	for (int row = 0; row < m; row++) {
		const complex float *_map0 = map0 + row * dim0;
		const complex float *_map1 = map1 + row * dim0;
		complex float *_tmpimg0 = tmpimg0 + row * dim0;
		complex float *_tmpimg1 = tmpimg1 + row * dim0;
		complex float *_cor0 = cor0 + row * dim0;
		complex float *_cor1 = cor1 + row * dim0;
		fftwf_execute_dft(plan, _tmpimg0, _tmpimg1);
		//#pragma simd
		for (int i = 0; i < dim0; i++) {
			float r0 = __real__ _map0[i];
			float r1 = __real__ _map1[i];
			float i0 = __imag__ _map0[i];
			float i1 = __imag__ _map1[i];
			float _r = __real__ _tmpimg1[i];
			float _i = __imag__ _tmpimg1[i];
			_cor0[i] += ((r0 * _r + i0 * _i) + (r0 * _i - i0 * _r) * _Complex_I) * sc;
			_cor1[i] += ((r1 * _r + i1 * _i) + (r1 * _i - i1 * _r) * _Complex_I) * sc;
		}
	}
}

void jtmodel_normal_benchmark_fast(const long cfimg_dims[DIMS],
		const long sens_dims[DIMS], const complex float *sens,
		const float *stkern_mat, complex float *dst, const complex float *src,
		fftwf_plan plan1d, fftwf_plan plan1d_inv, fftwf_plan plan1d_1, fftwf_plan plan1d_inv_1,
		unsigned int M, unsigned int N) {

	const unsigned long dim0 = cfimg_dims[PHS1_DIM];
	const unsigned long dim1 = cfimg_dims[PHS2_DIM];
	const unsigned long ncoils = sens_dims[COIL_DIM];
	const unsigned long ncoeffs = cfimg_dims[COEFF_DIM];

	complex float *cfksp0 = (complex float *)malloc(dim1 * dim0 * ncoils * ncoeffs *
			sizeof(complex float)); // 28
	complex float *cfksp1 = (complex float *)malloc(dim1 * dim0 * ncoils * ncoeffs *
			sizeof(complex float)); // 28
	complex float *cfksp2 = (complex float *)malloc(dim1 * dim0 * ncoils * ncoeffs *
			sizeof(complex float)); // 28
	complex float *cfksp3 = (complex float *)malloc(dim1 * dim0 * ncoils * ncoeffs *
			sizeof(complex float)); // 28

	complex float *tmpimg =
		(complex float *)malloc(dim0 * dim1 * sizeof(complex float));
	complex float *tmppanel0 =
		(complex float *)malloc(dim0 * dim1 * sizeof(complex float));
	complex float *tmppanel1 =
		(complex float *)malloc(dim0 * dim1 * sizeof(complex float));

	unsigned int P = (dim1 + M-1) / M;
	unsigned int P0 = (dim0 + M-1) / M;

	float sc = 1.0 / sqrt((double)dim0 * dim1);
	for (unsigned int map = 0; map < ncoils; map++) {
		for (unsigned int img = 0; img < ncoeffs; img++) {
			for (unsigned int p = 0; p < P; p++) {
				int rownum = ((p+1)*M > dim1) ? dim1-p*M: M;
				const complex float *map0 = sens + map * dim1 * dim0 + p * M * N;
				const complex float *map1 =
					sens + map * dim1 * dim0 + ncoils * dim0 * dim1 + p * M * N;
				const complex float *img0 = src + img * dim0 * dim1 * 2 + p * M * N;
				const complex float *img1 =
					src + dim0 * dim1 + img * dim0 * dim1 * 2 + p * M * N;
				complex float *tmpimg0 = tmpimg + p * M;
				panel_forward(map0, map1, img0, img1, tmppanel0, tmppanel1, tmpimg0,
						plan1d, sc, rownum,dim0, dim1);
			}
			for (unsigned int p = 0; p < P0; p++) {
				unsigned int rownum = ((p+1)*M > dim0) ? dim0-p*M: M;
				complex float *_tmpimg = tmpimg + p * M * dim1;
				complex float *cor_out =
					cfksp3 + map * dim1 * dim0 + img * ncoils * dim1 * dim0 + p * M * dim1;
				panel_forward2(_tmpimg, cor_out, plan1d_1, rownum, dim0, dim1);
			}
		}


		// stkern (ncoeffs x ncoeffs matrix multiplication)
		for (unsigned int pix_i = 0; pix_i < dim0; pix_i++) {
			for (unsigned int img_i = 0; img_i < ncoeffs; img_i++) {
				complex float *img_out = cfksp2 + map * dim1 * dim0 +
					img_i * dim0 * dim1 * ncoils + pix_i * dim1;
				//#pragma simd
				for (unsigned int pix = 0; pix < dim1; pix++) {
					img_out[pix] = 0;
				}
				for (unsigned int img_j = 0; img_j < ncoeffs; img_j++) {
					const complex float *img_in = cfksp3 + map * dim1 * dim0 +
						img_j * dim0 * dim1 * ncoils + pix_i * dim1;
					const float *mat = (img_i > img_j) ? stkern_mat + img_i * dim1 * dim0 + img_j * dim1 * dim0 * ncoeffs + pix_i * dim1 :
						stkern_mat + img_j * dim1 * dim0 + img_i * dim1 * dim0 * ncoeffs + pix_i * dim1;

					//#pragma simd
					for (unsigned int pix = 0; pix < dim1; pix++) {
						img_out[pix] += img_in[pix] * mat[pix];
					}
				}
			}
		}


		const complex float *map0 = sens + map * dim1 * dim0; // + img*ncoils*dim0*dim1*2;
		const complex float *map1 = sens + map * dim1 * dim0 +
			ncoils * dim0 * dim1; // + img*ncoils*dim0*dim1*2;
		for (unsigned int img = 0; img < ncoeffs; img++) {
			for (unsigned int p = 0; p < P0; p++) {
				int rownum = ((p+1)*M > dim0) ? dim0-p*M: M;
				complex float *img_in =
					cfksp2 + map * dim1 * dim0 + img * dim0 * dim1 * ncoils + p * M * dim1;
				complex float *tmpimg0 = tmpimg + p * M;

				panel_forward3(img_in, tmppanel0, tmpimg0, plan1d_inv_1, rownum, dim0, dim1);
			}
			for (unsigned int p = 0; p < P; p++) {
				unsigned int rownum = ((p+1)*M > dim1) ? dim1-p*M: M;
				complex float *cor0 = dst + img * dim1 * dim0 * 2 + p * M * N;
				complex float *cor1 =
					dst + dim1 * dim0 + img * dim1 * dim0 * 2 + p * M * N;
				const complex float *_map0 = map0 + p * M * N;
				const complex float *_map1 = map1 + p * M * N;
				complex float *tmpimg0 = tmpimg + p * M * N;
				if (map == 0) {
					//#pragma simd
					for (unsigned int pix = 0; pix < rownum * N; pix++) {
						cor0[pix] = 0;
						cor1[pix] = 0;
					}
				}
				panel_forward4(_map0, _map1, tmpimg0, tmppanel0, cor0, cor1, sc, plan1d_inv, rownum, dim0, dim1);
			}
		}
	}

	free(cfksp0);
	free(cfksp1);
	free(cfksp2);
	free(cfksp3);
}


#if 0
int main(int argc, char *argv[]) {
#pragma omp parallel
	{
		// Create matrices
		unsigned long ncoils = 7;
		unsigned long ncoeffs = 4;

		assert(argc==4);
		unsigned long dim0 = atoi(argv[1]);
		unsigned long dim1 = atoi(argv[2]);
		unsigned long m = atoi(argv[3]);
		printf("dim0, dim1, m,  %d %d %d\n", dim0, dim1, m);

		complex float *src =
			(complex float *)malloc(dim0 * dim1 * 2 * ncoeffs * sizeof(complex float));
		complex float *dst =
			(complex float *)malloc(dim0 * dim1 * 2 * ncoeffs * sizeof(complex float));
		complex float *dst_ref =
			(complex float *)malloc(dim0 * dim1 * 2 * ncoeffs * sizeof(complex float));
		complex float *sens =
			(complex float *)malloc(dim0 * dim1 * 2 * ncoils * sizeof(complex float));
		complex float *stkern_mat = (complex float *)malloc(
				dim0 * dim1 * ncoeffs * ncoeffs * sizeof(complex float));
		float *stkern_mat_trans = (float *)malloc(
				dim0 * dim1 * ncoeffs * ncoeffs * sizeof(float));
		complex float *tst1 = (complex float *)malloc(
				dim0*dim1*ncoils*ncoeffs*sizeof(complex float));
		complex float *tst2 = (complex float *)malloc(
				dim0*dim1*ncoils*ncoeffs*sizeof(complex float));

		fftwf_plan plan = fftwf_plan_dft_2d(dim1, dim0, src, dst, -1, FFTW_MEASURE);
		fftwf_plan plan2 = fftwf_plan_dft_2d(dim1, dim0, src, dst, 1, FFTW_MEASURE);
		fftwf_plan plan1d_0 = fftwf_plan_dft_1d(dim0, src, dst, -1, FFTW_MEASURE);
		fftwf_plan plan1d_inv_0 = fftwf_plan_dft_1d(dim0, src, dst, 1, FFTW_MEASURE);
		fftwf_plan plan1d_1 = fftwf_plan_dft_1d(dim1, src, dst, -1, FFTW_MEASURE);
		fftwf_plan plan1d_inv_1 = fftwf_plan_dft_1d(dim1, src, dst, 1, FFTW_MEASURE);

		set_all(src, dim0 * dim1 * 2 * ncoeffs);
		set_all(dst, dim0 * dim1 * 2 * ncoeffs);
		set_all(sens, dim0 * dim1 * 2 * ncoils);
		set_all(stkern_mat, dim0 * dim1 * ncoeffs * ncoeffs);

		// Symmetric real-valued 4x4
		for(int img0 = 0 ; img0 < ncoeffs; img0++)
		{
			for(int img1 = img0 ; img1 < ncoeffs; img1++)
			{
				complex float * mat = stkern_mat + img0 * dim0 * dim1 + img1 * dim0 * dim1 * ncoeffs;
				complex float * mat2 = stkern_mat + img1 * dim0 * dim1 + img0 * dim0 * dim1 * ncoeffs;
				for(int i = 0 ; i < dim1 ; i++)
				{
					for(int j = 0 ; j < dim0 ; j++)
					{
						float realval = creal(mat[i + j * dim1]);
						mat[i + j * dim1] = realval;
						mat2[i + j * dim1] = realval;
					}
				}
			}
		}

		//  load_mat("sens30.mat", sens, dim0*dim1*2*ncoils*sizeof(complex float)); //
		// 14
		//  load_mat("stkern30.mat", stkern_mat, dim0*dim1*ncoeffs*ncoeffs*sizeof(complex
		// float)); // 16
		//  load_mat("src30.mat", src, dim0*dim1*2*ncoeffs*sizeof(complex float)); // 8
		//  load_mat("dst30.mat", dst_ref, dim0*dim1*2*ncoeffs*sizeof(complex float)); //
		// 8

#pragma omp barrier
		struct timeval start, end;
		gettimeofday(&start, NULL);
		for (int iter = 0; iter < 250; iter++) {
			jtmodel_normal_benchmark(sens, stkern_mat, plan, plan2, dst_ref, src, dim0,
					dim1, ncoils, ncoeffs, tst1);
		}
		gettimeofday(&end, NULL);
		double elapsed =
			(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
		printf("%f elapsed\n", elapsed);

#pragma omp barrier

		// Transpose 4x4 matrices and set to float
		for(int img = 0 ; img < ncoeffs*ncoeffs ; img++)
		{
			complex float * nontrans = stkern_mat + img * dim0 * dim1;
			float * trans = stkern_mat_trans + img * dim0 * dim1;
			for(int i = 0 ; i < dim1 ; i++)
			{
				for(int j = 0 ; j < dim0 ; j++)
				{
					trans[i + j * dim1] = creal(nontrans[j + i * dim0]);
				}
			}
		}


#pragma omp barrier

		gettimeofday(&start, NULL);
		for (int iter = 0; iter < 250; iter++) {
			jtmodel_normal_benchmark_fast(sens, stkern_mat_trans, dst, src, dim0,
					dim1, ncoils, ncoeffs, plan1d_0, plan1d_inv_0, plan1d_1, plan1d_inv_1, tst2,m,dim0);
		}
		gettimeofday(&end, NULL);
		elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
		printf("%f elapsed\n", elapsed);
#pragma omp barrier

		double diff = 0.0;
		double ref = 0.0;
		for(int map = 0 ; map < 2 ; map++)
			//for(int map = 0 ; map < ncoils ; map++)
		{
			for(int img = 0 ; img < ncoeffs ; img++)
			{
				complex float * _t1 = dst_ref + dim0*dim1*map + dim0*dim1*2*img;
				complex float * _t2 = dst + dim0*dim1*map + dim0*dim1*2*img;
				//complex float * _t1 = tst1+ dim0*dim1*map + dim0*dim1*ncoils*img;
				//complex float * _t2 = tst2 + dim0*dim1*map + dim0*dim1*ncoils*img;
				for(int i = 0 ; i < dim1 ; i++)
				{
					for(int j = 0 ; j < dim0 ; j++)
					{
						complex float t1 = _t1[j + i * dim0];
						complex float t2 = _t2[j + i * dim0];
						double _diff = (t1-t2)*(t1-t2);
						double _ref = (t1) * (t1);
						diff += _diff;
						ref += _ref;
						if (sqrt(_diff / _ref) > 0.1) {
							//printf("Error %lu\t%.10e\t%.10e\n", j + i*dim0 + dim0*dim1*map + dim0*dim1*2*img, t1, t2);
							//printf("Error %lu\t%.10e\t%.10e\n", j + i*dim0 + dim0*dim1*map + dim0*dim1*ncoils*img, t1, t2);
						}
					}
				}
			}
		}

		printf("Relative norm error: %.10e\n", sqrt(diff / ref));

		free(src);
		free(dst);
		free(sens);
		free(stkern_mat);
	}
}
#endif
