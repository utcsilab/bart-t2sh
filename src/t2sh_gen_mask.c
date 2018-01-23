/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2017-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/pd.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


#define m_pi M_PI
#define pi M_PI
#define SUCCESS 0
#define FAILURE -1

#define ACQ_FULL       0
#define ACQ_ELLIPSE    1
#define CAL_CROSS      0
#define CAL_BOX        1
#define CAL_ELLIPSE    2

#define VIEW_ORDER_R_ECLIPSE 9

bool dbg_t2sh = false;
bool t2sh_make_cal = true;
int t2sh_num_masks = 1;
bool t2sh_dup_e2s = false;
int t2sh_cal_size= 8;
float t2sh_partial_fourier = 1.;
int sampling_pattern = 3;
bool t2sh_flag = false;

#define MAX_SIZE 262144
float y_list[MAX_SIZE];
float z_list[MAX_SIZE];
float y_sort[MAX_SIZE];
float z_sort[MAX_SIZE];
float y_sort_dab[MAX_SIZE];
float z_sort_dab[MAX_SIZE];



static const char usage_str[] = "<outfile>";
static const char help_str[] = "Computes T2 Shuffling sampling pattern.";

typedef struct
{
	size_t sz;
	size_t nele;
	float complex *arr;
} VarCFloatComplexArray;

typedef struct
{
	size_t h;
	size_t arrSz;
	size_t *nele;
	size_t *sz;
	float complex **arr;
} VarCFloatComplexGrid;

typedef struct arraytosort{
	float a;
	float b;
	float c;
	float d;
}arraytosort;


static void copy_vector(
		int    n, /* I */
		float* a, /* I */
		float* b  /* O */
		)
{
	int i;
	if ((b!=NULL) && (a!=NULL))
	{
		for (i=0; i<n; i++)
		{
			b[i] = a[i];
		}
	}
}

/* Compare Function for sorting forward direction */
	static int
comparefun(const void *array1, const void *array2)
{
	return (((arraytosort *)array1)->a > ((arraytosort *)array2)->a )? 1: -1;
}
/* Compare Function for sorting reverse direction */
	static int
comparefun2(const void *array1, const void *array2)
{
	return (((arraytosort *)array1)->a < ((arraytosort *)array2)->a )? 1: -1;
}

static void sort_vectors(
		int  index_start, /*  I  */
		int  index_end,   /*  I  */
		int  dir,         /*  I  */  /* direction 1=forward, -1=reverse : 02-May-07 */
		float* a,         /* I/O */
		float* b,         /* I/O */
		float* c,         /* I/O */
		float* d          /* I/O */
		)
{
	/*HCSDM00170864 Implement the sorting algorithm to be a faster qsort
	 * approach. The input and output is not impacted. DLH Nov 2012*/ 
	int   i;
	arraytosort *temparray = NULL;
	int arraysize = index_end - index_start + 1;
	if (arraysize < 1)
	{
		return;
	}

	if (a == NULL)
	{
		return;
	}

	temparray = (arraytosort *) malloc(sizeof(arraytosort) * arraysize);
	if (temparray == NULL)
	{
		return;
	}

	for (i = index_start; i <= index_end; i++)
	{
		temparray[i-index_start].a = a[i];
		if (b!=NULL)
		{
			temparray[i-index_start].b = b[i];
		}
		else
		{
			temparray[i-index_start].b = 0;
		}
		if (c!=NULL)
		{
			temparray[i-index_start].c = c[i];
		}
		else
		{
			temparray[i-index_start].c = 0;
		}
		if (d!=NULL)
		{
			temparray[i-index_start].d = d[i];
		}
		else
		{
			temparray[i-index_start].d = 0;
		}
	}
	if (dir > 0)
	{
		qsort(&temparray[0], arraysize, sizeof(arraytosort),comparefun);
	}
	else
	{
		qsort(&temparray[0], arraysize, sizeof(arraytosort),comparefun2);
	}
	for (i = index_start; i <= index_end; i++)
	{
		a[i] = temparray[i-index_start].a;
		if (b!=NULL)
		{
			b[i] = temparray[i-index_start].b;
		}
		if (c!=NULL)
		{
			c[i] = temparray[i-index_start].c;
		}
		if (d!=NULL)
		{
			d[i] = temparray[i-index_start].d;
		}
	}
	free(temparray);
}

static float ran0(long *idum)
{
	long k;
	float ans;
	long mask = 123459876;
	long iq = 127773;
	long ia = 16807;
	long im = 2147483647;
	long ir = 2836;
	float am;

	am = 1.0 / im;
	*idum ^= mask;
	k = (*idum) / iq;
	*idum = ia*(*idum - k*iq) - ir*k;
	if (*idum < 0) *idum += im;
	ans = am*(*idum);
	*idum ^= mask;
	return ans;
}


static void freeListGrid2D(VarCFloatComplexGrid *p) 
{
	if (p->nele) {
		free(p->nele);
		p->nele = NULL;
	}
	if (p->sz) {
		free(p->sz);
		p->sz = NULL;
	}
	if (p->arr) {
		unsigned int i;
		for (i = 0; i < p->arrSz; i++) {
			if (p->arr[i]) {
				free(p->arr[i]);
				p->arr[i] = NULL;
			}
		}
		free(p->arr);  
		p->arr = NULL;
	}
}


/* Set up an 2D array to mimic the python grid structure.  Each cell
   has space for some moderate amount of points, and the append
   function will increase that later if needed.  A bad return only
   happens if memory can't be allocated */
static int initListGrid2D(VarCFloatComplexGrid *p, int w, int h) 
{
	unsigned int i;
	int initsz = 10;

	p->h = h;
	p->arrSz = w*h;
	/* These need to be initialized in case freeListGrid2D is called early */
	p->nele = NULL;
	p->sz = NULL;
	p->arr = NULL;
	/* The number of actual points in each cell */
	p->nele = (size_t*)malloc(p->arrSz*sizeof(*p->nele)); 
	if (!p->nele) {
		freeListGrid2D(p);
		return FAILURE;
	}
	/* The max number of points in each cell */
	p->sz = (size_t*)malloc(p->arrSz*sizeof(*p->sz)); 
	if (!p->sz) {
		freeListGrid2D(p);
		return FAILURE;
	}
	/* Space for the grid */
	p->arr = (float complex**)malloc(p->arrSz*sizeof(*p->arr));
	if (!p->arr) {
		freeListGrid2D(p);
		return FAILURE;
	}
	/* Also in case freeListGrid2D gets called */
	for (i = 0; i < p->arrSz; i++) p->arr[i] = NULL;
	for (i = 0; i < p->arrSz; i++) {
		/* Start each out with initsz spaces */
		p->arr[i] = (float complex*)malloc(initsz*sizeof(**p->arr));
		if (!p->arr[i]) {
			freeListGrid2D(p);
			return FAILURE;
		}
		p->sz[i] = initsz;
		p->nele[i] = 0;
	}
	return SUCCESS;
}

/* Put the point "pt" into the grid at spot (wc, hc).  Failure only
   occurs if memory allocation fails as bad point locations are just
   ignored */
static int appendListGrid2D(VarCFloatComplexGrid *p, int wc, int hc, float complex pt) 
{
	/* Cell index */
	int idx = wc*p->h + hc;
	/* If out of range just return */
	if (idx < 0 || (unsigned int)idx >= p->arrSz) return SUCCESS;
	/* If not enough space, resize first */
	if (p->nele[idx] + 1 > p->sz[idx]) {
		float complex *tmp = (float complex*)realloc(p->arr[idx], 2*p->sz[idx]*sizeof(**p->arr));
		/* Didn't work! */
		if (!tmp) return FAILURE;
		p->arr[idx] = tmp;
		p->sz[idx] *= 2;
	}

	p->arr[idx][p->nele[idx]] = pt;
	p->nele[idx] += 1;

	return SUCCESS;
}

/* Set up an array to mimic the python Random Queue construct */
static int initRandomQueue(VarCFloatComplexArray *p, int sz) 
{
	/* Allocate the array */
	p->sz = sz;
	p->nele = 0;
	p->arr = (float complex*)malloc(sz*sizeof(*p->arr));
	if (!p->arr) return FAILURE;

	return SUCCESS;
}

static int pushRandomQueue(VarCFloatComplexArray *p, float complex pt) 
{
	/* If not enough space, resize first */
	if (p->nele + 1 > p->sz) {
		float complex *tmp = (float complex*)realloc(p->arr, 2*p->sz*sizeof(*p->arr));
		/* Didn't work! */
		if (!tmp) {
			free(p->arr);
			p->arr = NULL;
			return FAILURE;
		}
		p->arr = tmp;
		p->sz *= 2;
	}
	p->arr[p->nele] = pt;
	p->nele += 1;

	return SUCCESS;
}

/* This only fails if there are no points in the queue */
static int popRandomQueue(VarCFloatComplexArray *p, float complex *pt, long *seed) 
{
	if (p->nele == 0) return FAILURE;
	else {
		int idx = (int) floor(p->nele*ran0(seed));
		/* Just in case */
		if ((unsigned int)idx == p->nele) idx--;
		*pt = p->arr[idx];
		p->arr[idx] = p->arr[p->nele - 1];
		p->nele--;
	}

	return SUCCESS;
}

static void freeRandomQueue(VarCFloatComplexArray *p) 
{
	if (p->arr) {
		free(p->arr);
		p->arr = NULL;
	}
}

/**
 * Generate a uniform random integer from the range {a, ..., b} (inclusive)
 */
static int urand(int a, int b)
{
	float n = rand() / (1.0 + RAND_MAX);
	int range = b - a + 1;
	int n2 = (n * range) + a;

	// occasionally returns an integer out of bounds!
	if ((n2 < a) || (n2 > b))
		n2 = urand(a,b);

	return n2;
}

/**
 * Randomly permute the intergers {0, ..., n-1} and store in perm
 */
static void randperm(int n, int perm[])
{
	int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;

	for(i=0; i<n; i++)
	{
		j = urand(i, n-1);
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}

/**
 * Re-sort the views based on the starting echo index and jitter power
 */
static void t2sh_jitter_views(int num_echoes, int num_trains, int skips_start, float jitter_power, int jitter_start, float* y_sort, float* z_sort)
{

	int local_jitter_start;
	int jitter_window;
	int idx;
	int idx_offset;
	int train;
	int echo;

	int* perm_idx = NULL;
	float* y_orig = NULL;
	float* z_orig = NULL;

	if(jitter_power > 0)
	{
		local_jitter_start = (num_echoes-skips_start-1 < jitter_start)? (num_echoes-skips_start-1):jitter_start;

		jitter_window = (int)((num_echoes-skips_start-local_jitter_start - 1) * jitter_power) + 1;

		if(jitter_window > 0)
		{
			perm_idx = (int*)malloc(jitter_window * sizeof(int));
			y_orig = (float*)malloc(jitter_window * sizeof(float));
			z_orig = (float*)malloc(jitter_window * sizeof(float));

			for (train = 0; train < num_trains; train++)
			{
				for (echo=skips_start+local_jitter_start; echo<=(num_echoes-jitter_window); ++echo)
				{
					idx_offset = train*num_echoes+echo;

					randperm(jitter_window, perm_idx);

					// do first pass to copy originals
					for(idx = 0; idx < jitter_window; ++idx)
					{
						y_orig[idx] = y_sort[idx_offset + idx];
						z_orig[idx] = z_sort[idx_offset + idx];
					}

					// do second pass to permute order
					for(idx = 0; idx < jitter_window; idx++)
					{
						y_sort[idx_offset + idx] = y_orig[perm_idx[idx]];
						z_sort[idx_offset + idx] = z_orig[perm_idx[idx]];
					}
				}
			}

			free(perm_idx);
			free(y_orig);
			free(z_orig);
		}
	}
}

#if 0
/**
 * Create a spatial mask from the views.
 */
static void vieworder2mask(int num_trains, int num_echoes, int yres, int zres, int* mask, const float* y_sort, const float* z_sort)
{
	UNUSED(yres);

	int i, yy, zz;

	for (i = 0; i < num_trains * num_echoes; i++)
	{
		yy = (int)y_sort[i];
		zz = (int)z_sort[i];

		if ( (-1 != yy) && (-1 != zz) )
		{
			mask[yy*zres + zz] = 1;
		}
	}
}

/**
 * Create view orders from a mask index
 */
static void mask2vieworder(int num_trains, int num_echoes, int skips_start, int yres, int zres, float* y_sort, float* z_sort, const int* mask_idx)
{
	int i, yy, zz, idx;

	int* train_counter = (int*)malloc(num_trains * sizeof(int));

	for (i = 0; i < num_trains; i++)
		train_counter[i] = skips_start;

	for (yy = 0; yy < yres; yy++)
	{
		for (zz = 0; zz < zres; zz++)
		{
			idx = mask_idx[yy*zres + zz];

			if (idx != -1)
			{
				y_sort[idx*num_echoes + train_counter[idx]] = (float)yy;
				z_sort[idx*num_echoes + train_counter[idx]] = (float)zz;
				train_counter[idx]++;
			}
		}
	}

	free(train_counter);
}
#endif


/**
 * Generate an array of all-ones of size num_points, and randomly
 * set some elements to zero until there are num_trains non-zero
 * points. num_trains >= num_points
 */
static void rperm_list(int* list, int num_trains, long num_points)
{
	int i = 0;
	int* rperm = (int*)malloc(num_points * sizeof(int));

	for (i = 0; i < num_points; i++)
		list[i] = 1;

	randperm(num_points, rperm);

	if (num_points > num_trains) {

		for (i = 0; i < num_points - num_trains; i++)
			list[rperm[i]] = 0;
	}

	free(rperm);
}



static int in_cal_region(int ky, int kz, int cal_size, int yres, int zres, float yfov, float zfov) {

	float res_y = 0.;
	float res_z = 0.;

	float rsy = 0.;
	float rsz = 0.;

	float r0 = 0.;

	float yy = 0.;
	float zz = 0.;

	/* based on method in genVDPoisson */
	res_y = yfov / (float)yres;
	res_z = zfov / (float)zres;
	rsy = 1.;
	rsz = 1.;


	if (res_y > res_z)
		rsy = res_z / res_y;
	else
		rsz = res_y / res_z;

	r0 = (yres > zres) ? (float)cal_size / (float)yres : (float)cal_size / (float)zres;


	yy = -1 + 2.0 * ky / (yres - 1);
	zz = (zres == 1) ? 0 : -1 + 2.0 * kz / (zres - 1);

	if (sqrt(rsy*rsy*yy*yy) < r0 && sqrt(rsz*rsz*zz*zz) < r0)
		return 1;
	else
		return 0;
}


static void crawl(int* _direction, int* _py, int* _pz, int zres, int* mask)
{
	/* direction:
	   0 == right
	   1 == up
	   2 == left
	   3 == down
	   */

	int direction = *_direction;
	int py = *_py;
	int pz = *_pz;

	/* moving right, position to the right not yet selected, and position above is selected */
	if ((direction == 0) && (mask[(pz + 1) * zres + py] == 0) && (mask[pz * zres + (py - 1)] > 0)) {

		/* keep moving right */
		pz += 1;
	}
	/* moving right, position to the right not yet selected, and position above is not selected */
	else if ((direction == 0) && (mask[(pz + 1) * zres + py] == 0) && (mask[pz * zres + (py - 1)] == 0)) {

		/* start moving up */
		direction = 1;
		py -= 1;
	}
	/* moving up, position above not yet selected, and position to the left is selected */
	else if ((direction == 1) && (mask[pz * zres + (py - 1)] == 0) && (mask[(pz - 1) * zres + py] > 0)) {

		/* keep moving up */
		py -= 1;
	}
	/* moving up, position above not yet selected, and position to the left is not selected */
	else if ((direction == 1) && (mask[pz * zres + (py - 1)] == 0) && (mask[(pz - 1) * zres + py] == 0)) {

		/* start moving left */
		direction = 2;
		pz -= 1;
	}
	/* moving left, position to the left not yet selected, and position below is selected */
	else if ((direction == 2) && (mask[(pz - 1) * zres + py] == 0) && (mask[pz * zres + (py + 1)] > 0)) {

		/* keep moving left */
		pz -= 1;
	}
	/* moving left, position to the left not yet selected, and position below is not selected */
	else if ((direction == 2) && (mask[(pz - 1) * zres + py] == 0) && (mask[pz * zres + (py + 1)] == 0)) {

		/* start moving down */
		direction = 3;
		py += 1;
	}
	/* moving down, position below not yet selected, and position to the right is selected */
	else if ((direction == 3) && (mask[pz * zres + (py + 1)] == 0) && (mask[(pz + 1) * zres + py] > 0)) {

		/* keep moving down */
		py += 1;
	}
	/* moving down, position below not yet selected, and position to the right is not selected */
	else if ((direction == 3) && (mask[pz * zres + (py + 1)] == 0) && (mask[(pz + 1) * zres + py] == 0)) {

		/* start moving right */
		direction = 0;
		pz += 1;
	}

	*_direction = direction;
	*_py = py;
	*_pz = pz;
}


static void crawl_assign_cal(int num_trains, int num_echoes, int train, int echo, int* direction, int* py, int* pz, int zres, int* mask, float* y_sort, float* z_sort)
{
	while (train < num_trains) {

		crawl(direction, py, pz, zres, mask);
		mask[*pz * zres + *py] = 1;
		y_sort[train*num_echoes + echo] = *py;
		z_sort[train*num_echoes + echo] = *pz;

		train++;
	}
}


/**
 * Make calibration region out of the first skips_start echoes in each of the num_trains echo trains.
 * mode == 0: create growing rings of calibration region at each echo. The overall cal region will be bigger
 * mode == 1: create a calibration region, and repeat it skips_start times.
 jtamir - 2016-07
 */
/* FIXME: problem when mode == 0, sometimes there are gaps between the echoes.
   Would probably be better to use crawl_assign for all of the first echo as well,
   but need to check against aspect ratio
   */
static int t2sh_cal_e2s(int mode, int num_trains, int skips_start, int num_echoes, int yres, int zres, float* y_sort, float* z_sort)
{
	int i = 0;
	int j = 0;

	/* for assignment */
	int train = 0;
	int echo = 0;

	/* center of k-space */
	float cy = 0.;
	float cz = 0.;

	/* to keep track of the outermost point in the cal region */
	int py = 0;
	int pz = 0;

	/* used to crawl around the cal region */
	int direction = 0;


	cy = yres / 2.;
	cz = zres / 2.;

	/* keep track of selected points */
	int* mask = (int*)malloc(yres * zres * sizeof(int));
	for (i = 0; i < yres*zres; i++)
		mask[i] = 0;

	/* first make a calibration region with num_trains points, and assign to the first echo */
	train = 0;
	bool flag = false;
	for (i = 0; i < yres; i++) {

		if (flag)
			break;

		for (j = 0; j < zres; j++) {

			if (!flag) {
				if ((abs(i - cy) <= sqrt(num_trains * (float)yres / (float)zres) / 2) &&
						(abs(j - cz) <= sqrt(num_trains * (float)zres / (float)yres) / 2)) {
					y_sort[train*num_echoes + 0] = i;
					z_sort[train*num_echoes + 0] = j;

					mask[j * zres + i] = 1;

					py = i;
					pz = j;

					train++;

					if (train >= num_trains)
						flag = true;
				}
			}
		}
	}

	if (dbg_t2sh)
		printf("[DBG_T2SH] (py, pz) = (%d, %d)\n", py, pz);

	/* some points may not be assigned, because the number of echo trains doesn't form a perfect rectangle */
	direction = 0;
	if (train < num_trains) {

		if (dbg_t2sh)
			printf("[DBG_T2SH] train = %d, num_trains = %d. Crawling for remaining %d trains\n", train, num_trains, num_trains - train);
		crawl_assign_cal(num_trains, num_echoes, train, 0, &direction, &py, &pz, zres, mask, y_sort, z_sort);
	}

	if (dbg_t2sh)
		printf("[DBG_T2SH] (py, pz) = (%d, %d)\n", py, pz);

	if (mode == 0) {

		/* if mode == 0,  continue to crawl around the cal region to make it bigger.
		   each of the other skipped echoes will be rectangular shells
		   that fit around each other to form a bigger cal region */

		for (echo = 1; echo < skips_start; echo++)
			crawl_assign_cal(num_trains, num_echoes, 0, echo, &direction, &py, &pz, zres, mask, y_sort, z_sort);
	}
	else if (mode == 1) {

		/* if mode == 1, repeat the first echo across the other skipped echoes */
		for (train = 0; train < num_trains; train++) {

			for (echo = 1; echo < skips_start; echo++) {

				y_sort[train*num_echoes + echo] = y_sort[train*num_echoes + 0];
				z_sort[train*num_echoes + echo] = z_sort[train*num_echoes + 0];
			}
		}
	}
	else
		return -1;


	free(mask);

	return 0;
}


static void epic_error(int a, char* str, int b, int c)
{
	UNUSED(a);
	UNUSED(b);
	UNUSED(c);
	error(str);
}


/* Generates a variable density resolution Poisson disc sampling
   pattern. From M. Lustig

Output: acqmask - sampling pattern array. The array size has to be
at least sky*skz, and is indexed by acqmask[i*skz + j], 
with 0 <= i < sky, 0 <= j < skz. 
The row/i index reference ky values, while the
column/j index reference kz values

Input: fovy - FOV in Y, in mm
fovz - FOV in Z, in mm
sky - number of ky values
skz - number of kz values, has to be > 1
ry - acceleration rate in Y with no calibration area
rz - acceleration rate in Z with no calibration area
ncal - size of the calibration area in ky and kz
cutcorners - remove the ky, kz corners
*/
static int genVDPoissonSampling(int *acqmask, float fovy, float fovz, int sky, 
		int skz, float ry, float rz, int ncal, 
		int cutcorners, long ran_seed, int my_sampling_pattern) 
{
	const char* myname = "genVDPoissonSampling:";
	char tmpstr[80];
	int i, j;
	int masksum;
	int cellDim = 2;
	int genpts = 0;
	int numneighpts = 30;
	int loopcnt, looplimit = 100;
	float rsy = 1;
	float rsz = 1;
	float mr_top = 40;
	float mr_bot = 0;
	float res_y, res_z;
	float r0;
	float tol = 0.05;
	float grid_max;
	float gridCsz;
	int grid_w, grid_h;
	float pW, pH;
	VarCFloatComplexGrid grid;
	VarCFloatComplexArray proclist;

	short *mask = NULL;
	short *automask = NULL;
	float *R = NULL;
	float complex *r_grid = NULL;

	/* jtamir: change ranSeed functionality 2014.10.02 */
	/* Unless specified, always make the same pattern for a given input */
	long ranSeed = ran_seed ? ran_seed : 1251046800;  /* Aug 23, 2009 10am */

	/* Checks */
	if (sky <= 1) {
		sprintf(tmpstr, "%s The number of ky values (%d) must be > 1!\n", 
				myname, sky);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}
	if (skz <= 0) {
		sprintf(tmpstr, "%s The number of kz values (%d) must be > 0!\n", 
				myname, skz);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}
	if (ry <= 0) {
		sprintf(tmpstr, "%s The reduction rate in Y (%f) must be > 0!\n", 
				myname, ry);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}
	if (rz <= 0) {
		sprintf(tmpstr, "%s The reduction rate in Z (%f) must be > 0!\n", 
				myname, rz);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}
	if (fovy <= 0) {
		sprintf(tmpstr, "%s The FOV in Y (%f) must be > 0!\n", myname, fovy);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}
	if (fovz <= 0) {
		sprintf(tmpstr, "%s The volume thickness (%f) must be > 0!\n", 
				myname, fovz);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}

	/* 2D check */
	if (skz == 1) rz = 1;

	/* Setup */
	r0 = (sky > skz) ? (float)ncal/(float)sky : (float)ncal/(float)skz;
	/* Very unlikely, but protect against divide by 0 errors below */
	if (r0 == 1) {
		if (sky == ncal)
			sprintf(tmpstr, "%s The number of in-plane PE's must be > %d\n",
					myname, ncal);
		if (skz == ncal)
			sprintf(tmpstr, "%s The number of slice PE's must be > %d\n",
					myname, ncal);
		epic_error(0, tmpstr, 0, 0);
		return FAILURE;
	}

	/* Allocate arrays */
	mask = (short*)malloc(sky*skz*sizeof(*mask));
	automask = (short*)malloc(sky*skz*sizeof(*automask));
	R = (float*)malloc(sky*skz*sizeof(*R));
	r_grid = (float complex*)malloc(sky*skz*sizeof(*r_grid));
	if (mask == NULL || automask == NULL || R == NULL || r_grid == NULL) {
		sprintf(tmpstr, "%s Array allocation failed!\n", myname);
		epic_error(0, tmpstr, 0, 0);
		if (mask) free(mask);
		if (automask) free(automask);
		if (R) free(R);
		if (r_grid) free(r_grid);
		return FAILURE;
	}

	/* Clear out the acquisition array */
	for (i = 0; i < sky*skz; i++) acqmask[i] = 0;

	/* Adjust the scaling for the R matrix */
	res_y = fovy/sky;
	res_z = fovz/skz;
	if (res_y > res_z) rsy = res_z/res_y;
	else               rsz = res_y/res_z;
	/* Calculate the masks and the R array */
	masksum = 0;
	for (i = 0; i < sky; i++) {
		float y = -1 + 2.0*i/(sky - 1);
		for (j = 0; j < skz; j++) {
			float z = (skz == 1) ? 0 : -1 + 2.0*j/(skz - 1);
			/* Autocalibration lines */
			automask[i*skz + j] = 
				(sqrt(rsy*rsy*y*y) < r0 && sqrt(rsz*rsz*z*z) < r0) ? 1 : 0;
			/* Mask */
			if (cutcorners) mask[i*skz + j] = (sqrt(y*y + z*z) <= 1) ? 1 : 0;
			else            mask[i*skz + j] = 1;
			masksum += mask[i*skz + j];
			/* R */
			R[i*skz + j] = sqrt(rsy*rsy*y*y + rsz*rsz*z*z);
		}
	}

	/* Calculate the parameters that give the desired acceleration
	   numerically using bisection */
	loopcnt = 0;
	while (1) {
		float est_accl = 0;
		float dsum = 0;
		float mr = mr_bot/2.0 + mr_top/2.0;
		float rgridR, rgridI;
		grid_max = 0;
		for (i = 0; i < sky; i++) {
			for (j = 0; j < skz; j++) {
				int idx = i*skz + j;

				/*peng for user-defined variable density degree 08.05.11*/
				/*float rrval = ((R[i*skz + j] - r0)*(mr - 1.0))/(1.0 - r0);*/
				float rrval;

				if (my_sampling_pattern==10)
					rrval = ((R[i*skz + j] - r0)*(mr - 1.0))/(1.0 - r0);
				else if (R[i*skz + j] >= r0)
					rrval = (1.0 - 1.0/sqrt(my_sampling_pattern))/(1.0-r0) * mr * (R[i*skz + j] - 1.0) + mr - 1;
				else
					rrval = 0;
				/*peng ^^^^^^^^^^*/

				if (rrval < 0) rrval = 0;
				rgridR = rrval + 1;
				rgridI = rrval*rz/ry + 1;
				/* Save the value for the sampling density array */
				r_grid[idx] = rgridR + I*rgridI;
				dsum += mask[i*skz + j]/(rgridR*rgridI);
				/* Keep track of the max/min grid values */
				if (rgridR > grid_max) grid_max = rgridR;
				if (rgridI > grid_max) grid_max = rgridI;
			}
		}
		est_accl = 1.24*1.24*masksum/dsum;
		/* For very small reduction factors the estimate is only so good */
		if (++loopcnt > looplimit) break;
		/* All done */
		if (fabs(est_accl - ry*rz) < tol) break;
		/* Not done, so try different parameters */
		if (est_accl < ry*rz) mr_bot = mr;
		else                  mr_top = mr;
	}

	/* The "sample_poisson_ellipse" section of Miki's script */
	gridCsz = grid_max/sqrt(2.0);
	grid_w = (int) ceil(sky/gridCsz);
	grid_h = (int) ceil(skz/gridCsz);

	/* Initialize the grid */
	if (initListGrid2D(&grid, grid_w, grid_h) == FAILURE) {
		sprintf(tmpstr, "%s Can't allocate memory for ListGrid2D!\n", myname);
		epic_error(0, tmpstr, 0, 0);
		if (mask) free(mask);
		if (automask) free(automask);
		if (R) free(R);
		if (r_grid) free(r_grid);
		return FAILURE;
	}
	/* Initialize the random queue array */
	if (initRandomQueue(&proclist, sky*skz) == FAILURE) {
		sprintf(tmpstr, "%s Can't allocate memory for Random Queue!\n", myname);
		epic_error(0, tmpstr, 0, 0);
		freeListGrid2D(&grid);
		if (mask) free(mask);
		if (automask) free(automask);
		if (R) free(R);
		if (r_grid) free(r_grid);
		return FAILURE;
	}
	/* Generate the first point */
	pW = sky*ran0(&ranSeed);
	pH = (skz == 1) ? 0 : skz*ran0(&ranSeed);
	/* Put the point in the queue */
	if (pushRandomQueue(&proclist, pW + I*pH) == FAILURE) {
		sprintf(tmpstr, "%s Can't allocate memory in RQ push!\n", myname);
		epic_error(0, tmpstr, 0, 0);
		freeListGrid2D(&grid);
		freeRandomQueue(&proclist);
		if (mask) free(mask);
		if (automask) free(automask);
		if (R) free(R);
		if (r_grid) free(r_grid);
		return FAILURE;
	}
	/* Put the point in the grid */
	if (appendListGrid2D(&grid, (int) (pW/gridCsz), 
				(int) (pH/gridCsz), pW + I*pH) == FAILURE) {
		sprintf(tmpstr, "%s ListGrid2D append failed!\n", myname);
		epic_error(0, tmpstr, 0, 0);
		freeListGrid2D(&grid);
		freeRandomQueue(&proclist);
		if (mask) free(mask);
		if (automask) free(automask);
		if (R) free(R);
		if (r_grid) free(r_grid);
		return FAILURE;
	}
	/* And use it */
	acqmask[(int) pW *skz + (int) pH] = mask[(int) pW *skz + (int) pH];

	/* Generate other points from points in the queue */
	do {
		float rW, rH;
		float complex p;
		/* Get a point. This fails only if there are no more points */
		if (popRandomQueue(&proclist, &p, &ranSeed) == FAILURE) break;
		pW = creal(p);
		pH = cimag(p);
		rW = creal(r_grid[(int) pW*skz + (int) pH]);
		rH = cimag(r_grid[(int) pW*skz + (int) pH]);

		/* Generate a number of points around points already in the
		   sample, and then check if they are too close to other
		   points. Typically numneighpts = 30 is sufficient. The larger
		   numneighpts the slower the algorithm, but the more sample
		   points are produced */
		for (i = 0; i < numneighpts; i++) {
			int cW, cH;
			/* Generate a point randomly selected around p, between r and
			   2*r units away.  Note, r >= 1 (or should be!) */
			float ratio = rH/rW;
			float rr = rW*(1 + ran0(&ranSeed));
			float rt = 2*m_pi*ran0(&ranSeed);
			/* The generated point q */
			float qW = rr*sin(rt)*ratio + pW;
			float qH = (skz == 1) ? 0 : rr*cos(rt) + pH;
			int in_neighborhood = 0;
			/* If inside the rectangle continue */
			if (0 <= qW && qW < sky && 0 <= qH && qH < skz) {
				int cell;
				unsigned int l;
				/* This is the "in_neighbourhood(q, r)" condition */
				for (cW = -cellDim; cW <= cellDim; cW++) {
					int cellidxW = (int) (qW/gridCsz) + cW;
					if (cellidxW < 0 || cellidxW >= grid_w) continue;
					for (cH = -cellDim; cH <= cellDim; cH++) {
						int cellidxH = (int) (qH/gridCsz) + cH;
						if (cellidxH < 0 || cellidxH >= grid_h) continue;
						/* Cell index in the grid array */
						cell = cellidxW*grid_h + cellidxH;
						/* Number of points in the cell */
						for (l = 0; l < grid.nele[cell]; l++) {
							float cptW = creal(grid.arr[cell][l]);
							float cptH = cimag(grid.arr[cell][l]);
							/* To keep the point q, the following has to be false
							   for all points l in all cells */
							if ((qW - cptW)*(qW - cptW) + 
									(qH - cptH)*(qH - cptH)/(ratio*ratio) <= rW*rW) {
								in_neighborhood = 1;
								break;
							}
						}
						if (in_neighborhood) break;
					}
					if (in_neighborhood) break;
				}
				/* Add the point */
				if (!in_neighborhood) {
					/* Put the point in the queue */
					if (pushRandomQueue(&proclist, qW + I*qH) == FAILURE) {
						sprintf(tmpstr, "%s Can't allocate memory in RQ push!\n", myname);
						epic_error(0, tmpstr, 0, 0);
						freeListGrid2D(&grid);
						freeRandomQueue(&proclist);
						if (mask) free(mask);
						if (automask) free(automask);
						if (R) free(R);
						if (r_grid) free(r_grid);
						return FAILURE;
					}
					/* Put the point in the grid */
					if (appendListGrid2D(&grid, (int) (qW/gridCsz), 
								(int) (qH/gridCsz), qW + I*qH) == FAILURE) {
						sprintf(tmpstr, "%s ListGrid2D append failed!\n", myname);
						epic_error(0, tmpstr, 0, 0);
						freeListGrid2D(&grid);
						freeRandomQueue(&proclist);
						if (mask) free(mask);
						if (automask) free(automask);
						if (R) free(R);
						if (r_grid) free(r_grid);
						return FAILURE;
					}
					acqmask[(int) qW*skz + (int) qH] = mask[(int) qW*skz + (int) qH];
				}
			}
		}
	} while (proclist.nele);

	/* Add the autocalibration lines and count all the points.  genpts
	   isn't actually used, but it's easy to throw in here */
	genpts = 0;
	for (i = 0; i < sky; i++) {
		for (j = 0; j < skz; j++) {
			int idx = i*skz + j;
			if (automask[idx] && !acqmask[idx]) acqmask[idx] = 1;
			genpts += acqmask[idx];
		}
	}

	/*
	   sprintf(tmpstr, "%s generated %d points\n", myname, genpts);
	   fputs(tmpstr, stderr); fflush(stderr);
	   sprintf(tmpstr, "%s actual/desired acceleration is: %f/%f\n", 
	   myname, masksum/(float) genpts, ry*rz);
	   fputs(tmpstr, stderr); fflush(stderr);
	   */

	freeListGrid2D(&grid);
	freeRandomQueue(&proclist);
	if (mask) free(mask);
	if (automask) free(automask);
	if (R) free(R);
	if (r_grid) free(r_grid);

	return SUCCESS;
}


/**
 * Regenerate VDPoisson mask until there are at least num_trains points in the mask
 */
static int regenVDPoissonSampling(int num_trains, int *csmask, float yfov, float zfov, int yres,
		int zres, float ry, float rz, int ncal,
		int cut_corners, long ran_seed, int my_sampling_pattern, float my_partial_fourier)
{
	int err = 0;
	long i = 0;
	long j = 0;
	long num_points = 0;

	ry *= sqrt(my_partial_fourier);
	rz *= sqrt(my_partial_fourier);

	while (num_points < num_trains) {

		err = genVDPoissonSampling(csmask, yfov, zfov, yres, zres, ry, rz, ncal, cut_corners, ran_seed, my_sampling_pattern);

		if (err == -1)
			epic_error(0, "jitter:genVDPoissonSampling failed!\n", 0, 0);

		// zero out partial fourier
		if (my_partial_fourier < 1.) {
			for (i = 0; i < yres; i++) {
				for (j = (int)(zres * my_partial_fourier); j < zres; j++) {
					csmask[i*zres + j] = 0;
				}
			}
		}

		num_points = 0;
		for (i = 0; i < yres*zres; i++)
			num_points += csmask[i];

		ran_seed *= 2;
	}

	return num_points;

}


#if 0
static bool select_views(
    int    yres,           /* I */
    int    zres,           /* I */
    int    ystride,        /* I */
    int    zstride,        /* I */
    int    ycal,           /* I */
    int    zcal,           /* I */
    int    yover,          /* I */  
    int    zover,          /* I */  
    int    acq_shape,      /* I */
    int    cal_shape,      /* I */
    float* y_list,         /* O */
    float* z_list,         /* O */
    int*   num_views,      /* O */
    int*   tmparc_syn_pts, /* O */
    float* R               /* O */
    )
{
    int   y, z;
    int   yend, zend;  /* changed from start to end : 02-May-07 */ 
    float ycenter = yres/2 - 0.5;  /* changed from "+" to "-" because indexing starts at 0 now : 03-May-07 */
    float zcenter = zres/2 - 0.5;  /* changed from "+" to "-" because indexing starts at 0 now : 03-May-07 */
    int   views_within_mask = 0;
    int   view_count        = 0;
    int   within_mask;
    int   on_y_grid, on_z_grid, on_grid;
    int   within_y_cal, within_z_cal, within_cal;
    int   acquire_view;
    int   tmp_arc_pts = 0; /*tmp var to hold # of pts to be synthesized by ARC */ 

    /* -1 because indexing starts at 0 now : 03-May-07 */
    if (yover==0)  /* full ky */
        yend = yres - 1;
    else           /* partial ky */
        yend = yres/2 + yover - 1;

    if (zover==0)  /* full kz */
        zend = zres - 1;
    else           /* partial kz */
        zend = zres/2 + zover - 1;
    
    /* indexing starts at 0 now : 03-May-07 */
    for (z=0; z<=zend; z++) {      
        for (y=0; y<=yend; y++) {

            /* is view within mask? */
            if (ACQ_FULL == acq_shape)
                within_mask = 1;
            else
                within_mask = pow( (y-ycenter)/(yres/2), 2) + pow( (z-zcenter)/(zres/2), 2) <= 1;

            /* count views within mask */
            if (within_mask) {
                views_within_mask = views_within_mask + 1;
            }

            /* is view on accelerated grid? */
            on_y_grid = y % ystride==0;
            on_z_grid = z % zstride==0;
            on_grid   = on_y_grid && on_z_grid;

            /* is view within cal? */
            within_y_cal = fabs(y-ycenter)<(ycal/2);
            within_z_cal = fabs(z-zcenter)<(zcal/2);
            if (CAL_BOX == cal_shape)      
                within_cal = within_y_cal && within_z_cal;
            else if (CAL_ELLIPSE == cal_shape) 
                within_cal = pow( (y-ycenter)/(ycal/2), 2) + pow( (z-zcenter)/(zcal/2), 2) <= 1;

            /* determine if view is acquired */
            if (CAL_CROSS == cal_shape) { 
                acquire_view = within_mask && (on_y_grid || within_y_cal) && (on_z_grid || within_z_cal);
            } else if (CAL_BOX == cal_shape || CAL_ELLIPSE == cal_shape){            
                acquire_view = within_mask && (on_grid || within_cal);
            }

            if (within_mask && !acquire_view)
            {
                tmp_arc_pts++;
            }

            /* debug */
            if (0) printf("view=%d y=%d z=%d acquire=%d within_mask=%d on_y_grid=%d within_y_cal=%d on_z_grid=%d within_z_cal=%d \n", 
                             view_count, y, z, acquire_view, within_mask, on_y_grid, within_y_cal, on_z_grid, within_z_cal);

            /* count and assign views that are acquired */
            if (acquire_view) {
                y_list[view_count] = y;
                z_list[view_count] = z;

                /* debug */
                if (0) printf("view=%d y=%0.0f z=%0.0f\n", view_count, y_list[view_count], z_list[view_count]);

                view_count = view_count + 1;
            }

        }  /* for y */
    }  /* for z */

    *R = (float)views_within_mask / view_count;
    *num_views = view_count;
    *tmparc_syn_pts = tmp_arc_pts;

    return SUCCESS;
}
#endif



static void orderviews_t2sh(
		int num_trains,
		int num_echoes,
		int skips_start,
		int yres,
		int zres,
		int cal_size,
		int my_sampling_pattern,
		int* mask,
		long ran_seed
		)
{

	UNUSED(skips_start);


	int* csmask = NULL;
	int* list = NULL;
	float yfov = 0.;
	float zfov = 0.;
	float fudge = 1.;
	float accel = 1.;
	int cut_corners = 1;

	long num_points = 0;
	long count = 0;
	long count2 = 0;
	long ii = 0;
	long jj = 0;

	long cc = 0;
	long cc_true = 0;

	int* rlist = NULL;

	float my_partial_fourier = t2sh_partial_fourier;

	/* approx accel to get num_trains in each mask */
	fudge = 1.1;
#if 1
	accel = sqrtf( (float)yres * (float)zres * m_pi / (4 * fudge * num_trains * num_echoes) );

	float y_accel = accel;
	float z_accel = accel;

	if (dbg_t2sh)
		printf("[DBG_T2SH] vdpoisson scheme. accel = %f\n", accel);
#else
	float y_accel = (float)zres * sqrtf(m_pi) / (2 * sqrtf(fudge * num_trains * num_echoes)); 
	float z_accel = (float)yres * sqrtf(m_pi) / (2 * sqrtf(fudge * num_trains * num_echoes));
	accel = sqrtf(y_accel * z_accel);

	if (dbg_t2sh) {

		printf("[DBG_T2SH] vdpoisson scheme. accel = %f, yaccel=%f, zaccel=%f\n", accel, y_accel, z_accel);
	}
#endif

	csmask = (int*)malloc( yres*zres * sizeof(int) );
	yfov = 300.;
	zfov = 300.;
	cut_corners = 1;


	/* generate poisson mask until we have at least num_trains * num_echoes points */
	num_points = regenVDPoissonSampling(num_trains * num_echoes, csmask, yfov, zfov, yres, zres, y_accel, z_accel, cal_size,
			cut_corners, ran_seed++, my_sampling_pattern, my_partial_fourier);

	if (dbg_t2sh)
		printf("[DBG_T2SH] num_points = %d\n", (int)num_points);

	/* randomly prune the mask until it has exactly num_trains points */
	list = (int*)malloc(num_points * sizeof(int));
	rperm_list(list, num_trains * num_echoes, num_points);

	rlist = (int*)malloc(yres * zres * sizeof(int));
	randperm(yres * zres, rlist);



	if (dbg_t2sh) {
		printf("[DBG_T2SH] yres = %d\tyres = %d\n", yres, yres);
		printf("[DBG_T2SH] zres = %d\tzres = %d\n", zres, zres);
	}

	count = 0;
	count2 = 0;

	/* first pass: fill calibration region */
	for (ii = 0; ii < yres; ii++) {

		for (jj = 0; jj < zres; jj++) {

			if (in_cal_region(ii, jj, cal_size, yres, zres, yfov, zfov)) {

				mask[ii*zres + jj] = 1;
				count++;

				csmask[ii*zres + jj] = 0;
			}

			if (count == num_trains * num_echoes) {
				//printf("count == num_trains * num_echoes == %d\n", num_trains * num_echoes);
				break;
			}

		}
	}


	/* second pass: fill remainder until num_points is reached */
	for (cc = 0; cc < yres * zres; cc++) {

		cc_true = rlist[cc];

		ii = cc_true / zres;
		jj = cc_true - ii * zres;


		if (csmask[ii*zres + jj] == 1) {

			if (list[count2] != 0) {

				mask[ii*zres + jj] = 1;
				count++;
			}
			count2++;
		}

		if (count == num_trains * num_echoes) {
			//printf("count == num_trains * num_echoes == %d\n", num_trains * num_echoes);
			break;
		}

	}


	free(rlist);

	free(csmask);
	free(list);
}


static void orderviews_genmask(
		int num_trains,
		int num_echoes,
		int skips_start,
		int yres,
		int zres,
		int cal_size,
		int my_sampling_pattern,
		float* y_sort,
		float* z_sort,
		long ran_seed
		)
{

	int* mask = NULL;
	int i = 0;
	int ii = 0;
	int jj = 0;
	int c = 0;

	mask = (int*)malloc(yres * zres * sizeof(int));

	for (i = 0; i < yres * zres; i++)
		mask[i] = 0;

	orderviews_t2sh(num_trains, num_echoes, skips_start, yres, zres, cal_size, my_sampling_pattern, mask, ran_seed);

	for (i = 0; i < yres * zres; i++) {

		ii = i / zres;
		jj = i - ii * zres;

		if (1 == mask[i]) {

			y_sort[c] = ii;
			z_sort[c] = jj;
			c++;
		}
	}

	debug_printf(DP_DEBUG1, "c = %d, num_trains = %d, num_echoes = %d\n", c, num_trains, num_echoes);
	assert(c == num_trains * num_echoes);

	free(mask);
}


static void swap(float* y_sort, float* z_sort, int idx1, int idx2)
{
	float y = 0.;
	float z = 0.;

	y = y_sort[idx1];
	z = z_sort[idx1];

	y_sort[idx1] = y_sort[idx2];
	z_sort[idx1] = z_sort[idx2];

	y_sort[idx2] = y;
	z_sort[idx2] = z;
}


static float distsq(float y1, float z1, float y2, float z2)
{
	return pow(y1 - y2, 2) + pow(z1 - z2, 2);
}


static void orderviews_knn(
		int num_trains,
		int num_echoes,
		int skips_start,
		float* y_sort,
		float* z_sort
		)
{

	float y = 0.;
	float z = 0.;

	float y2 = 0.;
	float z2 = 0.;

	float y_new = 0.;
	float z_new = 0.;

	int train = 0;
	int train2 = 0;
	int min_train = 0;
	int echo = 0;

	float d = 0.;
	float d_min = 0.;

	/* need to sort the phase encodes from the first echo by distance from center. this way, we 
	   prioritize assigning echoes in the center of k-space first.
	   If echoes2skip>0, we could make the assumption that the first echo 
	   is close to the center for all echo trains, so that this naturally happens...
	   */


	for (train = 0; train < num_trains; train++) {

		y = y_sort[train*num_echoes + skips_start];
		z = z_sort[train*num_echoes + skips_start];

		if (y != -1 && z != -1) {

			for (echo = skips_start + 1; echo < num_echoes; echo++) {

				min_train = train;

				y2 = y_sort[train*num_echoes + echo];
				z2 = z_sort[train*num_echoes + echo];

				if (y2 != -1 && z2 != -1) {

					d_min = distsq(y, z, y2, z2);

					for (train2 = train; train2 < num_trains; train2++) {

						y2 = y_sort[train2*num_echoes + echo];
						z2 = z_sort[train2*num_echoes + echo];

						if (y2 != -1 && z2 != -1) {

							d = distsq(y, z, y2, z2);

							if (d < d_min) {

								y_new = y2;
								z_new = z2;
								min_train = train2;
								d_min = d;
							}
						}

					}

					swap(y_sort, z_sort, train*num_echoes + echo, min_train*num_echoes + echo);

					/* move to next echo */
					y = y_new;
					z = z_new;
				}
			}
		}

	}
}

/*peng, jtamir^^^^^^^^^^*/

static void sort_radius(
		int    num_views,       /* I */
		int    yres,            /* I */
		int    zres,            /* I */
		int    echo_dir,        /* I */  
		int    encode_flag,     /* I */ /* k-space encode mode. Cube variable TE MM */
		float  axis_ratio,      /* I */ /* Long to short axis ratio for eclipse ordering. Cube variable TE MM */
		float* y_list,          /* I */
		float* z_list          /* I */
		)
{
	int   list_index = 0;
	float y = 0.0;
	float z = 0.0;
	float r = 0.0;
	float r_max = 0.0;
	float beta = 0.0;
	float theta_loc = 0.0;
	float ycenter = yres/2 - 0.5; /* changed "+" to "-" because indexing starts at 0 now : 03-May-07 */
	float zcenter = zres/2 - 0.5; /* changed "+" to "-" because indexing starts at 0 now : 03-May-07 */

	float* s_list;
	float* r_list;

	/* Set the start point at the k-space edge. Cube variable TE MM */
	if (VIEW_ORDER_R_ECLIPSE == encode_flag) { ycenter = 0.5; }

	s_list      = (float*)malloc(num_views * sizeof(float));
	r_list      = (float*)malloc(num_views * sizeof(float));


	/*** STEP 1: Calculate RADIUS and THETA ***/

	for (list_index=0; list_index<num_views; list_index++)
	{

		y = y_list[list_index];
		z = z_list[list_index];

		/* calculate radius  */
		/* Added long to short axis ratio */ /* Cube variable TE MM */
		float yoff = (float)y-ycenter;
		float zoff = ((float)z-zcenter)*axis_ratio*(float)yres/zres;
		r = sqrt(yoff*yoff + zoff*zoff);
		r_list[list_index] = r;
		if (r > r_max) r_max = r;

		/* calculate theta */  
		/* Added long to short axis ratio */ /* Cube variable TE MM */
		beta = atan( fabs((float)z-zcenter) * axis_ratio / fabs((float)y-ycenter) );

		if      ( (y > ycenter) && (z > zcenter) ) /* first quadrant:   -pi   ->  -pi/2 */
			theta_loc = -pi + beta;
		else if ( (y < ycenter) && (z > zcenter) ) /* second quadrant:  -pi/2 ->  0     */
			theta_loc = - beta;
		else if ( (y < ycenter) && (z < zcenter) ) /* third quadrant:   0     -> pi/2   */
			theta_loc = beta;
		else if ( (y > ycenter) && (z < zcenter) ) /* fourth quadrant:  pi/2  ->  pi    */
			theta_loc = pi - beta;

		/* Flip theta for 180 degree. Theta should be between -90 and +90 degree. */ /* Cube variable TE MM */
		if (VIEW_ORDER_R_ECLIPSE == encode_flag) theta_loc -= pi;
		if (theta_loc < -pi) theta_loc += 2*pi;

		s_list[list_index] = theta_loc;   
	}


	/*** STEP 2: Sort by R ***/

	sort_vectors(0, num_views-1, echo_dir, r_list, y_list, z_list, s_list);  /* passed echo_dir argument : RFB 26-Oct-07 */ 

	free(s_list);
	free(r_list);
}

/* updated -- r then theta sort : RFB 17-Jul-07 */
/* removed unused yover & lope_fraction args; added echo_dir arg : RFB 26-Oct-07 */
/* Added encode_flag and axis_ratio: Cube variable TE MM */
static void orderviews_r(
		int    num_views,       /* I */
		int    num_trains,      /* I */
		int    num_echoes,      /* I */
		int    skips_start,     /* I */
		int    yres,            /* I */
		int    zres,            /* I */
		int    echo_dir,        /* I */  
		int    encode_flag,     /* I */ /* k-space encode mode. Cube variable TE MM */
		float  axis_ratio,      /* I */ /* Long to short axis ratio for eclipse ordering. Cube variable TE MM */
		int    my_t2sh_flag,
		float* y_list,          /* I */
		float* z_list,          /* I */
		float* y_sort_dab,      /* O */
		float* z_sort_dab,      /* O */
		float* y_sort,          /* O */
		float* z_sort           /* O */
		)
{
	int   list_index;
	int   sort_index;
	int   i, n, train, echo, skips_end_total;
	int   index_start, index_end;
	int   views_to_sort;
	float r_max = 0.0; /* Cube BB MM */

	/*peng, jtamir^^^^^^^^^^*/

	float y = 0.0;
	float z = 0.0;
	float r = 0.0;
	float beta = 0.0;
	float theta_loc = 0.0;
	float r_start = 0.0;
	float r_step = 0.0;
	float r_end = 0.0;
	float ycenter = yres/2 - 0.5; /* changed "+" to "-" because indexing starts at 0 now : 03-May-07 */
	float zcenter = zres/2 - 0.5; /* changed "+" to "-" because indexing starts at 0 now : 03-May-07 */

	float* s_list;
	float* r_list;
	float* s_sort;
	float* r_sort;
	float* y_sort_temp;
	float* z_sort_temp;
	float* s_sort_temp;
	float* r_sort_temp;
	float* y_temp;
	float* z_temp;
	float* r_temp;
	float* s_temp;
	int*   i_temp;
	int*   views_in_train;

	/* Set the start point at the k-space edge. Cube variable TE MM */
	if (VIEW_ORDER_R_ECLIPSE == encode_flag) { ycenter = 0.5; }

	s_list      = (float*)malloc(num_views * sizeof(float));
	r_list      = (float*)malloc(num_views * sizeof(float));
	s_sort      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	r_sort      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	y_sort_temp = (float*)malloc(num_trains*num_echoes * sizeof(float));
	z_sort_temp = (float*)malloc(num_trains*num_echoes * sizeof(float));
	s_sort_temp = (float*)malloc(num_trains*num_echoes * sizeof(float));
	r_sort_temp = (float*)malloc(num_trains*num_echoes * sizeof(float));
	y_temp      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	z_temp      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	r_temp      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	s_temp      = (float*)malloc(num_trains*num_echoes * sizeof(float));
	i_temp      = (int*)malloc(num_trains*num_echoes * sizeof(int));
	views_in_train = (int*)malloc(num_trains * sizeof(int));  /* bug fix: num_echoes -> num_trains : RFB 30-Jul-07 */


	/*** STEP 1: Calculate RADIUS and THETA ***/

	for (list_index=0; list_index<num_views; list_index++)
	{

		y = y_list[list_index];
		z = z_list[list_index];

		/* calculate radius  */
		/* Added long to short axis ratio */ /* Cube variable TE MM */
		float yoff = (float)y-ycenter;
		float zoff = ((float)z-zcenter)*axis_ratio*(float)yres/zres;
		r = sqrt(yoff*yoff + zoff*zoff);
		r_list[list_index] = r;
		if (r > r_max) r_max = r;

		/* calculate theta */  
		/* Added long to short axis ratio */ /* Cube variable TE MM */
		beta = atan( fabs((float)z-zcenter) * axis_ratio / fabs((float)y-ycenter) );

		if      ( (y > ycenter) && (z > zcenter) ) /* first quadrant:   -pi   ->  -pi/2 */
			theta_loc = -pi + beta;
		else if ( (y < ycenter) && (z > zcenter) ) /* second quadrant:  -pi/2 ->  0     */
			theta_loc = - beta;
		else if ( (y < ycenter) && (z < zcenter) ) /* third quadrant:   0     -> pi/2   */
			theta_loc = beta;
		else if ( (y > ycenter) && (z < zcenter) ) /* fourth quadrant:  pi/2  ->  pi    */
			theta_loc = pi - beta;

		/* Flip theta for 180 degree. Theta should be between -90 and +90 degree. */ /* Cube variable TE MM */
		if (VIEW_ORDER_R_ECLIPSE == encode_flag) theta_loc -= pi;
		if (theta_loc < -pi) theta_loc += 2*pi;

		s_list[list_index] = theta_loc;   
	}


	/*** STEP 2: Sort by R ***/

	sort_vectors(0, num_views-1, echo_dir, r_list, y_list, z_list, s_list);  /* passed echo_dir argument : RFB 26-Oct-07 */ 


	/*** STEP 3: Assign to Trains ***/

	/* initialize */
	for (long view=0; view<num_trains*num_echoes; view++)
	{
		if (y_sort_dab != NULL && z_sort_dab != NULL) {
			y_sort_dab[view] = -1;
			z_sort_dab[view] = -1;
		}
		y_sort[view] = -1;
		z_sort[view] = -1;
		r_sort[view] = -1;
		s_sort[view] = -1;
		y_sort_temp[view] = -1;
		z_sort_temp[view] = -1;
		r_sort_temp[view] = -1;
		s_sort_temp[view] = -1;
	}

	for (train=0; train<num_trains; train++)
	{
		views_in_train[train] = 0;
	}

	/* for each echo, sort by theta and distribute to trains */
	for (echo=skips_start; echo<num_echoes; echo++)
	{
		index_start     = (echo-skips_start  )*num_trains;
		index_end       = (echo-skips_start+1)*num_trains-1;
		views_to_sort   = index_end-index_start+1;
		skips_end_total = 0;

		/* last echo may be skipped in some trains */
		if (index_end > num_views-1)
		{
			skips_end_total = index_end - (num_views-1);
			views_to_sort   = views_to_sort - skips_end_total;
			index_end       = (num_views-1);
		}

		copy_vector(views_to_sort, &y_list[index_start], y_temp);
		copy_vector(views_to_sort, &z_list[index_start], z_temp);
		copy_vector(views_to_sort, &r_list[index_start], r_temp);
		copy_vector(views_to_sort, &s_list[index_start], s_temp);
		sort_vectors(0, views_to_sort-1, 1, s_temp, y_temp, z_temp, r_temp);

		for (i=0; i<views_to_sort; i++)
		{
			train = (int)(i*(float)num_trains/views_to_sort);
			y_sort[train*num_echoes + echo] = y_temp[i];
			z_sort[train*num_echoes + echo] = z_temp[i];
			r_sort[train*num_echoes + echo] = r_temp[i];
			s_sort[train*num_echoes + echo] = s_temp[i];
			views_in_train[train] = views_in_train[train] + 1;
		}
	}


	/*** STEP 4: Sort by THETA for each R ring (to minimize step size) ***/

	if (VIEW_ORDER_R_ECLIPSE == encode_flag) { /* Cube variable TE MM */
		r_step = 1.0;
		r_end = r_max;
	} else {
		r_step = 4; 
		r_end  = yres/2;
	}

	for (r_start = 0; r_start < r_step * axis_ratio * 2; r_start += r_step)
	{
		for (r=r_start; r<=r_end; r+=r_step) 
		{
			/* pull out points in R ring */ 
			n = 0; 
			for (sort_index=0; sort_index<(num_trains*num_echoes); sort_index++)
			{
				if (r_sort[sort_index]>=r && r_sort[sort_index]<(r+r_step) && y_sort[sort_index]>=0)
				{
					y_temp[n]  = y_sort[sort_index];
					z_temp[n]  = z_sort[sort_index];
					r_temp[n]  = r_sort[sort_index];
					s_temp[n]  = s_sort[sort_index];
					i_temp[n]  = sort_index;
					n = n+1;
				}
			}

			/* sort by THETA */
			if (n>1)
			{
				sort_vectors(0, n-1, 1, s_temp, r_temp, z_temp, y_temp);
			}

			/* rebuild sorted list */
			if (n>0) 
			{
				for (i=0; i<n; i++)
				{
					y_sort_temp[i_temp[i]] = y_temp[i];
					z_sort_temp[i_temp[i]] = z_temp[i];
					r_sort_temp[i_temp[i]] = r_temp[i];
					s_sort_temp[i_temp[i]] = s_temp[i];
				}
			}

		} /* for r */

		copy_vector(num_trains*num_echoes, y_sort_temp, y_sort);
		copy_vector(num_trains*num_echoes, z_sort_temp, z_sort);
		copy_vector(num_trains*num_echoes, r_sort_temp, r_sort);
		copy_vector(num_trains*num_echoes, s_sort_temp, s_sort);

	} /* for r_start */


	/*** STEP 5: Sort by R within trains ***/

	for (train=0; train<num_trains; train++)
	{
		sort_vectors(train*num_echoes+skips_start, (train+1)*num_echoes-2, echo_dir, r_sort, s_sort, y_sort, z_sort); /* echo_dir : RFB 27-Oct-07 */
	}

	/* WTC store ky-kz for dab storage */
	if (y_sort_dab != NULL && z_sort_dab != NULL) {
		copy_vector(num_trains*num_echoes, y_sort, y_sort_dab);
		copy_vector(num_trains*num_echoes, z_sort, z_sort_dab);
	}

	/* step 6 (Dither to implement lope_fraction) -- unused->ELIMINATED : RFB 29-Oct-07 */

	/*peng to introduce jittering along echoes for kr sampling for UCBerkeley 2013.11.03*/


	/* jitter */
	if (my_t2sh_flag)
		t2sh_jitter_views(num_echoes, num_trains, skips_start, 1., 0, y_sort, z_sort);

	if (y_sort_dab != NULL && z_sort_dab != NULL) {
		copy_vector(num_trains*num_echoes, y_sort, y_sort_dab);
		copy_vector(num_trains*num_echoes, z_sort, z_sort_dab);
	}
	/*peng ^^^^^^^^^^*/

	free(s_list);
	free(r_list);
	free(s_sort);
	free(r_sort);
	free(y_sort_temp);
	free(z_sort_temp);
	free(r_sort_temp);
	free(s_sort_temp);
	free(y_temp);
	free(z_temp);
	free(r_temp);
	free(s_temp);
	free(i_temp);
	free(views_in_train);
}

static void orderviews_r_helper(
		int    num_views,       /* I */
		int    num_trains,      /* I */
		int    num_echoes,      /* I */
		int    skips_start,     /* I */
		int    yres,            /* I */
		int    zres,            /* I */
		int    echo_dir,        /* I */  
		int    encode_flag,     /* I */ /* k-space encode mode. Cube variable TE MM */
		float  axis_ratio,      /* I */ /* Long to short axis ratio for eclipse ordering. Cube variable TE MM */
		float* y_list,          /* I */
		float* z_list,          /* I */
		float* y_sort_dab,      /* O */
		float* z_sort_dab,      /* O */
		float* y_sort,          /* O */
		float* z_sort,          /* O */
		long ran_seed           /* I */
		)
{
	int i = 0;
	int q = 0;
	int count = 0;
	int train = 0;
	int echo = 0;

	int num_echoes_dup = 0;
	int num_echoes_gen = 0;
	int num_echoes_adj = num_echoes - skips_start; /* adjust number of echoes by skips_start */
	int remaining_echoes = num_echoes_adj;

	float* y_list_dup;
	float* z_list_dup;

	float* y_sort_dup;
	float* z_sort_dup;

	float* y_sort_dab_dup;
	float* z_sort_dab_dup;

	long sran_seed = 1251046800;
	printf("sran seed disabled! %d\n", (int)sran_seed);

	/*
	   srand(sran_seed);
	   */


	if (dbg_t2sh) {
		printf("[DBG_T2SH] t2sh_num_masks = %d\n", t2sh_num_masks);
		printf("[DBG_T2SH] skips_start = %d\n", skips_start);
	}

	if (t2sh_num_masks > 1) {

		if (dbg_t2sh)
			printf("[DBG_T2SH] I'm in t2sh_num_masks\n");

		/* ran_seed = 1251046800; -- replace with function parameter -- jtamir 072016 */


		/* initialize */
		for (train = 0; train < num_trains; train++) {

			for (echo = 0; echo < num_echoes; echo++) {

				y_sort[train*num_echoes + echo] = y_list[train*num_echoes + echo];
				z_sort[train*num_echoes + echo] = z_list[train*num_echoes + echo];

				y_sort_dab[train*num_echoes + echo] = y_list[train*num_echoes + echo];
				z_sort_dab[train*num_echoes + echo] = z_list[train*num_echoes + echo];

			}
		}


		if (t2sh_make_cal) {

			/* make our own skipped echoes calibration region */
			/* FIXME: check for failure */
			t2sh_cal_e2s(t2sh_dup_e2s, num_trains, skips_start, num_echoes, yres, zres, y_sort, z_sort);

		}
		else {

			/* copy skipped echoes from center-out order */
			for (train = 0; train < num_trains; train++) {

				for (echo = 1; echo < skips_start; echo++) {

					if (t2sh_dup_e2s) {

						// duplicate the first echo for each skipped echo
						y_sort[train*num_echoes + echo] = y_sort[train*num_echoes + 0];
						z_sort[train*num_echoes + echo] = z_sort[train*num_echoes + 0];
					} else {

						// use center-out for the skipped echoes
						y_sort[train*num_echoes + echo] = y_sort[train*num_echoes + echo];
						z_sort[train*num_echoes + echo] = z_sort[train*num_echoes + echo];
					}
				}
			}
		}

		/* round up to nearest integer */
		num_echoes_dup = (num_echoes_adj + t2sh_num_masks - 1) / t2sh_num_masks;

		y_list_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));
		z_list_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));

		y_sort_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));
		z_sort_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));

		y_sort_dab_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));
		z_sort_dab_dup = (float*)malloc( num_trains * num_echoes_dup * sizeof(float));

		for (i = 0; i < t2sh_num_masks; i++) {

			num_echoes_gen = remaining_echoes - num_echoes_dup > 0 ? num_echoes_dup : remaining_echoes;
			remaining_echoes -= num_echoes_gen;

			if (dbg_t2sh)
				printf("[DBG_T2SH] num_echoes = %d, num_echoes_adj = %d, num_echoes_dup = %d, num_echoes_gen = %d, remaining_echoes = %d\n", num_echoes, num_echoes_adj, num_echoes_dup, num_echoes_gen, remaining_echoes);

			if (num_echoes_gen > 0) {
				for (q = 0; q < num_trains * num_echoes_dup; q++) {
					y_sort_dup[q] = -1;
					z_sort_dup[q] = -1;
				}


				orderviews_genmask(num_trains, num_echoes_gen, 0, yres, zres, t2sh_cal_size, sampling_pattern, y_sort_dup, z_sort_dup, ran_seed++);

				if (!t2sh_flag) {
					if (dbg_t2sh)
						printf("[DBG_T2SH] keep a center-out ordering\n");

					for (q = 0; q < num_trains * num_echoes_dup; q++) {

						y_list_dup[q] = y_sort_dup[q];
						z_list_dup[q] = z_sort_dup[q];

						y_sort_dab_dup[q] = y_sort_dup[q];
						z_sort_dab_dup[q] = z_sort_dup[q];
					}

					orderviews_r(num_trains * num_echoes_gen, num_trains, num_echoes_gen, 0, yres, zres, echo_dir, encode_flag, axis_ratio, 0, y_list_dup, z_list_dup, y_sort_dab_dup, z_sort_dab_dup, y_sort_dup, z_sort_dup);

				}

				count = 0;

				debug_printf(DP_DEBUG1, "skips start is %d\n", skips_start);

				for (train = 0; train < num_trains; train++) {

					for (echo = 0; echo < num_echoes_gen; echo++) {

						if (  y_sort[ train*num_echoes + (i * num_echoes_dup) + skips_start + echo ] != -1 &&
								z_sort[ train*num_echoes + (i * num_echoes_dup) + skips_start + echo ] != -1 ) {

							y_sort[ train*num_echoes + (i * num_echoes_dup) + skips_start + echo ] = y_sort_dup[ count ];
							z_sort[ train*num_echoes + (i * num_echoes_dup) + skips_start + echo ] = z_sort_dup[ count ];
						}
						count++;
					}


				}
			}
		}

		free(y_list_dup);
		free(z_list_dup);

		free(y_sort_dup);
		free(z_sort_dup);

		free(y_sort_dab_dup);
		free(z_sort_dab_dup);

		if (t2sh_flag) {

			if (dbg_t2sh)
				printf("[DBG_T2SH] knn sort\n");

			sort_radius(num_echoes, yres, zres, echo_dir, encode_flag, axis_ratio, y_sort, z_sort);
			orderviews_knn(num_trains, num_echoes, 0, y_sort, z_sort);
		}
	}
	else
		orderviews_r(num_views, num_trains, num_echoes, skips_start, yres, zres, echo_dir, encode_flag, axis_ratio, t2sh_flag, y_list, z_list, y_sort_dab, z_sort_dab, y_sort, z_sort);

}


int main_t2sh_gen_mask(int argc, char* argv[])
{
	int ran_seed = 125104680;
	unsigned int num_trains = 100;
	unsigned int num_echoes = 40;

	unsigned int skips_start = 0;
	unsigned int yres = 256;
	unsigned int zres = 256;
	int echo_dir = 1;
	int encode_flag = 8;
	float axis_ratio = 1.;

	bool fully_sampled = false;

	const struct opt_s opts[] = {

		OPT_UINT('Y', &yres, "size", "size dimension 1"),
		OPT_UINT('Z', &zres, "size", "size dimension 2"),
		OPT_SET('v', &dbg_t2sh, "verbose"),
		OPT_INT('d', &t2sh_num_masks, "num", "number of sampling masks"),
		OPT_SET('e', &t2sh_dup_e2s, "duplicate the first echo for each skipped echo"),
		OPT_INT('C', &t2sh_cal_size, "size", "calibration region size for each pattern"),
		OPT_FLOAT('f', &t2sh_partial_fourier, "frac", "partial fourier fraction"),
		OPT_INT('V', &sampling_pattern, "num", "variable density level"),
		OPT_SET('m', &t2sh_flag, "turn on t2 shuffling randomized sampling"),
		OPT_INT('r', &ran_seed, "seed", "random seed"),
		OPT_UINT('s', &skips_start, "E2S", "echoes to skip"),
		OPT_UINT('T', &num_trains, "trains", "number of echo trains"),
		OPT_UINT('E', &num_echoes, "ETL", "echo train length"),
		OPT_SET('F', &fully_sampled, "fully sampled -- overrides num_trains"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	num_rand_init(ran_seed);

	if (fully_sampled) {

		num_trains = (yres * zres * M_PI / 4.) / num_echoes;
		debug_printf(DP_DEBUG1, "fully sampled. num_trains = %d\n", num_trains);
	}

	long num_views = num_trains * num_echoes;
	debug_printf(DP_DEBUG1, "num_views=%d ny*nz=%d\n", num_views, yres*zres);

	assert(num_views < (yres * zres * M_PI / 4.));

	for (long i = 0; i < MAX_SIZE; i++) {
		y_list[i] = -1;
		z_list[i] = -1;
		y_sort[i] = -1;
		z_sort[i] = -1;
		y_sort_dab[i] = -1;
		z_sort_dab[i] = -1;
	}

	float ycenter = yres / 2 - 0.5;
	float zcenter = zres / 2 - 0.5;
	long c = 0;
	for (long ky = 0; ky < yres; ky++) {

		for (long kz = 0; kz < zres; kz++) {

			bool within_mask = pow((ky - ycenter) / (yres / 2), 2) + pow((kz - zcenter) / (zres / 2), 2) <= 1;
			if (c < num_views && within_mask) {
				y_list[c] = ky;
				z_list[c] = kz;
				c++;
			}
		}
	}

	orderviews_r_helper(num_views, num_trains, num_echoes, skips_start,
			yres, zres, echo_dir, encode_flag, axis_ratio,
			y_list, z_list, y_sort_dab, z_sort_dab, y_sort, z_sort, ran_seed);

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	dims[1] = yres;
	dims[2] = zres;
	dims[5] = num_echoes;

	complex float* out = create_cfl(argv[1], DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, 1);

	long pos[DIMS];
	md_singleton_strides(DIMS, pos);

	for (long train = 0; train < num_trains; train++) {

		for (long echo = 0; echo < num_echoes; echo++) {

			long ky = y_sort[train * num_echoes + echo];
			long kz = z_sort[train * num_echoes + echo];

			pos[1] = ky;
			pos[2] = kz;
			pos[5] = echo;

			if (-1 != ky && -1 != kz) {
				long off = md_calc_offset(DIMS, strs, pos);
				out[off] = 1.;
			}
		}
	}

	unmap_cfl(DIMS, dims, out);

	exit(0);
}



