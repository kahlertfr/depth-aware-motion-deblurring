/* match.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include <time.h>
#include "match.h"

/************************************************************/
/************************************************************/
/************************************************************/

Match::Match(char *name_left, char *name_right, bool color)
{
	Coord p;

	if (!color)
	{
		im_color_left = im_color_right = NULL;
		im_color_left_min = im_color_right_min = NULL;
		im_color_left_max = im_color_right_max = NULL;

		im_left = (GrayImage) imLoad(IMAGE_GRAY, name_left);
		if (!im_left) { fprintf(stderr, "Can't load %s\n", name_left); exit(1); }
		im_right = (GrayImage) imLoad(IMAGE_GRAY, name_right);
		if (!im_right) { fprintf(stderr, "Can't load %s\n", name_right); exit(1); }

		im_size.x = imGetXSize(im_left); im_size.y = imGetYSize(im_left);

		if ( im_size.x != imGetXSize(im_right) || im_size.y != imGetYSize(im_right) )
		{
			fprintf(stderr, "Image sizes are different!\n");
			exit(1);
		}

		im_left_min = im_left_max = im_right_min = im_right_max = NULL;
	}
	else
	{
		im_left = im_right = NULL;
		im_left_min = im_right_min = NULL;
		im_left_max = im_right_max = NULL;

		im_color_left = (RGBImage) imLoad(IMAGE_RGB, name_left);
		if (!im_color_left) { fprintf(stderr, "Can't load %s\n", name_left); exit(1); }
		im_color_right = (RGBImage) imLoad(IMAGE_RGB, name_right);
		if (!im_color_right) { fprintf(stderr, "Can't load %s\n", name_right); exit(1); }

		im_size.x = imGetXSize(im_color_left); im_size.y = imGetYSize(im_color_left);

		if ( im_size.x != imGetXSize(im_color_right) || im_size.y != imGetYSize(im_color_right) )
		{
			fprintf(stderr, "Image sizes are different!\n");
			exit(1);
		}

		im_color_left_min = im_color_left_max = im_color_right_min = im_color_right_max = NULL;
	}

	disp_base = Coord(0, 0); disp_max = Coord(0, 0); disp_size = Coord(1, 1);

	x_left  = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
	y_left  = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
	x_right = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
	y_right = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
	if (!x_left || !y_left || !x_right || !y_right)
	{ fprintf(stderr, "Not enough memory!\n"); exit(1); }
	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		IMREF(x_left, p) = IMREF(x_right, p) = OCCLUDED;
	}
	unique_flag = true;

	ptr_im1 = (PtrImage) imNew(IMAGE_PTR, im_size.x, im_size.y);
	ptr_im2 = (PtrImage) imNew(IMAGE_PTR, im_size.x, im_size.y);
	if (!ptr_im1 || !ptr_im2)
	{ fprintf(stderr, "Not enough memory!\n"); exit(1); }

	if (im_left)
		printf("Gray images %s and %s of size %d x %d loaded\n\n", name_left, name_right, im_size.x, im_size.y);
	else
		printf("Color images %s and %s of size %d x %d loaded\n\n", name_left, name_right, im_size.x, im_size.y);

	segm_left = im_left;
	segm_right = im_right;
	segm_color_left = im_color_left;
	segm_color_right = im_color_right;
}

Match::~Match()
{
	if (segm_left && segm_left != im_left) imFree(segm_left);
	if (segm_right && segm_right != im_right) imFree(segm_right);
	if (segm_color_left && segm_color_left != im_color_left) imFree(segm_color_left);
	if (segm_color_right && segm_color_right != im_color_right) imFree(segm_color_right);

	if (im_left) imFree(im_left);
	if (im_right) imFree(im_right);
	if (im_color_left) imFree(im_color_left);
	if (im_color_right) imFree(im_color_right);

	if (im_left_min) imFree(im_left_min);
	if (im_left_max) imFree(im_left_max);
	if (im_right_min) imFree(im_right_min);
	if (im_right_max) imFree(im_right_max);
	if (im_color_left_min) imFree(im_color_left_min);
	if (im_color_left_max) imFree(im_color_left_max);
	if (im_color_right_min) imFree(im_color_right_min);
	if (im_color_right_max) imFree(im_color_right_max);

	imFree(x_left);
	imFree(y_left);
	imFree(x_right);
	imFree(y_right);

	imFree(ptr_im1);
	imFree(ptr_im2);
}

void Match::LoadSegm(char *name_left, char *name_right, bool color)
{
	if (!color)
	{
		segm_color_left = segm_color_right = NULL;

		segm_left = (GrayImage) imLoad(IMAGE_GRAY, name_left);
		if (!segm_left) { fprintf(stderr, "Can't load %s\n", name_left); exit(1); }
		segm_right = (GrayImage) imLoad(IMAGE_GRAY, name_right);
		if (!segm_right) { fprintf(stderr, "Can't load %s\n", name_right); exit(1); }

		if ( im_size.x != imGetXSize(segm_left)  || im_size.y != imGetYSize(segm_left) ||
		     im_size.x != imGetXSize(segm_right) || im_size.y != imGetYSize(segm_right) )
		{
			fprintf(stderr, "Segmentation and image sizes are different!\n");
			exit(1);
		}

		printf("Gray segmentation images %s and %s loaded\n\n", name_left, name_right);
	}
	else
	{
		segm_left = segm_right = NULL;

		segm_color_left = (RGBImage) imLoad(IMAGE_RGB, name_left);
		if (!segm_color_left) { fprintf(stderr, "Can't load %s\n", name_left); exit(1); }
		segm_color_right = (RGBImage) imLoad(IMAGE_RGB, name_right);
		if (!segm_color_right) { fprintf(stderr, "Can't load %s\n", name_right); exit(1); }

		if ( im_size.x != imGetXSize(segm_color_left)  || im_size.y != imGetYSize(segm_color_left) ||
		     im_size.x != imGetXSize(segm_color_right) || im_size.y != imGetYSize(segm_color_right) )
		{
			fprintf(stderr, "Segmentation and image sizes are different!\n");
			exit(1);
		}

		printf("Color segmentation images %s and %s loaded\n\n", name_left, name_right);
	}
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::SaveXLeft(char *file_name, bool flag)
{
	Coord p;
	GrayImage im = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);

	printf("Saving left x-disparity map as %s\n\n", file_name);

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d = IMREF(x_left, p), c;
		if (d==OCCLUDED) IMREF(im, p) = 255;
		else
		{
			if (flag) c = d - disp_base.x;
			else      c = disp_max.x - d;
			IMREF(im, p) = c;
		}
	}

	imSave(im, file_name);
	imFree(im);
}

void Match::SaveYLeft(char *file_name, bool flag)
{
	Coord p;
	GrayImage im = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);

	printf("Saving left y-disparity map as %s\n\n", file_name);

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d = IMREF(x_left, p), c;
		if (d==OCCLUDED) IMREF(im, p) = 255;
		else
		{
			d = IMREF(y_left, p);
			if (flag) c = disp_max.y - d;
			else      c = d - disp_base.y;
			IMREF(im, p) = c;
		}
	}

	imSave(im, file_name);
	imFree(im);
}

void Match::SaveScaledXLeft(char *file_name, bool flag)
{
	Coord p;
	RGBImage im = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);

	printf("Saving scaled left x-disparity map as %s\n\n", file_name);

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d = IMREF(x_left, p), c;
		if (d==OCCLUDED) { IMREF(im, p).r = 255; IMREF(im, p).g = IMREF(im, p).b = 0; }
		else
		{
			if (disp_size.x == 0) c = 255;
			else if (flag) c = 255 - (255-64)*(disp_max.x - d)/disp_size.x;
			else           c = 255 - (255-64)*(d - disp_base.x)/disp_size.x;
			IMREF(im, p).r = IMREF(im, p).g = IMREF(im, p).b = c;
		}
	}

	imSave(im, file_name);
	imFree(im);
}

void Match::SaveScaledYLeft(char *file_name, bool flag)
{
	Coord p;
	RGBImage im = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);

	printf("Saving scaled left y-disparity map as %s\n\n", file_name);

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d = IMREF(x_left, p), c;
		if (d==OCCLUDED) { IMREF(im, p).r = 255; IMREF(im, p).g = IMREF(im, p).b = 0; }
		else
		{
			d = IMREF(y_left, p);
			if (disp_size.y == 0) c = 255;
			else if (flag) c = 255 - (255-64)*(disp_max.y - d)/disp_size.y;
			else           c = 255 - (255-64)*(d - disp_base.y)/disp_size.y;
			IMREF(im, p).r = IMREF(im, p).g = IMREF(im, p).b = c;
		}
	}

	imSave(im, file_name);
	imFree(im);
}

void Match::LoadXLeft(char *file_name, bool flag)
{
	Coord p;
	GrayImage im = (GrayImage) imLoad(IMAGE_GRAY, file_name);

	if (!im) { fprintf(stderr, "Can't load %s\n", file_name); exit(1); }
	if ( im_size.x != imGetXSize(im) || im_size.y != imGetYSize(im) )
	{
		fprintf(stderr, "Size of the disparity map in %s is different!\n", file_name);
		exit(1);
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d, c = IMREF(im, p);

		if (c>=0 && c<disp_size.x)
		{
			if (flag) d = c + disp_base.x;
			else      d = disp_max.x - c;
			IMREF(x_left, p) = d;
			IMREF(y_left, p) = 0;
		}
		else IMREF(x_left, p) = IMREF(y_left, p) = OCCLUDED;
	}

	printf("Left x-disparity map from %s loaded\n\n", file_name);
	imFree(im);
	unique_flag = false;
}

void Match::LoadYLeft(char *file_name, bool flag)
{
	Coord p;
	GrayImage im = (GrayImage) imLoad(IMAGE_GRAY, file_name);

	if (!im) { fprintf(stderr, "Can't load %s\n", file_name); exit(1); }
	if ( im_size.x != imGetXSize(im) || im_size.y != imGetYSize(im) )
	{
		fprintf(stderr, "Size of the disparity map in %s is different!\n", file_name);
		exit(1);
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		int d, c = IMREF(im, p);

		if (c>=0 && c<disp_size.y)
		{
			if (flag) d = c + disp_base.y;
			else      d = disp_max.y - c;
			IMREF(y_left, p) = d;
		}
		else IMREF(x_left, p) = IMREF(y_left, p) = OCCLUDED;
	}

	printf("Left y-disparity map from %s loaded\n\n", file_name);
	imFree(im);
	unique_flag = false;
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::SetDispRange(Coord _disp_base, Coord _disp_max)
{
	disp_base = _disp_base;
	disp_max = _disp_max;
	disp_size = disp_max - disp_base + Coord(1, 1);
	if (! (disp_base <= disp_max) ) { fprintf(stderr, "Error: wrong disparity range!\n"); exit(1); }
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::CLEAR()
{
	Coord p;

	printf("CLEAR\n\n");

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		IMREF(x_left, p) = IMREF(x_right, p) = OCCLUDED;
	}
	
}

void Match::SWAP_IMAGES()
{
	Coord c_tmp;
	GrayImage g_tmp;
	RGBImage r_tmp;
	LongImage l_tmp;

	printf("SWAP_IMAGES\n\n");

	c_tmp = disp_base;
	disp_base = -disp_max;
	disp_max = -c_tmp;

	if (im_left)
	{
		g_tmp = im_left;
		im_left = im_right;
		im_right = g_tmp;

		g_tmp = im_left_min;
		im_left_min = im_right_min;
		im_right_min = g_tmp;

		g_tmp = im_left_max;
		im_left_max = im_right_max;
		im_right_max = g_tmp;
	}
	else
	{
		r_tmp = im_color_left;
		im_color_left = im_color_right;
		im_color_right = r_tmp;

		r_tmp = im_color_left_min;
		im_color_left_min = im_color_right_min;
		im_color_right_min = r_tmp;

		r_tmp = im_color_left_max;
		im_color_left_max = im_color_right_max;
		im_color_right_max = r_tmp;
	}

	l_tmp = x_left;
	x_left = x_right;
	x_right = l_tmp;

	l_tmp = y_left;
	y_left = y_right;
	y_right = l_tmp;
}

void Match::MAKE_UNIQUE()
{
	Coord p, d, pd, d2;

	printf("MAKE_UNIQUE\n\n");

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		IMREF(x_right, p) = OCCLUDED;
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d.x = IMREF(x_left, p); if (d.x == OCCLUDED) continue;
		d.y = IMREF(y_left, p);

		pd = p + d;
		if (pd>=Coord(0,0) && pd<im_size)
		{
			d2 = Coord(IMREF(x_right, pd), IMREF(y_right, pd));

			if (d2.x != OCCLUDED)
			{
				IMREF(x_left, pd+d2) = OCCLUDED;
			}

			IMREF(x_right, pd) = -d.x;
			IMREF(y_right, pd) = -d.y;
		}
		else IMREF(x_left, p) = OCCLUDED;
	}

	unique_flag = true;
}

void Match::CROSS_CHECK()
{
	Coord p, d, pd, d2;

	printf("CROSS_CHECK\n\n");

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d.x = IMREF(x_left, p); if (d.x == OCCLUDED) continue;
		d.y = IMREF(y_left, p);

		pd = p + d;
		if (pd>=Coord(0,0) && pd<im_size)
		{
			d2 = Coord(IMREF(x_right, pd), IMREF(y_right, pd));
			if (-d == d2) continue;
		}
		IMREF(x_left, p) = OCCLUDED;
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d.x = IMREF(x_right, p); if (d.x == OCCLUDED) continue;
		d.y = IMREF(y_right, p);

		pd = p + d;
		if (pd>=Coord(0,0) && pd<im_size)
		{
			d2 = Coord(IMREF(x_left, pd), IMREF(y_left, pd));
			if (-d == d2) continue;
		}
		IMREF(x_right, p) = OCCLUDED;
	}

	unique_flag = true;
}

void Match::FILL_OCCLUSIONS()
{
	Coord p, q, d;

	printf("FILL_OCCLUSIONS\n\n");

	for (p.y=0; p.y<im_size.y; p.y++)
	{
		for (p.x=0; p.x<im_size.x; p.x++)
		{
			d.x = IMREF(x_left, p);
			if (d.x != OCCLUDED) { d.y = IMREF(y_left, p); break; }
		}
		if (p.x == im_size.x) continue;
		for (q.x=0, q.y=p.y; q.x<p.x; q.x++)
		{
			IMREF(x_left, q) = d.x;
			IMREF(y_left, q) = d.y;
		}

		for (; p.x<im_size.x; p.x++)
		{
			if (IMREF(x_left, p) == OCCLUDED)
			{
				IMREF(x_left, p) = d.x;
				IMREF(y_left, p) = d.y;
			}
			else
			{
				d.x = IMREF(x_left, p);
				d.y = IMREF(y_left, p);
			}
		}
	}

	unique_flag = false;
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::KZ1()
{
	Coord p;

	if ( params.K < 0 ||
	     params.I_threshold < 0 ||
	     params.lambda1 < 0 ||
	     params.lambda2 < 0 ||
	     params.denominator < 1 )
	{
		fprintf(stderr, "Error in KZ1: wrong parameter!\n");
		exit(1);
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		if (IMREF(x_left, p) == OCCLUDED) IMREF(y_left, p) = OCCLUDED;
		if (IMREF(x_right, p) == OCCLUDED) IMREF(y_right, p) = OCCLUDED;
	}

	/* printing parameters */
	if (params.denominator == 1)
	{
		printf("KZ1:  K = %d\n", params.K);
		printf("      I_threshold = %d, lambda1 = %d, lambda2 = %d\n",
			params.I_threshold, params.lambda1, params.lambda2);
	}
	else
	{
		printf("KZ1:  K = %d/%d\n", params.K, params.denominator);
		printf("      I_threshold = %d, lambda1 = %d/%d, lambda2 = %d/%d\n",
			params.I_threshold, params.lambda1, params.denominator,
			                    params.lambda2, params.denominator);
	}
	printf("      sub_pixel = %s, data_cost = L%d\n",
		params.sub_pixel ? "true" : "false", params.data_cost==Parameters::L1 ? 1 : 2);

	if (disp_base.y==disp_max.y && disp_max.x<=0)
		KZ1_visibility = true;
	else
	{
		KZ1_visibility = false;
		printf("Visibility constraint is not enforced! (not a stereo case)\n");
	}

	Run_KZ_BVZ(METHOD_KZ1);

	unique_flag = false;
}

void Match::KZ2()
{
	if ( params.K < 0 ||
	     params.I_threshold2 < 0 ||
	     params.lambda1 < 0 ||
	     params.lambda2 < 0 ||
	     params.denominator < 1 )
	{
		fprintf(stderr, "Error in KZ2: wrong parameter!\n");
		exit(1);
	}
	if (!unique_flag) MAKE_UNIQUE();

	/* printing parameters */
	if (params.denominator == 1)
	{
		printf("KZ2:  K = %d\n", params.K);
		printf("      I_threshold2 = %d, lambda1 = %d, lambda2 = %d\n",
			params.I_threshold2, params.lambda1, params.lambda2);
	}
	else
	{
		printf("KZ2:  K = %d/%d\n", params.K, params.denominator);
		printf("      I_threshold2 = %d, lambda1 = %d/%d, lambda2 = %d/%d\n",
			params.I_threshold2, params.lambda1, params.denominator,
			                    params.lambda2, params.denominator);
	}
	printf("      sub_pixel = %s, data_cost = L%d\n",
		params.sub_pixel ? "true" : "false", params.data_cost==Parameters::L1 ? 1 : 2);

	Run_KZ_BVZ(METHOD_KZ2);
}

void Match::BVZ()
{
	Coord p;

	if ( params.occlusion_penalty < 0 ||
	     params.I_threshold < 0 ||
	     params.lambda1 < 0 ||
	     params.lambda2 < 0 ||
	     params.denominator < 1 )
	{
		fprintf(stderr, "Error in BVZ: wrong parameter!\n");
		exit(1);
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		if (IMREF(x_left, p) == OCCLUDED) IMREF(y_left, p) = OCCLUDED;
	}

	/* printing parameters */
	printf("BVZ:  occlusion_penalty = %d\n", params.occlusion_penalty);
	if (params.denominator == 1)
	{
		printf("      I_threshold = %d, lambda1 = %d, lambda2 = %d\n",
			params.I_threshold, params.lambda1, params.lambda2);
	}
	else
	{
		printf("      I_threshold = %d, lambda1 = %d/%d, lambda2 = %d/%d\n",
			params.I_threshold, params.lambda1, params.denominator,
			                    params.lambda2, params.denominator);
	}
	printf("      sub_pixel = %s, data_cost = L%d\n",
		params.sub_pixel ? "true" : "false", params.data_cost==Parameters::L1 ? 1 : 2);

	Run_KZ_BVZ(METHOD_BVZ);

	unique_flag = false;
}

/************************************************************/
/************************************************************/
/************************************************************/

void generate_permutation(int *buf, int n)
{
	int i, j;

	for (i=0; i<n; i++) buf[i] = i;
	for (i=0; i<n-1; i++)
	{
		j = i + (int) (((double)rand()/(RAND_MAX+1.0))*(n - i));
		int tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp;
	}
}

void Match::Run_KZ_BVZ(Method method)
{
	Coord a;
	int label_num;
	int *permutation; /* contains random permutation of 0, 1, ..., label_num-1 */
	bool *buf;  /* if buf[l] is true then expansion of label corresponding to l
	               cannot decrease the energy */
	int buf_num; /* number of 'false' entries in buf */
	int i, index, label;
	int step, iter;
	int E_old;

	unsigned int seed = time(NULL);
	printf("Random seed = %d\n", seed);
	srand(seed);

	label_num = disp_size.x * disp_size.y;
	if (method==METHOD_BVZ && params.occlusion_penalty<INFINITY) label_num ++;
	permutation = new int[label_num];
	buf = new bool[label_num];
	if (!permutation || !buf) { fprintf(stderr, "Not enough memory!\n"); exit(1); }

	switch (method)
	{
		case METHOD_KZ1: KZ1_ComputeEnergy(); break;
		case METHOD_KZ2: KZ2_ComputeEnergy(); break;
		case METHOD_BVZ: BVZ_ComputeEnergy(); break;
	}
	printf("E = %d\n", E);

	/* starting the algorithm */
	for (i=0; i<label_num; i++) buf[i] = false;
	buf_num = label_num;
	step = 0;
	for (iter=0; iter<params.iter_max && buf_num>0; iter++)
	{
		if (iter==0 || params.randomize_every_iteration)
			generate_permutation(permutation, label_num);

		for (index=0; index<label_num; index++)
		{
			label = permutation[index];
			if (buf[label]) continue;

			a.x = disp_base.x + label / disp_size.y;
			a.y = disp_base.y + label % disp_size.y;
			if (a.x > disp_max.x) a.x = a.y = OCCLUDED;

			E_old = E;

			switch (method)
			{
				case METHOD_KZ1: KZ1_Expand(a); break;
				case METHOD_KZ2: KZ2_Expand(a); break;
				case METHOD_BVZ: BVZ_Expand(a); break;
			}

#ifndef NDEBUG
			{
				int E_tmp = E;
				switch (method)
				{
					case METHOD_KZ1: KZ1_ComputeEnergy(); break;
					case METHOD_KZ2: KZ2_ComputeEnergy(); break;
					case METHOD_BVZ: BVZ_ComputeEnergy(); break;
				}
				if (E_tmp != E)
				{
					fprintf(stderr, "E and E_tmp are different! (E = %d, E_tmp = %d)\n", E, E_tmp);
					exit(1);
				}
			}
#endif

			step ++;
			if (E_old == E) printf("-");
			else printf("*");
			fflush(stdout);

			if (E_old == E)
			{
				if (!buf[label]) { buf[label] = true; buf_num --; }
			}
			else
			{
				int i;
				for (i=0; i<label_num; i++) buf[i] = false;
				buf[label] = true;
				buf_num = label_num - 1;
			}
		}
		printf(" E = %d\n", E); fflush(stdout);
	}

	printf("%.1f iterations\n", ((float)step)/label_num);

	delete permutation;
	delete buf;
}

/************************************************************/
/************************************************************/
/************************************************************/

