/* data.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

/*
	Functions depending on input images:

	data_penalty_X(Coord l, Coord r)
	smoothness_penalty_left_X(Coord p, Coord np, Coord disp, Coord ndisp)
	smoothness_penalty_right_X(Coord p, Coord np, Coord disp, Coord ndisp)
	smoothness_penalty2_X(Coord p, Coord np, Coord disp)

	where X describes the appropriate case (GRAY/COLOR, SUBPIXEL/no SUBPIXEL)
*/

#include <stdio.h>
#include <string.h>
#include "match.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/************************************************************/
/********************* data penalty *************************/
/************************************************************/

#define CUTOFF 1000

int Match::data_penalty_GRAY(Coord l, Coord r)
{
	int d;

	d = IMREF(im_left, l) - IMREF(im_right, r);
	if (params.data_cost==Parameters::L1) { if (d<0) d = -d; } else d = d*d;
	if (d>CUTOFF) d = CUTOFF;

	return d;
}

int Match::data_penalty_COLOR(Coord l, Coord r)
{
	int d, d_sum = 0;

	/* red component */
	d = IMREF(im_color_left, l).r - IMREF(im_color_right, r).r;
	if (params.data_cost==Parameters::L1) { if (d<0) d = -d; } else d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	/* green component */
	d = IMREF(im_color_left, l).g - IMREF(im_color_right, r).g;
	if (params.data_cost==Parameters::L1) { if (d<0) d = -d; } else d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	/* blue component */
	d = IMREF(im_color_left, l).b - IMREF(im_color_right, r).b;
	if (params.data_cost==Parameters::L1) { if (d<0) d = -d; } else d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	return d_sum/3;
}

int Match::data_penalty_SUBPIXEL_GRAY(Coord l, Coord r)
{
	int dl, dr, d;
	int Il, Il_min, Il_max, Ir, Ir_min, Ir_max;

	Il     = IMREF(im_left,     l); Ir     = IMREF(im_right,     r);
	Il_min = IMREF(im_left_min, l); Ir_min = IMREF(im_right_min, r);
	Il_max = IMREF(im_left_max, l); Ir_max = IMREF(im_right_max, r);

	if      (Il < Ir_min) dl = Ir_min - Il;
	else if (Il > Ir_max) dl = Il - Ir_max;
	else return 0;

	if      (Ir < Il_min) dr = Il_min - Ir;
	else if (Ir > Il_max) dr = Ir - Il_max;
	else return 0;

	d = MIN(dl, dr); if (params.data_cost==Parameters::L2) d = d*d;
	if (d>CUTOFF) d = CUTOFF;

	return d;
}

int Match::data_penalty_SUBPIXEL_COLOR(Coord l, Coord r)
{
	int dl, dr, d, d_sum = 0;
	int Il, Il_min, Il_max, Ir, Ir_min, Ir_max;

	/* red component */
	Il     = IMREF(im_color_left,     l).r; Ir     = IMREF(im_color_right,     r).r;
	Il_min = IMREF(im_color_left_min, l).r; Ir_min = IMREF(im_color_right_min, r).r;
	Il_max = IMREF(im_color_left_max, l).r; Ir_max = IMREF(im_color_right_max, r).r;

	if      (Il < Ir_min) dl = Ir_min - Il;
	else if (Il > Ir_max) dl = Il - Ir_max;
	else dl = 0;

	if      (Ir < Il_min) dr = Il_min - Ir;
	else if (Ir > Il_max) dr = Ir - Il_max;
	else dr = 0;

	d = MIN(dl, dr); if (params.data_cost==Parameters::L2) d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	/* green component */
	Il     = IMREF(im_color_left,     l).g; Ir     = IMREF(im_color_right,     r).g;
	Il_min = IMREF(im_color_left_min, l).g; Ir_min = IMREF(im_color_right_min, r).g;
	Il_max = IMREF(im_color_left_max, l).g; Ir_max = IMREF(im_color_right_max, r).g;

	if      (Il < Ir_min) dl = Ir_min - Il;
	else if (Il > Ir_max) dl = Il - Ir_max;
	else dl = 0;

	if      (Ir < Il_min) dr = Il_min - Ir;
	else if (Ir > Il_max) dr = Ir - Il_max;
	else dr = 0;

	d = MIN(dl, dr); if (params.data_cost==Parameters::L2) d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	/* blue component */
	Il     = IMREF(im_color_left,     l).b; Ir     = IMREF(im_color_right,     r).b;
	Il_min = IMREF(im_color_left_min, l).b; Ir_min = IMREF(im_color_right_min, r).b;
	Il_max = IMREF(im_color_left_max, l).b; Ir_max = IMREF(im_color_right_max, r).b;

	if      (Il < Ir_min) dl = Ir_min - Il;
	else if (Il > Ir_max) dl = Il - Ir_max;
	else dl = 0;

	if      (Ir < Il_min) dr = Il_min - Ir;
	else if (Ir > Il_max) dr = Ir - Il_max;
	else dr = 0;

	d = MIN(dl, dr); if (params.data_cost==Parameters::L2) d = d*d;
	if (d>CUTOFF) d = CUTOFF;
	d_sum += d;

	return d_sum/3;
}

/************************************************************/
/******************* sub_pixel preprocessing ****************/
/************************************************************/

void Match::InitSubPixel()
{
	if (params.sub_pixel && im_left && !im_left_min)
	{
		im_left_min  = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);
		im_left_max  = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);
		im_right_min = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);
		im_right_max = (GrayImage) imNew(IMAGE_GRAY, im_size.x, im_size.y);

		if (!im_left_min || !im_left_max || !im_right_min || !im_right_max)
		{ fprintf(stderr, "Not enough memory!\n"); exit(1); }

		SubPixel(im_left,  im_left_min,  im_left_max);
		SubPixel(im_right, im_right_min, im_right_max);
	}
	if (params.sub_pixel && im_color_left && !im_color_left_min)
	{
		im_color_left_min  = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);
		im_color_left_max  = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);
		im_color_right_min = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);
		im_color_right_max = (RGBImage) imNew(IMAGE_RGB, im_size.x, im_size.y);

		if (!im_color_left_min || !im_color_left_max || !im_color_right_min || !im_color_right_max)
		{ fprintf(stderr, "Not enough memory!\n"); exit(1); }

		SubPixelColor(im_color_left,  im_color_left_min,  im_color_left_max);
		SubPixelColor(im_color_right, im_color_right_min, im_color_right_max);
	}
}

void Match::SubPixel(GrayImage Im, GrayImage ImMin, GrayImage ImMax)
{
	Coord p;
	int I, I1, I2, I3, I4, I_min, I_max;

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		I = I_min = I_max = imRef(Im, p.x, p.y);
		if (p.x>0)           I1 = (imRef(Im, p.x-1, p.y) + I) / 2;
		else                 I1 = I;
		if (p.x<im_size.x-1) I2 = (imRef(Im, p.x+1, p.y) + I) / 2;
		else                 I2 = I;
		if (p.y>0)           I3 = (imRef(Im, p.x, p.y-1) + I) / 2;
		else                 I3 = I;
		if (p.y<im_size.y-1) I4 = (imRef(Im, p.x, p.y+1) + I) / 2;
		else                 I4 = I;

		if (I_min > I1) I_min = I1;
		if (I_min > I2) I_min = I2;
		if (I_min > I3) I_min = I3;
		if (I_min > I4) I_min = I4;
		if (I_max < I1) I_max = I1;
		if (I_max < I2) I_max = I2;
		if (I_max < I3) I_max = I3;
		if (I_max < I4) I_max = I4;

		imRef(ImMin, p.x, p.y) = I_min;
		imRef(ImMax, p.x, p.y) = I_max;
	}
}

void Match::SubPixelColor(RGBImage Im, RGBImage ImMin, RGBImage ImMax)
{
	Coord p;
	int I, I1, I2, I3, I4, I_min, I_max;

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		/* red component */
		I = I_min = I_max = imRef(Im, p.x, p.y).r;
		if (p.x>0)           I1 = (imRef(Im, p.x-1, p.y).r + I) / 2;
		else                 I1 = I;
		if (p.x<im_size.x-1) I2 = (imRef(Im, p.x+1, p.y).r + I) / 2;
		else                 I2 = I;
		if (p.y>0)           I3 = (imRef(Im, p.x, p.y-1).r + I) / 2;
		else                 I3 = I;
		if (p.y<im_size.y-1) I4 = (imRef(Im, p.x, p.y+1).r + I) / 2;
		else                 I4 = I;

		if (I_min > I1) I_min = I1;
		if (I_min > I2) I_min = I2;
		if (I_min > I3) I_min = I3;
		if (I_min > I4) I_min = I4;
		if (I_max < I1) I_max = I1;
		if (I_max < I2) I_max = I2;
		if (I_max < I3) I_max = I3;
		if (I_max < I4) I_max = I4;

		imRef(ImMin, p.x, p.y).r = I_min;
		imRef(ImMax, p.x, p.y).r = I_max;


		/* green component */
		I = I_min = I_max = imRef(Im, p.x, p.y).g;
		if (p.x>0)           I1 = (imRef(Im, p.x-1, p.y).g + I) / 2;
		else                 I1 = I;
		if (p.x<im_size.x-1) I2 = (imRef(Im, p.x+1, p.y).g + I) / 2;
		else                 I2 = I;
		if (p.y>0)           I3 = (imRef(Im, p.x, p.y-1).g + I) / 2;
		else                 I3 = I;
		if (p.y<im_size.y-1) I4 = (imRef(Im, p.x, p.y+1).g + I) / 2;
		else                 I4 = I;

		if (I_min > I1) I_min = I1;
		if (I_min > I2) I_min = I2;
		if (I_min > I3) I_min = I3;
		if (I_min > I4) I_min = I4;
		if (I_max < I1) I_max = I1;
		if (I_max < I2) I_max = I2;
		if (I_max < I3) I_max = I3;
		if (I_max < I4) I_max = I4;

		imRef(ImMin, p.x, p.y).g = I_min;
		imRef(ImMax, p.x, p.y).g = I_max;


		/* blue component */
		I = I_min = I_max = imRef(Im, p.x, p.y).b;
		if (p.x>0)           I1 = (imRef(Im, p.x-1, p.y).b + I) / 2;
		else                 I1 = I;
		if (p.x<im_size.x-1) I2 = (imRef(Im, p.x+1, p.y).b + I) / 2;
		else                 I2 = I;
		if (p.y>0)           I3 = (imRef(Im, p.x, p.y-1).b + I) / 2;
		else                 I3 = I;
		if (p.y<im_size.y-1) I4 = (imRef(Im, p.x, p.y+1).b + I) / 2;
		else                 I4 = I;

		if (I_min > I1) I_min = I1;
		if (I_min > I2) I_min = I2;
		if (I_min > I3) I_min = I3;
		if (I_min > I4) I_min = I4;
		if (I_max < I1) I_max = I1;
		if (I_max < I2) I_max = I2;
		if (I_max < I3) I_max = I3;
		if (I_max < I4) I_max = I4;

		imRef(ImMin, p.x, p.y).b = I_min;
		imRef(ImMax, p.x, p.y).b = I_max;
	}
}

/************************************************************/
/****************** smoothness penalty **********************/
/******************** (static clues) ************************/
/************************************************************/

int Match::smoothness_penalty_left_GRAY(Coord p, Coord np, Coord disp, Coord ndisp)
{
	int d, R;

	if (disp == ndisp) return 0;
	if (disp.x == OCCLUDED || ndisp.x == OCCLUDED) R = params.interaction_radius;
	else
	{
		int Rx = disp.x - ndisp.x; if (Rx < 0) Rx = -Rx;
		int Ry = disp.y - ndisp.y; if (Ry < 0) Ry = -Ry;
		R = Rx + Ry;
		if (R > params.interaction_radius) R = params.interaction_radius;
	}

	d = IMREF(segm_left, p) - IMREF(segm_left, np);
	if (d<0) d = -d;

	if (d<params.I_threshold) return R*params.lambda1;
	else                      return R*params.lambda2;
}

int Match::smoothness_penalty_left_COLOR(Coord p, Coord np, Coord disp, Coord ndisp)
{
	int d, d_max, R;

	if (disp == ndisp) return 0;
	if (disp.x == OCCLUDED || ndisp.x == OCCLUDED) R = params.interaction_radius;
	else
	{
		int Rx = disp.x - ndisp.x; if (Rx < 0) Rx = -Rx;
		int Ry = disp.y - ndisp.y; if (Ry < 0) Ry = -Ry;
		R = Rx + Ry;
		if (R > params.interaction_radius) R = params.interaction_radius;
	}

	d_max = 0;

	/* red component */
	d = IMREF(segm_color_left, p).r - IMREF(segm_color_left, np).r;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* green component */
	d = IMREF(segm_color_left, p).g - IMREF(segm_color_left, np).g;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* blue component */
	d = IMREF(segm_color_left, p).b - IMREF(segm_color_left, np).b;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	if (d_max<params.I_threshold) return R*params.lambda1;
	else                          return R*params.lambda2;
}

int Match::smoothness_penalty_right_GRAY(Coord p, Coord np, Coord disp, Coord ndisp)
{
	int d, R;

	if (disp == ndisp) return 0;
	if (disp.x == OCCLUDED || ndisp.x == OCCLUDED) R = params.interaction_radius;
	else
	{
		int Rx = disp.x - ndisp.x; if (Rx < 0) Rx = -Rx;
		int Ry = disp.y - ndisp.y; if (Ry < 0) Ry = -Ry;
		R = Rx + Ry;
		if (R > params.interaction_radius) R = params.interaction_radius;
	}

	d = IMREF(segm_right, p) - IMREF(segm_right, np);
	if (d<0) d = -d;

	if (d<params.I_threshold) return R*params.lambda1;
	else                      return R*params.lambda2;
}

int Match::smoothness_penalty_right_COLOR(Coord p, Coord np, Coord disp, Coord ndisp)
{
	int d, d_max, R;

	if (disp == ndisp) return 0;
	if (disp.x == OCCLUDED || ndisp.x == OCCLUDED) R = params.interaction_radius;
	else
	{
		int Rx = disp.x - ndisp.x; if (Rx < 0) Rx = -Rx;
		int Ry = disp.y - ndisp.y; if (Ry < 0) Ry = -Ry;
		R = Rx + Ry;
		if (R > params.interaction_radius) R = params.interaction_radius;
	}

	d_max = 0;

	/* red component */
	d = IMREF(segm_color_right, p).r - IMREF(segm_color_right, np).r;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* green component */
	d = IMREF(segm_color_right, p).g - IMREF(segm_color_right, np).g;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* blue component */
	d = IMREF(segm_color_right, p).b - IMREF(segm_color_right, np).b;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	if (d_max<params.I_threshold) return R*params.lambda1;
	else                          return R*params.lambda2;
}

int Match::smoothness_penalty2_GRAY(Coord p, Coord np, Coord disp)
{
	int dl, dr;

	dl = IMREF(segm_left, p) - IMREF(segm_left, np);   
	dr = IMREF(segm_right, p+disp) - IMREF(segm_right, np+disp); 

	if (dl<0) dl = -dl; if (dr<0) dr = -dr;

	if (dl<params.I_threshold2 && dr<params.I_threshold2) return params.lambda1;
	else                                                  return params.lambda2;
}

int Match::smoothness_penalty2_COLOR(Coord p, Coord np, Coord disp)
{
	int d, d_max;

	d_max = 0;

	/* red component */
	d = IMREF(segm_color_left, p).r - IMREF(segm_color_left, np).r;
	if (d<0) d = -d; if (d_max<d) d_max = d;
	d = IMREF(segm_color_right, p+disp).r - IMREF(segm_color_right, np+disp).r;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* green component */
	d = IMREF(segm_color_left, p).g - IMREF(segm_color_left, np).g;
	if (d<0) d = -d; if (d_max<d) d_max = d;
	d = IMREF(segm_color_right, p+disp).g - IMREF(segm_color_right, np+disp).g;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	/* blue component */
	d = IMREF(segm_color_left, p).b - IMREF(segm_color_left, np).b;
	if (d<0) d = -d; if (d_max<d) d_max = d;
	d = IMREF(segm_color_right, p+disp).b - IMREF(segm_color_right, np+disp).b;
	if (d<0) d = -d; if (d_max<d) d_max = d;

	if (d_max<params.I_threshold2) return params.lambda1;
	else                           return params.lambda2;
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::SetParameters(Parameters *_params)
{
	memcpy(&params, _params, sizeof(params));

	if (im_left)
	{
		if (params.sub_pixel) data_penalty_func = &Match::data_penalty_SUBPIXEL_GRAY;
		else                  data_penalty_func = &Match::data_penalty_GRAY;
	}
	else
	{
		if (params.sub_pixel) data_penalty_func = &Match::data_penalty_SUBPIXEL_COLOR;
		else                  data_penalty_func = &Match::data_penalty_COLOR;
	}

	if (segm_left)
	{
		smoothness_penalty_left_func = &Match::smoothness_penalty_left_GRAY;
		smoothness_penalty_right_func = &Match::smoothness_penalty_right_GRAY;
		smoothness_penalty2_func = &Match::smoothness_penalty2_GRAY;
	}
	else
	{
		smoothness_penalty_left_func = &Match::smoothness_penalty_left_COLOR;
		smoothness_penalty_right_func = &Match::smoothness_penalty_right_COLOR;
		smoothness_penalty2_func = &Match::smoothness_penalty2_COLOR;
	}

	InitSubPixel();
}
