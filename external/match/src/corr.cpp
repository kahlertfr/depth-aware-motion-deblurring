/* corr.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include "match.h"

/************************************************************/
/************************************************************/
/************************************************************/

inline int Match::CORR_data_penalty(Coord l, Coord r)
{
	return (this->*data_penalty_func)(l, r);
}

/************************************************************/
/************************************************************/
/************************************************************/

#define DEFAULT_V 100

/*
  compute correlation values for disparity d
  using 1D correlation window (along horizontal lines)
*/
void Match::CORR_hor(Coord d, LongImage v_hor)
{
	Coord p, pd;
	int w_size, w2_size;
	int x_start, x_finish;
	int v, v_sum;

	w_size = params.corr_size;
	w2_size = 2*w_size + 1;

	for (p.y=0; p.y<im_size.y; p.y++)
	{
		pd.y = p.y + d.y;
		if (pd.y<0 || pd.y>=im_size.y)
		{
			for (p.x=0; p.x<im_size.x; p.x++) IMREF(v_hor, p) = w2_size*DEFAULT_V;
			continue;
		}

		if (d.x > 0)
		{
			x_start = 0;
			x_finish = im_size.x - 1 - d.x;
			if (x_start > x_finish) x_finish = x_start - 1;
		}
		else
		{
			x_start = 0 - d.x;
			x_finish = im_size.x - 1;
			if (x_start > x_finish) x_start = x_finish + 1;
		}
		/* now we can access L(p) and R(p+d) if x_start<=p.x<=x_finish */

		v_sum = w2_size*DEFAULT_V;
		for (p.x=x_start; p.x<x_start+w_size; p.x++)
		{
			pd.x = p.x + d.x;
			v = CORR_data_penalty(p, pd);
			v_sum += v;
			v_sum -= DEFAULT_V;
		}
		for (; p.x<x_start+w2_size; p.x++)
		{
			pd.x = p.x + d.x;
			v = CORR_data_penalty(p, pd);
			v_sum += v;
			v_sum -= DEFAULT_V;
			imRef(v_hor, p.x-w_size, p.y) = v_sum;
		}
		for (; p.x<=x_finish; p.x++)
		{
			pd.x = p.x + d.x;
			v = CORR_data_penalty(p, pd);
			v_sum += v;
			v = CORR_data_penalty(Coord(p.x-w2_size, p.y), Coord(pd.x-w2_size, pd.y));
			v_sum -= v;
			imRef(v_hor, p.x-w_size, p.y) = v_sum;
		}
		for (; p.x<=x_finish+w_size; p.x++)
		{
			pd.x = p.x + d.x;
			v_sum += DEFAULT_V;
			v = CORR_data_penalty(Coord(p.x-w2_size, p.y), Coord(pd.x-w2_size, pd.y));
			v_sum -= v;
			imRef(v_hor, p.x-w_size, p.y) = v_sum;
		}

		if (d.x > 0)
		{
			v_sum = imRef(v_hor, x_finish, p.y);
			for (p.x=x_finish+1; p.x<im_size.x; p.x++)
			{
				if (p.x-w_size-1>=0 && p.x-w_size-1+d.x<im_size.x)
				{
					v = CORR_data_penalty(Coord(p.x-w_size-1, p.y), Coord(p.x-w_size-1+d.x, pd.y));
					v_sum -= v;
				}
				else v_sum -= DEFAULT_V;
				v_sum += DEFAULT_V;
				IMREF(v_hor, p) = v_sum;
			}
		}
		else
		{
			v_sum = imRef(v_hor, x_start, p.y);
			for (p.x=x_start-1; p.x>=0; p.x--)
			{
				if (p.x+w_size+1<im_size.x && p.x+w_size+1+d.x>=0)
				{
					v = CORR_data_penalty(Coord(p.x+w_size+1, p.y), Coord(p.x+w_size+1+d.x, pd.y));
					v_sum -= v;
				}
				else v_sum -= DEFAULT_V;
				v_sum += DEFAULT_V;
				IMREF(v_hor, p) = v_sum;
			}
		}
	}
}

/************************************************************/
/************************************************************/
/************************************************************/

/*
  compute correlation values
  using full 2D correlation window
*/
void Match::CORR_full(LongImage v_hor, LongImage v_full)
{
	Coord p;
	int w_size, w2_size;
	int v, v_sum;

	w_size = params.corr_size;
	w2_size = 2*w_size + 1;

	for (p.x=0; p.x<im_size.x; p.x++)
	{
		v_sum = w2_size*w2_size*DEFAULT_V;
		for (p.y=0; p.y<w_size; p.y++)
		{
			v = IMREF(v_hor, p);
			v_sum += v;
			v_sum -= w2_size*DEFAULT_V;
		}
		for (; p.y<w2_size; p.y++)
		{
			v = IMREF(v_hor, p);
			v_sum += v;
			v_sum -= w2_size*DEFAULT_V;
			imRef(v_full, p.x, p.y-w_size) = v_sum;
		}
		for (; p.y<im_size.y; p.y++)
		{
			v = IMREF(v_hor, p);
			v_sum += v;
			v = imRef(v_hor, p.x, p.y-w2_size);
			v_sum -= v;
			imRef(v_full, p.x, p.y-w_size) = v_sum;
		}
		for (; p.y<im_size.y+w_size; p.y++)
		{
			v_sum += w2_size*DEFAULT_V;
			v = imRef(v_hor, p.x, p.y-w2_size);
			v_sum -= v;
			imRef(v_full, p.x, p.y-w_size) = v_sum;
		}
	}
}

/************************************************************/
/************************************************************/
/************************************************************/

void Match::CORR()
{
	Coord p, d, pd;
	int w_size, w2_size;
	PtrImage V_FULL;
	LongImage v_full, v_hor;
	int v;

	w_size = params.corr_size;
	w2_size = 2*w_size + 1;

	if (w_size < 0)
	{
		fprintf(stderr, "Error in CORR: wrong parameter!\n");
		exit(1);
	}

	printf("CORR:  corr_size = %d, data_cost = L%d\n",
		w_size, params.data_cost == Parameters::L1 ? 1 : 2);

	/* allocating memory */
	V_FULL = (PtrImage) imNew(IMAGE_PTR, disp_size.x, disp_size.y);
	if (!V_FULL) { fprintf(stderr, "CORR: Not enough memory!\n"); exit(1); }
	for (d.y=disp_base.y; d.y<=disp_max.y; d.y++)
	for (d.x=disp_base.x; d.x<=disp_max.x; d.x++)
	{
		IMREF(V_FULL, d-disp_base) = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
		if (!IMREF(V_FULL, d-disp_base)) { fprintf(stderr, "CORR: Not enough memory!\n"); exit(1); }
	}
	v_hor  = (LongImage) imNew(IMAGE_LONG, im_size.x, im_size.y);
	if (!v_hor) { fprintf(stderr, "CORR: Not enough memory!\n"); exit(1); }

	/* computing correlation values */
	for (d.y=disp_base.y; d.y<=disp_max.y; d.y++)
	for (d.x=disp_base.x; d.x<=disp_max.x; d.x++)
	{
		/* computing v_hor */
		CORR_hor(d, v_hor);

		/* computing v_full */
		v_full = (LongImage) IMREF(V_FULL, d-disp_base);
		CORR_full(v_hor, v_full);

#ifdef NOT_DEFINED
		/* sanity check */
		{
			for (p.y=im_base.y; p.y<=im_max.y; p.y++)
			for (p.x=im_base.x; p.x<=im_max.x; p.x++)
			{
				Coord q, qd;
				int v_sum = 0;
				for (q.y=p.y-w_size; q.y<=p.y+w_size; q.y++)
				for (q.x=p.x-w_size; q.x<=p.x+w_size; q.x++)
				{
					qd = q + d;
					if (q>=im_base && q<=im_max && qd>=im_base && qd<=im_max)
					{
						v = CORR_value(q, qd);
						v_sum += v;
					}
					else v_sum += DEFAULT_V;
				}
				if (v_sum != IMREF(v_full, p)) { fprintf(stderr, "Error in CORR!\n"); exit(1); }
			}
		}
#endif
	}

	/* finding disparity with minimum correlation value */
	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		Coord d_min;
		int v_min = -1;

		for (d.y=disp_base.y; d.y<=disp_max.y; d.y++)
		for (d.x=disp_base.x; d.x<=disp_max.x; d.x++)
		{
			v_full = (LongImage) IMREF(V_FULL, d-disp_base);
			v = IMREF(v_full, p);
			if (v_min < 0 || v < v_min)
			{
				v_min = v;
				d_min = d;
			}
		}

		IMREF(x_left, p) = d_min.x;
		IMREF(y_left, p) = d_min.y;
	}

	/* freeing data */
	for (d.y=disp_base.y; d.y<=disp_max.y; d.y++)
	for (d.x=disp_base.x; d.x<=disp_max.x; d.x++)
	{
		v_full = (LongImage) (IMREF(V_FULL, d-disp_base));
		imFree(v_full);
	}
	imFree(V_FULL);
	imFree(v_hor);

	unique_flag = false;
}
