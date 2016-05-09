/* statistics.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include "match.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/************************************************************/
/************************************************************/
/************************************************************/

/*
	Heuristic for selecting parameter 'K'
	Details are described in my PhD thesis
*/
   

float Match::GetK()
{
	Coord p, p_base, p_max, d;
	int i, k;
	int *array, delta, sum = 0, num = 0;
	float K;

	i = disp_size.x * disp_size.y;
	k = (i + 2)/4; /* 0.25 times the number of disparities */
	if (k < 3) k = 3; if (k > i) k = i;

	array = new int[k];
	if (!array) { fprintf(stderr, "GetK: Not enough memory!\n"); exit(1); }

	p_base.x = MAX(-disp_base.x, 0);
	p_base.y = MAX(-disp_base.y, 0);
	p_max.x = MIN(im_size.x-1, im_size.x-1-disp_max.x);
	p_max.y = MIN(im_size.y-1, im_size.y-1-disp_max.y);

	for (p.y=p_base.y; p.y<=p_max.y; p.y++)
	for (p.x=p_base.x; p.x<=p_max.x; p.x++)
	{
		/* compute k'th smallest value among data_penalty(p, p+d) for all d */
		i = 0;

		for (d.y=disp_base.y; d.y<=disp_max.y; d.y++)
		for (d.x=disp_base.x; d.x<=disp_max.x; d.x++)
		{
			delta = (this->*data_penalty_func)(p, p+d);
			if (i < k) array[i++] = delta;
			else
			{
				for (i=0; i<k; i++)
				if (delta < array[i])
				{
					int tmp = delta;
					delta = array[i];
					array[i] = tmp;
				}
			}
		}

		delta = array[0];
		for (i=1; i<k; i++) if (delta < array[i]) delta = array[i];

		sum += delta;
		num ++;
	}

	delete array;
	if (num == 0) { fprintf(stderr, "GetK: Not enough samples!\n"); exit(1); }
	if (sum == 0) { fprintf(stderr, "GetK failed: K is 0!\n"); exit(1); }

	K = ((float)sum)/num;
	printf("Computing statistics: data_penalty noise is %f\n", K);
	return K;
}

