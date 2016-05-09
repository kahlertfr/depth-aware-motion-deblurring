/* bvz.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include "match.h"
#include "energy.h"

/************************************************************/
/************************************************************/
/************************************************************/

inline int Match::BVZ_data_penalty(Coord p, Coord d)
{
	register Coord pd;

	if (d.x == OCCLUDED) return params.denominator*params.occlusion_penalty;
	pd = p + d;
	if (!(pd>=Coord(0,0) && pd<im_size)) return 1000;
	return params.denominator*(this->*data_penalty_func)(p, pd);
}

inline int Match::BVZ_smoothness_penalty(Coord p, Coord np, Coord d, Coord nd)
{
	return (this->*smoothness_penalty_left_func)(p, np, d, nd);
}

/************************************************************/
/************************************************************/
/************************************************************/

/* computes current energy */
int Match::BVZ_ComputeEnergy()
{
	int k;
	Coord p, d, q, dq;

	E = 0;

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));

		E += BVZ_data_penalty(p, d);

		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];

			if (q>=Coord(0,0) && q<im_size)
			{
				dq = Coord(IMREF(x_left, q), IMREF(y_left, q));
				E += BVZ_smoothness_penalty(p, q, d, dq);
			}
		}
	}

	return E;
}

/************************************************************/
/************************************************************/
/************************************************************/

#define node_vars ptr_im1
#define VAR_ACTIVE ((Energy::Var)0)

#define BVZ_ALPHA_SINK
/*
	if BVZ_ALPHA_SINK is defined then interpretation of a cut is as follows:
		SOURCE means initial label
		SINK   means new label \alpha

	if BVZ_ALPHA_SINK is not defined then SOURCE and SINK are swapped
*/
#ifdef BVZ_ALPHA_SINK
	#define ADD_TERM1(var, E0, E1) add_term1(var, E0, E1)
	#define ADD_TERM2(var1, var2, E00, E01, E10, E11) add_term2(var1, var2, E00, E01, E10, E11)
	#define VALUE0 0
	#define VALUE1 1
#else
	#define ADD_TERM1(var, E0, E1) add_term1(var, E1, E0)
	#define ADD_TERM2(var1, var2, E00, E01, E10, E11) add_term2(var1, var2, E11, E10, E01, E00)
	#define VALUE0 1
	#define VALUE1 0
#endif

void BVZ_error_function(char *msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

void Match::BVZ_Expand(Coord a)
{
	Coord p, d, q, dq;
	Energy::Var var, qvar;
	int E_old, E00, E0a, Ea0;
	int k;

	/* node_vars stores variables corresponding to nodes */

	Energy *e = new Energy(BVZ_error_function);

	/* initializing */
	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		if (a == d)
		{
			IMREF(node_vars, p) = VAR_ACTIVE;
			e -> add_constant(BVZ_data_penalty(p, d));
		}
		else
		{
			IMREF(node_vars, p) = var = e -> add_variable();
			e -> ADD_TERM1(var, BVZ_data_penalty(p, d), BVZ_data_penalty(p, a));
		}
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		var = (Energy::Var) IMREF(node_vars, p);

		/* smoothness term */
		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];
			if ( ! ( q>=Coord(0,0) && q<im_size ) ) continue;
			qvar = (Energy::Var) IMREF(node_vars, q);
			dq = Coord(IMREF(x_left, q), IMREF(y_left, q));

			if (var != VAR_ACTIVE && qvar != VAR_ACTIVE)
				E00 = BVZ_smoothness_penalty(p, q, d, dq);
			if (var != VAR_ACTIVE)
				E0a = BVZ_smoothness_penalty(p, q, d, a);
			if (qvar != VAR_ACTIVE)
				Ea0 = BVZ_smoothness_penalty(p, q, a, dq);

			if (var != VAR_ACTIVE)
			{
				if (qvar != VAR_ACTIVE) e -> ADD_TERM2(var, qvar, E00, E0a, Ea0, 0);
				else                    e -> ADD_TERM1(var, E0a, 0);
			}
			else
			{
				if (qvar != VAR_ACTIVE) e -> ADD_TERM1(qvar, Ea0, 0);
				else                    {}
			}
			
		}
	}

	E_old = E;
	E = e -> minimize();

	if (E < E_old)
	{
		for (p.y=0; p.y<im_size.y; p.y++)
		for (p.x=0; p.x<im_size.x; p.x++)
		{
			var = (Energy::Var) IMREF(node_vars, p);
			if (var!=VAR_ACTIVE && e->get_var(var)==VALUE1)
			{
				IMREF(x_left, p) = a.x; IMREF(y_left, p) = a.y;
			}
		}
	}

	delete e;
}
