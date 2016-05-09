/* kz2.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include "match.h"
#include "energy.h"

/************************************************************/
/************************************************************/
/************************************************************/

inline int Match::KZ2_data_penalty(Coord l, Coord r)
{
	return params.denominator*(this->*data_penalty_func)(l, r) - params.K;
}

inline int Match::KZ2_smoothness_penalty2(Coord p, Coord np, Coord d)
{
	return (this->*smoothness_penalty2_func)(p, np, d);
}

/************************************************************/
/************************************************************/
/************************************************************/

/* computes current energy */
int Match::KZ2_ComputeEnergy()
{
	int k;
	Coord p, d, np, nd;

	E = 0;

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));

		if (d.x != OCCLUDED) E += KZ2_data_penalty(p, p+d);

		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			np = p + NEIGHBORS[k];

			if (np>=Coord(0,0) && np<im_size)
			{
				nd = Coord(IMREF(x_left, np), IMREF(y_left, np));
				if (d == nd) continue;
				if (d.x!=OCCLUDED && np+d>=Coord(0,0) && np+d<im_size)
					E += KZ2_smoothness_penalty2(p, np, d);
				if (nd.x!=OCCLUDED && p+nd>=Coord(0,0) && p+nd<im_size)
					E += KZ2_smoothness_penalty2(p, np, nd);
			}
		}
	}

	return E;
}

/************************************************************/
/************************************************************/
/************************************************************/

#define node_vars_0 ptr_im1
#define node_vars_a ptr_im2
#define VAR_ACTIVE     ((Energy::Var)0)
#define VAR_NONPRESENT ((Energy::Var)1)
#define IS_VAR(var) ((unsigned)var>1)

#define KZ2_ALPHA_SINK
/*
	if KZ2_ALPHA_SINK is defined then interpretation of a cut is as in the paper:
	for assignments in A^0:
		SOURCE means 1
		SINK   means 0
	for assigments in A^{\alpha}:
		SOURCE means 0
		SINK   means 1

	if KZ2_ALPHA_SINK is not defined then SOURCE and SINK are swapped
*/
#ifdef KZ2_ALPHA_SINK
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

void KZ2_error_function(char *msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

/* computes the minimum a-expansion configuration */
void Match::KZ2_Expand(Coord a)
{
	Coord p, d, pd, pa, np, nd;
	int E_old, delta;
	Energy::Var var_0, var_a, nvar_0, nvar_a;
	int k;
	
	Energy *e = new Energy(KZ2_error_function);

	/* initializing */
	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		pd = p + d;
		if (a == d)
		{
			IMREF(node_vars_0, p) = VAR_ACTIVE;
			IMREF(node_vars_a, p) = VAR_ACTIVE;
			e -> add_constant(KZ2_data_penalty(p, pd));
			continue;
		}

		if (d.x != OCCLUDED)
		{
			IMREF(node_vars_0, p) = var_0 = e -> add_variable();
			e -> ADD_TERM1(var_0, KZ2_data_penalty(p, pd), 0);
		}
		else IMREF(node_vars_0, p) = VAR_NONPRESENT;

		pa = p + a;
		if (pa>=Coord(0,0) && pa<im_size)
		{
			IMREF(node_vars_a, p) = var_a = e -> add_variable();
			e -> ADD_TERM1(var_a, 0, KZ2_data_penalty(p, pa));
		}
		else IMREF(node_vars_a, p) = VAR_NONPRESENT;
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		var_0 = (Energy::Var) IMREF(node_vars_0, p);
		var_a = (Energy::Var) IMREF(node_vars_a, p);

		/* adding smoothness */
		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			np = p + NEIGHBORS[k];
			if ( ! ( np>=Coord(0,0) && np<im_size ) ) continue;
			nd = Coord(IMREF(x_left, np), IMREF(y_left, np));
			nvar_0 = (Energy::Var) IMREF(node_vars_0, np);
			nvar_a = (Energy::Var) IMREF(node_vars_a, np);

			/* disparity a */
			if (var_a!=VAR_NONPRESENT && nvar_a!=VAR_NONPRESENT)
			/* p+a and np+a are inside the right image */
			{
				delta = KZ2_smoothness_penalty2(p, np, a);

				if (var_a != VAR_ACTIVE)
				{
					if (nvar_a != VAR_ACTIVE) e -> ADD_TERM2(var_a, nvar_a, 0, delta, delta, 0);
					else                      e -> ADD_TERM1(var_a, delta, 0);
				}
				else
				{
					if (nvar_a != VAR_ACTIVE) e -> ADD_TERM1(nvar_a, delta, 0);
					else                      {}
				}
			}

			/* disparity d (unless it was checked before) */
			if (IS_VAR(var_0) && np+d>=Coord(0,0) && np+d<im_size)
			{
				delta = KZ2_smoothness_penalty2(p, np, d);
				if (d == nd) e -> ADD_TERM2(var_0, nvar_0, 0, delta, delta, 0);
				else e -> ADD_TERM1(var_0, delta, 0);
			}

			/* direction nd (unless it was checked before) */
			if (IS_VAR(nvar_0) && d!=nd && p+nd>=Coord(0,0) && p+nd<im_size)
			{
				delta = KZ2_smoothness_penalty2(p, np, nd);
				e -> ADD_TERM1(nvar_0, delta, 0);
			}
		}

		/* adding hard constraints in the left image */
		if (IS_VAR(var_0) && var_a!=VAR_NONPRESENT)
			e -> ADD_TERM2(var_0, var_a, 0, INFINITY, 0, 0);

		/* adding hard constraints in the right image */
		d = Coord(IMREF(x_right, p), IMREF(y_right, p));
		if (d.x != OCCLUDED)
		{
			var_0 = (Energy::Var) IMREF(node_vars_0, p + d);
			if (var_0 != VAR_ACTIVE)
			{
				pa = p - a;
				if (pa>=Coord(0,0) && pa<im_size)
				{
					var_a = (Energy::Var) IMREF(node_vars_a, pa);
					e -> ADD_TERM2(var_0, var_a, 0, INFINITY, 0, 0);
				}
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
			IMREF(x_right, p) = OCCLUDED;
		}

		for (p.y=0; p.y<im_size.y; p.y++)
		for (p.x=0; p.x<im_size.x; p.x++)
		{
			var_0 = (Energy::Var) IMREF(node_vars_0, p);
			var_a = (Energy::Var) IMREF(node_vars_a, p);
			if ( (IS_VAR(var_0) && e->get_var(var_0)==VALUE0) ||
			     (var_0==VAR_ACTIVE) )
			{
				d = Coord(IMREF(x_left, p), IMREF(y_left, p));
				pd = p + d;
				IMREF(x_right, pd) = -d.x; IMREF(y_right, pd) = -d.y;
			}
			else if (IS_VAR(var_a) && e->get_var(var_a)==VALUE1)
			{
				pa = p + a;
				IMREF(x_left, p) = a.x;    IMREF(y_left, p) = a.y;
				IMREF(x_right, pa) = -a.x; IMREF(y_right, pa) = -a.y;
			}
			else IMREF(x_left, p) = OCCLUDED;
		}
	}

	delete e;
}

