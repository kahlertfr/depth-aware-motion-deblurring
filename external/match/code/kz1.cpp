/* kz1.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#include <stdio.h>
#include "match.h"
#include "energy.h"

/*
	if d1 and d2 are two disparities for a pixel p in the left image,
	then is_blocked(d1, d2) is true if (p,d2) is closer to the camera
	than (p,d1)
*/
#define is_blocked(d1, d2) (KZ1_visibility && ((d1).x > (d2).x)) /* true only for stereo */

/************************************************************/
/************************************************************/
/************************************************************/

inline int Match::KZ1_data_penalty(Coord l, Coord r)
{
	register int v = params.denominator*(this->*data_penalty_func)(l, r);
	v -= params.K;
	if (v>0) v = 0;
	return v;
}

inline int Match::KZ1_smoothness_penalty_left(Coord p, Coord np, Coord d, Coord nd)
{
	return (this->*smoothness_penalty_left_func)(p, np, d, nd);
}

inline int Match::KZ1_smoothness_penalty_right(Coord p, Coord np, Coord d, Coord nd)
{
	return (this->*smoothness_penalty_right_func)(p, np, d, nd);
}

#define KZ1_OCCLUSION_PENALTY 1000

/************************************************************/
/************************************************************/
/************************************************************/

/* computes current energy */
int Match::KZ1_ComputeEnergy()
{
	int k;
	Coord p, d, q, dq;

	E = 0;

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		/* left image and data penalty */
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));

		if (d.x == OCCLUDED) E += KZ1_OCCLUSION_PENALTY;
		else
		{
			q = p + d;
			if (q>=Coord(0,0) && q<im_size)
			{
				dq = Coord(IMREF(x_right, q), IMREF(y_right, q));
				if (d == -dq)
				{
					E += KZ1_data_penalty(p, q);
				}
#ifndef NDEBUG
				/* check visibility constaint */
				if (dq.x != OCCLUDED && is_blocked(-dq, d))
				{
					fprintf(stderr, "KZ1: Visibility constraint is violated!\n");
					exit(1);
				}
#endif
			}
		}

		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];

			if (q>=Coord(0,0) && q<im_size)
			{
				dq = Coord(IMREF(x_left, q), IMREF(y_left, q));
				E += KZ1_smoothness_penalty_left(p, q, d, dq);
			}
		}


		/* right image */
		d = Coord(IMREF(x_right, p), IMREF(y_right, p));

		if (d.x == OCCLUDED) E += KZ1_OCCLUSION_PENALTY;
#ifndef NDEBUG
		else
		{
			/* check visibility constaint */
			q = p + d;
			if (q>=Coord(0,0) && q<im_size)
			{
				dq = Coord(IMREF(x_left, q), IMREF(y_left, q));
				if (dq.x != OCCLUDED && is_blocked(dq, -d))
				{
					fprintf(stderr, "KZ1: Visibility constraint is violated!\n");
					exit(1);
				}
			}
		}
#endif

		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];

			if (q>=Coord(0,0) && q<im_size)
			{
				dq = Coord(IMREF(x_right, q), IMREF(y_right, q));
				E += KZ1_smoothness_penalty_right(p, q, d, dq);
			}
		}
	}

	return E;
}

/************************************************************/
/************************************************************/
/************************************************************/

#define node_vars_left ptr_im1
#define node_vars_right ptr_im2
#define VAR_ACTIVE ((Energy::Var)0)

#define KZ1_ALPHA_SINK
/*
	if KZ1_ALPHA_SINK is defined then interpretation of a cut is as follows:
		SOURCE means initial label
		SINK   means new label \alpha

	if KZ1_ALPHA_SINK is not defined then SOURCE and SINK are swapped
*/
#ifdef KZ1_ALPHA_SINK
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

void KZ1_error_function(char *msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

/* computes the minimum a-expansion configuration */
void Match::KZ1_Expand(Coord a)
{
	Coord p, q, d, dq;
	Energy::Var var, qvar;
	int E_old, delta, E00, E0a, Ea0, Eaa;
	int k;

	Energy *e = new Energy(KZ1_error_function);

	/* initializing */
	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		if (a == d) IMREF(node_vars_left, p) = VAR_ACTIVE;
		else
		{
			IMREF(node_vars_left, p) = var = e -> add_variable();
			if (d.x == OCCLUDED) e -> ADD_TERM1(var, KZ1_OCCLUSION_PENALTY, 0);
		}

		d = Coord(IMREF(x_right, p), IMREF(y_right, p));
		if (a == -d) IMREF(node_vars_right, p) = VAR_ACTIVE;
		else
		{
			IMREF(node_vars_right, p) = var = e -> add_variable();
			if (d.x == OCCLUDED) e -> ADD_TERM1(var, KZ1_OCCLUSION_PENALTY, 0);
		}
	}

	for (p.y=0; p.y<im_size.y; p.y++)
	for (p.x=0; p.x<im_size.x; p.x++)
	{
		/* data and visibility terms */
		d = Coord(IMREF(x_left, p), IMREF(y_left, p));
		var = (Energy::Var) IMREF(node_vars_left, p);
		if (d != a && d.x != OCCLUDED)
		{
			q = p + d;
			if (q>=Coord(0,0) && q<im_size)
			{
				qvar = (Energy::Var) IMREF(node_vars_right, q);
				dq = Coord(IMREF(x_right, q), IMREF(y_right, q));
				if (d == -dq)
				{
					delta = (is_blocked(a, d)) ? INFINITY : 0;
					e -> ADD_TERM2(var, qvar, KZ1_data_penalty(p, q), delta, delta, 0);
				}
				else if (is_blocked(a, d))
				{
					e -> ADD_TERM2(var, qvar, 0, INFINITY, 0, 0);
				}
			}
		}

		q = p + a;
		if (q>=Coord(0,0) && q<im_size)
		{
			qvar = (Energy::Var) IMREF(node_vars_right, q);
			dq = Coord(IMREF(x_right, q), IMREF(y_right, q));

			E0a = (is_blocked(d, a)) ? INFINITY : 0;
			Ea0 = (is_blocked(-dq, a)) ? INFINITY : 0;
			Eaa = KZ1_data_penalty(p, q);

			if (var != VAR_ACTIVE)
			{
				if (qvar != VAR_ACTIVE) e -> ADD_TERM2(var, qvar, 0, E0a, Ea0, Eaa);
				else                    e -> ADD_TERM1(var, E0a, Eaa);
			}
			else
			{
				if (qvar != VAR_ACTIVE) e -> ADD_TERM1(qvar, Ea0, Eaa);
				else                    e -> add_constant(Eaa);
			}
		}

		/* left smoothness term */
		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];
			if ( ! ( q>=Coord(0,0) && q<im_size ) ) continue;
			qvar = (Energy::Var) IMREF(node_vars_left, q);
			dq = Coord(IMREF(x_left, q), IMREF(y_left, q));

			if (var != VAR_ACTIVE && qvar != VAR_ACTIVE)
				E00 = KZ1_smoothness_penalty_left(p, q, d, dq);
			if (var != VAR_ACTIVE)
				E0a = KZ1_smoothness_penalty_left(p, q, d, a);
			if (qvar != VAR_ACTIVE)
				Ea0 = KZ1_smoothness_penalty_left(p, q, a, dq);

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

		/* right smoothness term */
		d = Coord(IMREF(x_right, p), IMREF(y_right, p));
		var = (Energy::Var) IMREF(node_vars_right, p);
		for (k=0; k<NEIGHBOR_NUM; k++)
		{
			q = p + NEIGHBORS[k];
			if ( ! ( q>=Coord(0,0) && q<im_size ) ) continue;
			qvar = (Energy::Var) IMREF(node_vars_right, q);
			dq = Coord(IMREF(x_right, q), IMREF(y_right, q));

			if (var != VAR_ACTIVE && qvar != VAR_ACTIVE)
				E00 = KZ1_smoothness_penalty_right(p, q, d, dq);
			if (var != VAR_ACTIVE)
				E0a = KZ1_smoothness_penalty_right(p, q, d, -a);
			if (qvar != VAR_ACTIVE)
				Ea0 = KZ1_smoothness_penalty_right(p, q, -a, dq);

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

		/* visibility term */
		if (d.x != OCCLUDED && is_blocked(a, -d))
		{
			q = p + d;
			if (q>=Coord(0,0) && q<im_size)
			{
				if (d.x != -IMREF(x_left, q) || d.y != -IMREF(y_left, q))
					e -> ADD_TERM2(var, (Energy::Var) IMREF(node_vars_left, q),
					               0, INFINITY, 0, 0);
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
			var = (Energy::Var) IMREF(node_vars_left, p);

			if (var != VAR_ACTIVE && e->get_var(var)==VALUE1)
			{
				IMREF(x_left, p) = a.x; IMREF(y_left, p) = a.y;
			}

			var = (Energy::Var) IMREF(node_vars_right, p);

			if (var != VAR_ACTIVE && e->get_var(var)==VALUE1)
			{
				IMREF(x_right, p) = -a.x; IMREF(y_right, p) = -a.y;
			}
		}
	}

	delete e;
}

