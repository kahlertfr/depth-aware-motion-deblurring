/* match.h */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001-2003. */

#ifndef __MATCH_H__
#define __MATCH_H__

#include "image.h"

struct Coord
{
	int x, y;

	Coord() {}
	Coord(int a, int b) { x = a; y = b; }

	Coord operator- ()        { return Coord(-x, -y); }
	Coord operator+ (Coord a) { return Coord(x + a.x, y + a.y); }
	Coord operator- (Coord a) { return Coord(x - a.x, y - a.y); }
	bool  operator< (Coord a) { return (x <  a.x) && (y <  a.y); }
	bool  operator<=(Coord a) { return (x <= a.x) && (y <= a.y); }
	bool  operator> (Coord a) { return (x >  a.x) && (y >  a.y); }
	bool  operator>=(Coord a) { return (x >= a.x) && (y >= a.y); }
	bool  operator==(Coord a) { return (x == a.x) && (y == a.y); }
	bool  operator!=(Coord a) { return (x != a.x) || (y != a.y); }
};
#define IMREF(im, p) (imRef((im), (p).x, (p).y))




/* (half of) the neighborhood system
   the full neighborhood system is edges in NEIGHBORS
   plus reversed edges in NEIGHBORS */
const struct Coord NEIGHBORS[] = { Coord(1, 0), Coord(0, -1) };
#define NEIGHBOR_NUM (sizeof(NEIGHBORS) / sizeof(Coord))




class Match
{
public:
	Match(char *name_left, char *name_right, bool color = false);
	~Match();

	/* load segmentation images */
	void LoadSegm(char *name_left, char *name_right, bool color);
	/* save disparity maps as .pgm images */
	void SaveXLeft(char *file_name, bool flag); /* if flag is TRUE then larger */
	void SaveYLeft(char *file_name, bool flag); /* disparities are brighter    */
	/* save disparity maps as scaled .ppm images */
	void SaveScaledXLeft(char *file_name, bool flag); /* if flag is TRUE then larger */
	void SaveScaledYLeft(char *file_name, bool flag); /* disparities are brighter    */
	/* load disparity maps as .pgm images */
	void LoadXLeft(char *file_name, bool flag); /* if flag is TRUE then larger */
	void LoadYLeft(char *file_name, bool flag); /* disparities are brighter    */
	
	void SetDispRange(Coord disp_base, Coord disp_max);

	float GetK(); /* compute statistics of data_penalty */

	/* Parameters of KZ1, KZ2, BVZ and CORR algorithms. */
	/* Description is in the config file */
	struct Parameters
	{
		/********** data term for CORR, KZ1, KZ2, BVZ **********/
		/*
			if sub_pixel is true then the data term is computed as described in

			Stan Birchfield and Carlo Tomasi
			"A pixel dissimilarity measure that is insensitive to image sampling"
			PAMI 20(4):401-406, April 98

			with one distinction: intensity intervals for a pixels
			are computed from 4 neighbors rather than 2.
		*/
		bool			sub_pixel;
		enum { L1, L2 } data_cost;
		int				denominator; /* data term is multiplied by denominator.  */
									 /* Equivalent to using lambda1/denominator, */
									 /* lambda2/denominator, K/denominator       */

		/********** smoothness term for KZ1, KZ2, BVZ **********/
		int				I_threshold;  /* intensity threshold for KZ1 and BVZ */
		int				I_threshold2; /* intensity threshold for KZ2 */
		int				interaction_radius; /* 1 for Potts, >1 for truncated linear */
		int				lambda1, lambda2;

		/********** penalty for an assignment being inactive for KZ1, KZ2 **********/
		int				K;

		/********** occlusion penalty for BVZ (usually INFINITY) **********/
		int				occlusion_penalty;

		/********** iteration parameters for KZ1, KZ2, BVZ **********/
		int				iter_max;
		bool			randomize_every_iteration;

		/********** correlation window for CORR **********/
		int				corr_size;
	};
	void SetParameters(Parameters *params);


	/* algorithms */
	void CLEAR();
	void SWAP_IMAGES();
	void MAKE_UNIQUE();
	void CROSS_CHECK();
	void FILL_OCCLUSIONS();
	void CORR();
	void KZ1();
	void KZ2();
	void BVZ();








private:
	/************** BASIC DATA *****************/
	Coord			im_size;					/* image dimensions */
	GrayImage		im_left, im_right;			/* original images */
	RGBImage		im_color_left, im_color_right;	/* original color images */
	GrayImage		segm_left, segm_right;			/* segmentation images */
	RGBImage		segm_color_left, segm_color_right;	/* segmentation color images */
	GrayImage		im_left_min, im_left_max,	/* contain range of intensities */
					im_right_min, im_right_max; /* based on intensities of neighbors */
	RGBImage		im_color_left_min, im_color_left_max,
					im_color_right_min, im_color_right_max;
	Coord			disp_base, disp_max, disp_size;	/* range of disparities */
#define OCCLUDED 255
	LongImage		x_left, x_right,
					y_left, y_right;
	/*
		disparity map
		IMREF(x_..., p)==OCCLUDED means that 'p' is occluded
		if l - pixel in the left image, r - pixel in the right image, then
		r == l + Coord(IMREF(x_left, l), IMREF(y_left, l))
		l == r + Coord(IMREF(x_right, r), IMREF(y_right, r))
	*/
	bool			unique_flag;	/* true if current configuration is unique */
									/* (each pixel corresponds to at most one pixel in the other image */
	Parameters		params;

	/********* INTERNAL VARIABLES **************/
	int				E;					/* current energy */
	PtrImage		ptr_im1, ptr_im2;	/* used for storing variables corresponding to nodes */

	/********* INTERNAL FUNCTIONS **************/
	typedef enum
	{
		METHOD_KZ1,
		METHOD_KZ2,
		METHOD_BVZ
	} Method;
	void		Run_KZ_BVZ(Method method);
	void		InitSubPixel();
	void		SubPixel(GrayImage Im, GrayImage ImMin, GrayImage ImMax);
	void		SubPixelColor(RGBImage Im, RGBImage ImMin, RGBImage ImMax);

	/* data penalty functions for CORR, KZ1, KZ2, BVZ */
	int			data_penalty_GRAY(Coord l, Coord r);
	int			data_penalty_COLOR(Coord l, Coord r);
	int			data_penalty_SUBPIXEL_GRAY(Coord l, Coord r);
	int			data_penalty_SUBPIXEL_COLOR(Coord l, Coord r);

	/* smoothness penalty functions for KZ1, BVZ */
	int			smoothness_penalty_left_GRAY(Coord p, Coord np, Coord d, Coord nd);
	int			smoothness_penalty_left_COLOR(Coord p, Coord np, Coord d, Coord nd);
	int			smoothness_penalty_right_GRAY(Coord p, Coord np, Coord d, Coord nd);
	int			smoothness_penalty_right_COLOR(Coord p, Coord np, Coord d, Coord nd);

	/* smoothness penalty functions for KZ2 */
	int			smoothness_penalty2_GRAY(Coord p, Coord np, Coord d);
	int			smoothness_penalty2_COLOR(Coord p, Coord np, Coord d);
	
	/* pointers to correct data and smoothness penalty functions */
	int			(Match::*data_penalty_func)(Coord l, Coord r);
	int			(Match::*smoothness_penalty_left_func)(Coord p, Coord np, Coord d, Coord nd);
	int			(Match::*smoothness_penalty_right_func)(Coord p, Coord np, Coord d, Coord nd);
	int			(Match::*smoothness_penalty2_func)(Coord p, Coord np, Coord d);




	/*************** CORR ALGORITHM *************/
	int			CORR_data_penalty(Coord l, Coord r);
	void		CORR_hor(Coord d, LongImage v_hor);
	void		CORR_full(LongImage v_hor, LongImage v_full);





	/**************** KZ1 ALGORITHM *************/
	int			KZ1_data_penalty(Coord l, Coord r);
	int			KZ1_smoothness_penalty_left(Coord p, Coord np, Coord d, Coord nd);
	int			KZ1_smoothness_penalty_right(Coord p, Coord np, Coord d, Coord nd);
	int			KZ1_ComputeEnergy();			/* computes current energy */
	void		KZ1_Expand(Coord a);			/* computes the minimum a-expansion configuration */
	bool		KZ1_visibility;		/* defined only for stereo - then visibility constraint is enforced */





	/**************** KZ2 ALGORITHM *************/
	int			KZ2_data_penalty(Coord l, Coord r);
	int			KZ2_smoothness_penalty2(Coord p, Coord np, Coord d);
	int			KZ2_ComputeEnergy();			/* computes current energy */
	void		KZ2_Expand(Coord a);			/* computes the minimum a-expansion configuration */





	/**************** BVZ ALGORITHM *************/
	int			BVZ_data_penalty(Coord p, Coord d);
	int			BVZ_smoothness_penalty(Coord p, Coord np, Coord d, Coord nd);
	int			BVZ_ComputeEnergy();			/* computes current energy */
	void		BVZ_Expand(Coord a);			/* computes the minimum a-expansion configuration */
};

#define INFINITY 10000		/* infinite capacity */

#endif
