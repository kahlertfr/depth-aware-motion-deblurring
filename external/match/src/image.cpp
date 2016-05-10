/* image.cpp */
/* Vladimir Kolmogorov (vnk@cs.cornell.edu), 2001. */

#include <stdio.h>
#include "image.h"

const int ONE = 1;
const int SWAP_BYTES = (((char *)(&ONE))[0] == 0) ? 1 : 0;

/************************************************************/
/************************************************************/
/************************************************************/

void * imNew(ImageType type, int xsize, int ysize)
{
	void *ptr;
	GeneralImage im;
	int data_size;
	int y;

	if (xsize<=0 || ysize<=0) return NULL;

	switch (type)
	{
		case IMAGE_BINARY: data_size = sizeof(unsigned char);    break;
		case IMAGE_GRAY:   data_size = sizeof(unsigned char);    break;
		case IMAGE_SHORT:  data_size = sizeof(short);            break;
		case IMAGE_RGB:    data_size = sizeof(unsigned char[3]); break;
		case IMAGE_LONG:   data_size = sizeof(long);             break;
		case IMAGE_PTR:    data_size = sizeof(void *);           break;
		case IMAGE_FLOAT:  data_size = sizeof(float);            break;
		case IMAGE_DOUBLE: data_size = sizeof(double);           break;
		default: return NULL;
	}

	ptr = malloc(sizeof(ImageHeader) + ysize*sizeof(void*));
	if (!ptr) return NULL;
	im = (GeneralImage) ((char*)ptr + sizeof(ImageHeader));

	imHeader(im) -> type      = type;
	imHeader(im) -> data_size = data_size;
	imHeader(im) -> xsize     = xsize;
	imHeader(im) -> ysize     = ysize;

	im->data = (void *) malloc(xsize*ysize*data_size);
	for (y=1; y<ysize; y++) (im+y)->data = ((char*)(im->data)) + xsize*y*data_size;

	return im;
}

/************************************************************/
/************************************************************/
/************************************************************/

inline int is_digit(unsigned char c) { return (c>='0' && c<='9'); }

void SwapBytes(GeneralImage im)
{
	if (SWAP_BYTES)
	{
		ImageType type = imHeader(im) -> type;

		if (type==IMAGE_SHORT || type==IMAGE_LONG || type==IMAGE_FLOAT || type==IMAGE_DOUBLE)
		{
			char *ptr, c;
			int i, k;

			int size = (imHeader(im) -> xsize) * (imHeader(im) -> ysize);
			int data_size = imHeader(im) -> data_size;

			ptr = (char *) (im->data);
			for (i=0; i<size; i++)
			{
				for (k=0; k<data_size/2; k++)
				{
					c = ptr[k];
					ptr[k] = ptr[data_size-k-1];
					ptr[data_size-k-1] = c;
				}
				ptr += data_size;
			}
		}
	}
}

void * imLoad(ImageType type, char *filename)
{
	FILE *fp;
	unsigned char LINE[70], c;
	int i, type_read = 0, num_read = 0, num_max = 3;
	int num[3];
	int xsize, ysize;
	int is_text = 0;
	int data_size;
	GeneralImage im;

	fp = fopen(filename, "rb");
	if (!fp) return NULL;

	while (fgets((char *)LINE, sizeof(LINE), fp))
	{
		i = 0;

		if (!type_read)
		{
			if (LINE[0] == 'P')
			{
				switch (LINE[1])
				{
					case '1': if (type != IMAGE_BINARY) { fclose(fp); return NULL; } is_text = 1; num_max = 2; break;
					case '2': if (type != IMAGE_GRAY)   { fclose(fp); return NULL; } is_text = 1; break;
					case '3': if (type != IMAGE_RGB)    { fclose(fp); return NULL; } is_text = 1; break;
					case '4': if (type != IMAGE_BINARY) { fclose(fp); return NULL; } num_max = 2;  break;
					case '5': if (type != IMAGE_GRAY)   { fclose(fp); return NULL; } break;
					case '6': if (type != IMAGE_RGB)    { fclose(fp); return NULL; } break;
					default: fclose(fp); return NULL;
				}
			}
			else if (LINE[0] == 'Q')
			{
				switch (LINE[1])
				{
					case '4': if (type != IMAGE_SHORT)  { fclose(fp); return NULL; } break;
					case '3': if (type != IMAGE_LONG)   { fclose(fp); return NULL; } break;
					case '1': if (type != IMAGE_FLOAT)  { fclose(fp); return NULL; } break;
					case '2': if (type != IMAGE_DOUBLE) { fclose(fp); return NULL; } break;
					default: fclose(fp); return NULL;
				}
				num_max = 2;
			}
			else { fclose(fp); return NULL; }
			if (is_digit(LINE[2])) { fclose(fp); return NULL; }
			i = 2;
			type_read = 1;
		}

		for (; c=LINE[i]; i++)
		{
			if (c == '#') break;
			if (is_digit(c))
			{
				if (num_read >= num_max) { fclose(fp); return NULL; }
				num[num_read] = c - '0';
				for (; c=LINE[i+1]; i++)
				{
					if (!is_digit(c)) break;
					num[num_read] = 10*num[num_read] + c - '0';
				}
				num_read ++;
			}
		}

		if (num_read >= num_max) break;
	}

	if (num_read != num_max) { fclose(fp); return NULL; }

	xsize = num[0];
	ysize = num[1];
	im = (GeneralImage) imNew(type, xsize, ysize);
	if (!im) { fclose(fp); return NULL; }
	data_size = imHeader(im) -> data_size;

	if (is_text)
	{
		num_read = 0;
		if (type == IMAGE_RGB) num_max = 3*xsize*ysize;
		else                   num_max = xsize*ysize;

		while (fgets((char *)LINE, sizeof(LINE), fp))
		{
			for (i=0; c=LINE[i]; i++)
			{
				if (c == '#') break;
				if (is_digit(c))
				{
					int tmp;
					if (num_read >= num_max) { imFree(im); fclose(fp); return NULL; }
					tmp = c - '0';
					for (; c=LINE[i+1]; i++)
					{
						if (!is_digit(c)) break;
						tmp = 10*tmp + c - '0';
					}
					if      (type == IMAGE_BINARY) imRef((BinaryImage)im, num_read, 0) = tmp;
					else if (type == IMAGE_GRAY)   imRef((GrayImage)im,   num_read, 0) = tmp;
					else
					{
						switch (num_read % 3)
						{
							case 0: imRef((RGBImage)im, num_read/3, 0).r = tmp; break;
							case 1: imRef((RGBImage)im, num_read/3, 0).g = tmp; break;
							case 2: imRef((RGBImage)im, num_read/3, 0).b = tmp; break;
						}
					}
					num_read ++;
				}
			}
		}

		if (num_read != num_max) { imFree(im); fclose(fp); return NULL; }
	}
	else
	{
		if (type == IMAGE_BINARY)
		{
			int s = (xsize*ysize+7)/8;
			unsigned char *b = (unsigned char *) malloc(s);
			if (!b) { imFree(im); fclose(fp); return NULL; }
			if (fread(b, 1, s, fp) != (size_t)s) { free(b); imFree(im); fclose(fp); return NULL; }
			for (i=0; i<xsize*ysize; i++)
			{
				((BinaryImage)im)->data[i] = ( b[i/8] & (1<<(7-(i%8))) ) ? 1 : 0;
			}
			free(b);
		}
		else
		{
			if (fread(im->data, data_size, xsize*ysize, fp) != (size_t)(xsize*ysize)) { imFree(im); fclose(fp); return NULL; }
		}
		SwapBytes(im);
	}

	fclose(fp);

	return im;
}

/************************************************************/
/************************************************************/
/************************************************************/

int imSave(void *im, char *filename)
{
	int i;
	FILE *fp;
	int im_max = 0;
	ImageType type = imHeader(im) -> type;
	int xsize = imHeader(im) -> xsize, ysize = imHeader(im) -> ysize;
	int data_size = imHeader(im) -> data_size;

	fp = fopen(filename, "wb");
	if (!fp) return -1;

	switch (type)
	{
		case IMAGE_BINARY:
			fprintf(fp, "P4\n%d %d\n", xsize, ysize);
			break;
		case IMAGE_GRAY:
			for (i=0; i<xsize*ysize; i++)
			if (im_max < ((GrayImage)im)->data[i]) im_max = ((GrayImage)im)->data[i];
			fprintf(fp, "P5\n%d %d\n%d\n", xsize, ysize, im_max);
			break;
		case IMAGE_RGB:
			for (i=0; i<xsize*ysize; i++)
			{
				if (im_max < ((RGBImage)im)->data[i].r) im_max = ((RGBImage)im)->data[i].r;
				if (im_max < ((RGBImage)im)->data[i].g) im_max = ((RGBImage)im)->data[i].g;
				if (im_max < ((RGBImage)im)->data[i].b) im_max = ((RGBImage)im)->data[i].b;
			}
			fprintf(fp, "P6\n%d %d\n%d\n", xsize, ysize, im_max);
			break;
		case IMAGE_SHORT:
			fprintf(fp, "Q4\n%d %d\n", xsize, ysize);
			break;
		case IMAGE_LONG:
			fprintf(fp, "Q3\n%d %d\n", xsize, ysize);
			break;
		case IMAGE_FLOAT:
			fprintf(fp, "Q1\n%d %d\n", xsize, ysize);
			break;
		case IMAGE_DOUBLE:
			fprintf(fp, "Q2\n%d %d\n", xsize, ysize);
			break;
		default:
			fclose(fp);
			return -1;
	}

	if (type == IMAGE_BINARY)
	{
		int s = (xsize*ysize+7)/8;
		unsigned char *b = (unsigned char *) malloc(s);
		if (!b) { fclose(fp); return -1; }
		for (i=0; i<s; i++) b[i] = 0;
		for (i=0; i<xsize*ysize; i++)
		{
			if (((BinaryImage)im)->data[i])
				b[i/8] ^= (1<<(7-(i%8)));
		}
		i = fwrite(b, 1, s, fp);
		free(b);
		if (i != s) { fclose(fp); return -1; }
	}
	else
	{
		SwapBytes((GeneralImage)im);
		i = fwrite(((GeneralImage)im)->data, data_size, xsize*ysize, fp);
		SwapBytes((GeneralImage)im);
		if (i != xsize*ysize) { fclose(fp); return -1; }
	}

	fclose(fp);
	return 0;
}


