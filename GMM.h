/*
 * GrabCut implementation source code Copyright(c) 2005-2006 Justin Talbot
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 */

/**********************************************************************************
*
*					          GMM
*					   by Hu yangyang 2016/1/12
*
***********************************************************************************/

#ifndef GMM_H
#define GMM_H

#include "Color.h"
#include "GaussianFitter.h"

#include <vector>
using namespace std;

typedef unsigned int uint;


class GMM
{
public:

	// Initialize GMM with number of gaussians desired.
	GMM(unsigned int K);
	~GMM();

	unsigned int K() const { return mK; }

	// Returns the probability density of color c in this GMM
	Real p(Color c);

	// Returns the probability density of color c in just Gaussian k
	Real p(unsigned int i, Color c);

	int Build(double** data, uint nrows);

private:

	unsigned int mK;		// number of gaussians
	GaussianPDF* mGaussians;	// an array of K gaussians
};

#endif //GMM_H