/**********************************************************************************
*
*					        GaussianPDF
*					   by Hu yangyang 2016/1/12
*
***********************************************************************************/

#include "GaussianFitter.h"

// GaussianFitter functions
GaussianFitter::GaussianFitter()
{
	s = Color();

	p[0][0] = 0; p[0][1] = 0; p[0][2] = 0;
	p[1][0] = 0; p[1][1] = 0; p[1][2] = 0;
	p[2][0] = 0; p[2][1] = 0; p[2][2] = 0;

	count = 0;
}

// Add a color sample
void GaussianFitter::add(Color c)
{
	s.r += c.r; s.g += c.g; s.b += c.b;

	p[0][0] += c.r*c.r; p[0][1] += c.r*c.g; p[0][2] += c.r*c.b;
	p[1][0] += c.g*c.r; p[1][1] += c.g*c.g; p[1][2] += c.g*c.b;
	p[2][0] += c.b*c.r; p[2][1] += c.b*c.g; p[2][2] += c.b*c.b;

	count++;
}

// Build the gaussian out of all the added colors
void GaussianFitter::finalize(GaussianPDF& g, unsigned int totalCount) const
{
	// Running into a singular covariance matrix is problematic. So we'll add a small epsilon
	// value to the diagonal elements to ensure a positive definite covariance matrix.
	const Real Epsilon = (Real)0.0001;

	if (count==0)
	{
		g.pi = 0;
	}
	else
	{
		// Compute mean of gaussian
		g.mu.r = s.r/count;
		g.mu.g = s.g/count;
		g.mu.b = s.b/count;

		// Compute covariance matrix
		g.covariance[0][0] = p[0][0]/count-g.mu.r*g.mu.r + Epsilon; g.covariance[0][1] = p[0][1]/count-g.mu.r*g.mu.g; g.covariance[0][2] = p[0][2]/count-g.mu.r*g.mu.b;
		g.covariance[1][0] = p[1][0]/count-g.mu.g*g.mu.r; g.covariance[1][1] = p[1][1]/count-g.mu.g*g.mu.g + Epsilon; g.covariance[1][2] = p[1][2]/count-g.mu.g*g.mu.b;
		g.covariance[2][0] = p[2][0]/count-g.mu.b*g.mu.r; g.covariance[2][1] = p[2][1]/count-g.mu.b*g.mu.g; g.covariance[2][2] = p[2][2]/count-g.mu.b*g.mu.b + Epsilon;

		// Compute determinant of covariance matrix
		g.determinant = g.covariance[0][0]*(g.covariance[1][1]*g.covariance[2][2]-g.covariance[1][2]*g.covariance[2][1]) 
			- g.covariance[0][1]*(g.covariance[1][0]*g.covariance[2][2]-g.covariance[1][2]*g.covariance[2][0]) 
			+ g.covariance[0][2]*(g.covariance[1][0]*g.covariance[2][1]-g.covariance[1][1]*g.covariance[2][0]);

		// Compute inverse (cofactor matrix divided by determinant)
		g.inverse[0][0] =  (g.covariance[1][1]*g.covariance[2][2] - g.covariance[1][2]*g.covariance[2][1]) / g.determinant;
		g.inverse[1][0] = -(g.covariance[1][0]*g.covariance[2][2] - g.covariance[1][2]*g.covariance[2][0]) / g.determinant;
		g.inverse[2][0] =  (g.covariance[1][0]*g.covariance[2][1] - g.covariance[1][1]*g.covariance[2][0]) / g.determinant;
		g.inverse[0][1] = -(g.covariance[0][1]*g.covariance[2][2] - g.covariance[0][2]*g.covariance[2][1]) / g.determinant;
		g.inverse[1][1] =  (g.covariance[0][0]*g.covariance[2][2] - g.covariance[0][2]*g.covariance[2][0]) / g.determinant;
		g.inverse[2][1] = -(g.covariance[0][0]*g.covariance[2][1] - g.covariance[0][1]*g.covariance[2][0]) / g.determinant;
		g.inverse[0][2] =  (g.covariance[0][1]*g.covariance[1][2] - g.covariance[0][2]*g.covariance[1][1]) / g.determinant;
		g.inverse[1][2] = -(g.covariance[0][0]*g.covariance[1][2] - g.covariance[0][2]*g.covariance[1][0]) / g.determinant;
		g.inverse[2][2] =  (g.covariance[0][0]*g.covariance[1][1] - g.covariance[0][1]*g.covariance[1][0]) / g.determinant;

		// The weight of the gaussian is the fraction of the number of pixels in this Gaussian to the number of 
		// pixels in all the gaussians of this GMM.
		g.pi = (Real)count/totalCount;


	}
}