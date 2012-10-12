#pragma once
/*
#include <functional>
#include "glm/glm.hpp"
using namespace glm;
using namespace std;

template<int NumThetaBins, int NumPhiBins>
float validate_importance_sampling(
	function<direction<WorldCS>(RandomPair, InversePdf*)> rand_to_xyz,
	function<RandomPair(void)> generate_rand_nums_func,
	function<float(NormalizedSphericalCS)> invPdfFunc,
	long num_samples)
{
	const float ThetaRange = PI / 2 - 0;
	const float PhiRange = 2 * PI;
	long long bins[NumThetaBins][NumPhiBins];
	double bins_pdf[NumThetaBins][NumPhiBins];
	for(int i = 0; i < NumThetaBins; i++)
	{
		for(int j = 0; j < NumPhiBins; j++)
		{
			bins[i][j] = 0;
			bins_pdf[i][j] = 0;
		}
	}
	for(int sample_idx = 0; sample_idx < num_samples; sample_idx++)
	{
		InversePdf inv_pdf;
		auto theta_phi = spherical(rand_to_xyz(generate_rand_nums_func(), &inv_pdf));
		theta_phi.y += PI; //[-pi, pi] to [0, 2pi]
		int theta_bin_idx = floor((theta_phi.x / ThetaRange) * NumThetaBins);
		int phi_bin_idx = floor((theta_phi.y / PhiRange) * NumPhiBins);
		assert(theta_bin_idx < NumThetaBins && theta_bin_idx >= 0);
		assert(phi_bin_idx < NumPhiBins && phi_bin_idx >= 0);
		assert(1.f/inv_pdf > 0);
		double this_ratio = 1/(++bins[theta_bin_idx][phi_bin_idx]);
		double new_bin_pdf = 
			this_ratio * (1.f/inv_pdf) 
			+ (1-this_ratio) * (bins_pdf[theta_bin_idx][phi_bin_idx]);
		bins_pdf[theta_bin_idx][phi_bin_idx] = new_bin_pdf;
		assert(sample_idx >= 0);
	}
	double pdf_sum = 0;
	double chi_squared = 0;

	const float INTEGRATION_DIVISIONS = 20;
	const float integration_step_size = 1.f / INTEGRATION_DIVISIONS;
	const float half_integ_step = integration_step_size / 2;
	for(int i = 0; i < NumThetaBins; i++)
	{
		for(int j = 0; j < NumPhiBins; j++)
		{
			float integratedPdf = 0;
			//integrate pdf
			for(int k = 0; k < INTEGRATION_DIVISIONS; k++)
			{
				for(int l = 0; l < INTEGRATION_DIVISIONS; l++)
				{
					float theta = (k / INTEGRATION_DIVISIONS + half_integ_step + i) 
						/ NumThetaBins * ThetaRange;
					float phi = (l / INTEGRATION_DIVISIONS + half_integ_step + j) 
						/ NumPhiBins * PhiRange;
					float pdf = 1.f / invPdfFunc(NormalizedSphericalCS(theta, phi));
					float area = (1.f / NumThetaBins * ThetaRange) 
						* (1.f / NumPhiBins * PhiRange);
					integratedPdf += pdf * area * sin(theta);
				}
			}
			integratedPdf /= (float)(INTEGRATION_DIVISIONS * INTEGRATION_DIVISIONS);
			float elevation_scale = sin((double)(i + 0.5f) / NumThetaBins * ThetaRange);
			pdf_sum += integratedPdf;
			int actual = bins[i][j];
			assert(integratedPdf > 0 || actual == 0);
			if(integratedPdf > 0)
			{
				int expected = (int)floor(integratedPdf * num_samples);
				int diff = actual - expected;
				chi_squared += (diff * diff) / expected;
			}
		}
	}
	int degrees_of_freedom = NumThetaBins * NumPhiBins;
	printf("dof = %d, chi_squared = %f\n", degrees_of_freedom, chi_squared);

	printf("pdf sum = %f\n", pdf_sum);
	assert(close_to(pdf_sum, 1, .0001));
	assert(chi_squared < 300);
	return 0;
}
*/
