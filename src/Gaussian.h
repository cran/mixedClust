#pragma once
#include "Distribution.h"
#define _USE_MATH_DEFINES
#include <math.h>

class Gaussian :
	public Distribution
{
public:
	Gaussian(mat& xsep, int kr, int kc, int nbSEM);
	void missingValuesInit();
	TabProbsResults SEstep(const mat& V, const mat& W);
	mat SEstepRow(const mat& W);
	mat SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample);
	mat SEstepCol(const mat& V);
	LogProbs SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec);	
	void imputeMissingData(const mat& V, const mat& W);
	void Mstep(uvec rowind, uvec colind, int k, int h, bool init);
	void MstepVW(const mat& V, const mat& W, bool init);
	void fillParameters(int iteration);
	void getBurnedParameters(int burn);
	void printResults();
	List returnResults();
	List returnParamsChain();
	void putParamsToZero();
	double computeICL(int i, int d, int k, int h);
	void MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample);
	Gaussian();
	~Gaussian();
protected:
	mat _sigmas;
	mat _mus;
	mat _ressigmas;
	mat _resmus;
	cube _allsigmas;
	cube _allmus;
};
