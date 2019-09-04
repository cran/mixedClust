#pragma once
#include "Distribution.h"

// [[Rcpp::depends(RcppArmadillo)]] 
#include <armadillo>
//#include <limits>
//#include <cmath>


class Poisson :
	public Distribution
{
public:
	Poisson(mat& xsep, int kr, int kc, int nbSEM);
	Poisson();
	~Poisson();
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
protected:
	vec _mus;
	vec _nus;
	mat _gammas;
	mat _constant;
	mat _musnus;
	//mat _lambdas;
	vec _resmus;
	vec _resnus;
	mat _resgammas;
	//mat _reslambdas;
	mat _allmus;
	mat _allnus;
	cube _allgammas;
	//cube _alllambdas;
	double factorial(int n);
	double logfactorial(int n);
};

