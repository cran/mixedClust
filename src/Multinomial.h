#pragma once
#include "Distribution.h"

// [[Rcpp::depends(RcppArmadillo)]] 
#include <armadillo>
#include <limits>
#include <cmath>
#include <list>
#include <math.h> 

using namespace arma;
using namespace std;

class Multinomial :
	public Distribution
{
public:
	Multinomial(mat& xsep, int kr, int kc, int m, int nbSEM);
	Multinomial();
	~Multinomial();
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
	void printAlphas();
protected:
	int _m;
	cube _alphas;
	vector<cube> _allalphas;
	cube _resalphas;
	vec getAlpha(rowvec block_kh);// maybe put on private

};
