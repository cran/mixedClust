#pragma once
#include "LogProbs.h"
#include "TabProbsResults.h"


// [[Rcpp::depends(RcppArmadillo)]] 
//#include <armadillo>
#include <limits>
//#include <cmath>
#include <list>
#include <random>
#include <vector>

#include <RcppArmadillo.h>

using namespace std;
using namespace arma;
using namespace Rcpp;


class Distribution
{
public:
	explicit Distribution(mat& xsep, int kr, int kc, int nbSEM);
	Distribution();
	virtual ~Distribution();
	virtual void missingValuesInit();
	virtual TabProbsResults SEstep(const mat& V, const mat& W);
	virtual mat SEstepRow(const mat& W);
	virtual mat SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample);
	virtual mat SEstepCol(const mat& V);
	virtual LogProbs SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec);
	virtual void imputeMissingData(const mat& V, const mat& W);
	virtual void Mstep(uvec rowind, uvec colind, int k, int h, bool init);
	virtual void MstepVW(const mat& V, const mat& W, bool init);
	virtual void fillParameters(int iteration);
	virtual void getBurnedParameters(int burn);
	virtual void printResults();
	virtual List returnResults();
	virtual List returnParamsChain();
	virtual void putParamsToZero();
	virtual double computeICL(int i, int d, int k, int h);
	bool verif(const mat& V, const mat& W, int nbindmini);
	mat colkmeans();
	double getDistance(vec &a, vec &b);
	mat returnXhat();
	void initParams(uvec rowSample, uvec colSample, mat Vsample, mat Wsample);
	virtual void MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample);

	int verification(const mat& V, const mat& W, int nbindmini);
protected:
	rowvec getDatablockkh(uvec rowind, uvec colind);
	string _name;
	mat _xsep;
	vector<vector<int>> _miss;
	int _Nr;
	int _Jc;
	int _kr;
	int _kc;
	int _nbSEM;
	random_device _rd; // to sample distributions
};

