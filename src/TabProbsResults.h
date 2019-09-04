#pragma once

// [[Rcpp::depends(RcppArmadillo)]] 
//#include <armadillo>

#include <RcppArmadillo.h>

//using namespace std;
using namespace arma;

class TabProbsResults
{
public:
	TabProbsResults(int N, int kr, int J, int kc);
	~TabProbsResults();
	mat _tabprobaV;
	mat _tabprobaW;
};

