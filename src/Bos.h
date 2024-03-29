
#pragma once
#include "Distribution.h"
#include "Mupi.h"

// [[Rcpp::depends(RcppArmadillo)]] 
//#include <armadillo>
//#include <limits>
//#include <cmath>


extern const double inf;
using namespace std;

class Bos :
	public Distribution
{
public:
	Bos(mat& xsep, int kr, int kc, int m, int nbSEM, unsigned int iterordiEM);
	Bos();
	~Bos();
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
	mat colkmeans();
	cube getCubeProbs();
	cube gettabpej();
	double logsum(rowvec logx);
	void printResults();
	List returnResults();
	List returnParamsChain();
	void putParamsToZero();
	double computeICL(int i, int d, int k, int h);
	void MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample);
protected:
	cube _xsepCube;
	cube _cubeProbs;
	int _m;
	mat _pis;
	umat _mus;
	cube _allpis;
	ucube _allmus;
	mat _respis;
	umat _resmus;
	unsigned int _iterordiEM;
	cube _tab_pejs;
	Mupi ordiemCpp(const arma::colvec& datablock_kh,
		const arma::colvec& tabmu0,
		const arma::colvec& tabp0,
		double eps,
		int iter_max);
	arma::umat allej(int j, int m);
	bool compare_vec(arma::urowvec vec1, arma::rowvec vec2);
	int unsigned_to_signed(unsigned x);
	double pejp1_yjej(arma::urowvec ejp1, int yj, arma::urowvec ej, int mu, double p);
	double pejp1zj1_yjej(arma::urowvec ejp1, unsigned int yj, arma::urowvec ej, int mu, double p);
	double pejp1zj1_ej(arma::urowvec ejp1, arma::urowvec ej, int mu, double p);
	double pyj_ej(unsigned int yj, arma::urowvec ej);
	double pejp1_ej(arma::urowvec ejp1, arma::urowvec ej, int mu, double p);
	double pej(arma::urowvec& ej, int j, int m, int mu, double p, arma::colvec& z1tozjm1);
	int getModeFromVec(uvec vector);
	void printMus();
	void printPis();
};

