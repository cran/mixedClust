#include "Distribution.h"


Distribution::Distribution(mat& xsep, int kr, int kc, int nbSEM)
{
	this->_nbSEM = nbSEM;
	this->_xsep = xsep;
	this->_N = xsep.n_rows;
	this->_J = xsep.n_cols;
	vector<vector<int>> miss_tmp;
	for (int i = 0; i < this->_N; i++)
	{
		for (int j = 0; j < this->_J; j++)
		{
			if (isnan(xsep(i, j)))  {
				vector<int> coordinates;
				coordinates.push_back(i);
				coordinates.push_back(j);
				miss_tmp.push_back(coordinates);
			}
		}
	}
	this->_miss = miss_tmp;
	this->_kr = kr;
	this->_kc = kc;

	// random generator for when we are going to sample partitions
	random_device _rd;

}

Distribution::Distribution()
{
}


Distribution::~Distribution()
{
}

void Distribution::missingValuesInit() {
	return;
}

TabProbsResults Distribution::SEstep(const mat& V, const mat& W)
{
	TabProbsResults result(_N, _kr, _J, _kc);
	return result;
}

mat Distribution::SEstepRow(const mat& W){
	mat result(_N, _kr);
	return(result);
}

mat Distribution::SEstepRowRandomParamsInit(mat& Wsamples, uvec& colSamples){
	mat result(_N, _kr);
	return(result);
}

mat Distribution::SEstepCol(const mat& V){
	mat result(_J, _kc);
	return(result);
}

LogProbs Distribution::SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec)
{
	LogProbs result(0, 0);
	return result;
}

void Distribution::imputeMissingData(const mat& V, const mat& W) {
	return;
}


void Distribution::Mstep(uvec rowind, uvec colind, int k, int h, bool init)
{
	return;
}

void Distribution::MstepVW(const mat& V, const mat& W, bool init)
{
	return;
}

rowvec Distribution::getDatablockkh(uvec rowind, uvec colind) {
	// function that exists so that we can get a vector of values for block kh
	// furthermore, we want to get rid of missing values:
	mat datablock_kh = _xsep.submat(rowind, colind);
	
	rowvec result = conv_to<rowvec>::from(vectorise(datablock_kh));
	// TODO: find a way for missing values:
	uvec todelete = find(result == -1);
	for (int i = 0; i < todelete.n_elem; i++) {
		result.shed_col(todelete(i));
	}
	return result;
}

void Distribution::fillParameters(int iteration) {
	return;
}

void Distribution::getBurnedParameters(int burn) {
	return;
}

void Distribution::printResults() {
	return;
}

List Distribution::returnResults() {
	List x;
	return(x);
}

List Distribution::returnParamsChain() {
	List x;
	return(x);
}

void Distribution::putParamsToZero() {
	return;
}

double Distribution::computeICL(int i, int d, int k, int h) {
	return(0);
}
bool Distribution::verif(const mat& V, const mat& W, int nbindmini) {
	bool result = true;
	for (int k = 0; k < _kr; k++) {
		for (int h = 0; h < _kc; h++) {
			uvec indicesV = arma::find(V.col(k) == 1);
			uvec indicesW = arma::find(W.col(h) == 1);
			int size = indicesV.n_elem * indicesW.n_elem;
			if (size < nbindmini) {
				return false;
			}
		}
	}

	return result;
}
int Distribution::verification(const mat& V, const mat& W, int nbindmini) {
	int result = -1;
	for (int k = 0; k < _kr; k++) {
		for (int h = 0; h < _kc; h++) {
			uvec indicesV = arma::find(V.col(k) == 1);
			uvec indicesW = arma::find(W.col(h) == 1);
			int size = indicesV.n_elem * indicesW.n_elem;
			if (size < nbindmini) {
				
				if(indicesV.n_elem<indicesW.n_elem){
					return (-2);
				}
				else{
					return h;
				}
			}
		}
	}

	return result;
}

void Distribution::initParams(uvec rowSample, uvec colSample, mat Vsample, mat Wsample){
	mat xsample = _xsep.submat(rowSample, colSample);
	this->MstepInitRandomParams(xsample, Vsample, Wsample);

}

void Distribution::MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample){
	return;
}

mat Distribution::returnXhat(){
	return(this->_xsep);
}


mat Distribution::colkmeans() {
	mat result(_J, _kc);
	result.zeros();

	mat colmeans;

	bool status = arma::kmeans(colmeans, _xsep, _kc, random_subset, 2, false);
	if (status == false)
	{
		return result;
	}

	for (int d = 0; d < _J; d++) {
		int num_clust = -1;
		double dst_old = -1;
		double dst = -1;

		for (int h = 0; h < _kc; h++) {
			vec a(_N);
			vec b(_N);
			for (int ireconstruct = 0; ireconstruct < _N; ireconstruct++) {
				a(ireconstruct) = colmeans.col(h)(ireconstruct);
				b(ireconstruct) = _xsep.col(d)(ireconstruct);
			}
			dst = this->getDistance(a, b);
			if (dst_old < 0 || dst < dst_old) {
				dst_old = dst;
				num_clust = h;
			}
		}
		result(d, num_clust) = 1;
	}
	return result;
}

double Distribution::getDistance(vec &a, vec &b) {
	vec temp = a - b;
	return arma::norm(temp);

}
