#include "Gaussian.h"


Gaussian::Gaussian(mat& xsep, int kr, int kc, int nbSEM)
	:Distribution(xsep, kr, kc, nbSEM)
{
	this->_name = "Gaussian";
	this->_mus = zeros(_kr, _kc);
	this->_sigmas = zeros(_kr, _kc);
	this->_resmus = zeros(_kr, _kc);
	this->_ressigmas = zeros(_kr, _kc);
	this->_allmus = zeros(_kr, _kc, _nbSEM);
	this->_allsigmas = zeros(_kr, _kc, _nbSEM);
}
Gaussian::Gaussian()
{
}


Gaussian::~Gaussian()
{
}

void Gaussian::missingValuesInit() {
	for (int imiss = 0; imiss < _miss.size(); imiss++) {
		mt19937 gen(_rd());
		uniform_real_distribution<double> d(0.0,1.0);
		double sample = d(gen);
		_xsep(_miss.at(imiss)[0], _miss.at(imiss)[1]) = sample;
	}
	return;
}



TabProbsResults Gaussian::SEstep(const mat& V, const mat& W) {
	TabProbsResults result = TabProbsResults(_N, _kr, _J, _kc);
	for (int i = 0; i < _N; i++)
	{

		for (int k = 0; k < _kr; k++)
		{

			for (int d = 0; d < _J; d++)
			{

				for (int h = 0; h < _kc; h++)
				{
					double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2* M_PI))*std::exp(-0.5*(std::pow((_xsep(i,d)-_mus(k,h))/ _sigmas(k, h),2)));
					if (!(tocompute > 0)) {
						tocompute = 1e-300;
					}
					float density = (float)log(tocompute);
					
					result._tabprobaV(i, k) = result._tabprobaV(i, k) +
						W(d, h) * density;
					result._tabprobaW(d, h) = result._tabprobaW(d, h) +
						V(i, k) * density;

				}
			}
		}
	}
	return(result);
}

mat Gaussian::SEstepRow(const mat& W){
	mat result(_N, _kr);
	result.zeros();
	
	for (int d = 0; d < _J; d++)
	{
		for (int h = 0; h < _kc; h++)
		{

			if(W(d,h)==1){
				for (int i = 0; i < _N; i++)
				{

					for (int k = 0; k < _kr; k++)
					{
						double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2* M_PI))*std::exp(-0.5*(std::pow((_xsep(i,d)-_mus(k,h))/ _sigmas(k, h),2)));
						if (!(tocompute > 0)) {
							tocompute = 1e-300;
						}
						float density = (float)log(tocompute);
						result(i, k) +=  density;

					}
				}
			}
			
		}
	}


	return(result);
}

mat Gaussian::SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample){
	mat result(_N, _kr);
	result.zeros();

	mat xsample = _xsep.cols(colSample);

	
	for (int d = 0; d < Wsample.n_rows; d++)
	{
		for (int h = 0; h < _kc; h++)
		{

			if(Wsample(d,h)==1){
				for (int i = 0; i < _N; i++)
				{

					for (int k = 0; k < _kr; k++)
					{
						double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2* M_PI))*std::exp(-0.5*(std::pow((xsample(i,d)-_mus(k,h))/ _sigmas(k, h),2)));
						if (!(tocompute > 0)) {
							tocompute = 1e-300;
						}
						float density = (float)log(tocompute);
						result(i, k) +=  density;

					}
				}
			}
			
		}
	}

	return result;
}

mat Gaussian::SEstepCol(const mat& V) {
	mat result(_J, _kc);
	result.zeros();
	for (int i = 0; i < _N; i++)
	{

		for (int k = 0; k < _kr; k++)
		{
			if(V(i,k)==1){
				for (int d = 0; d < _J; d++)
				{

					for (int h = 0; h < _kc; h++)
					{
						double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2* M_PI))*std::exp(-0.5*(std::pow((_xsep(i,d)-_mus(k,h))/ _sigmas(k, h),2)));
						if (!(tocompute > 0)) {
							tocompute = 1e-300;
						}
						float density = (float)log(tocompute);
						result(d, h) +=  density;

					}
				}
			}
		}
	}
	return(result);
}

LogProbs Gaussian::SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec) {
	
	LogProbs result(0,0);
	
	double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2 * M_PI))*std::exp(-0.5*(std::pow((x_id - _mus(k, h)) / _sigmas(k, h), 2)));
	if (!(tocompute > 0)) {
		tocompute = 1e-300;
	}
	float density = (float)log(tocompute);

	result._row = density;
	result._col = density;


	return(result);

}

void Gaussian::imputeMissingData(const mat& V, const mat& W) {

	for (int imiss = 0; imiss < _miss.size(); imiss++) {

		vector<int> coordinates = _miss.at(imiss);
		int miss_row = coordinates.at(0);
		uvec k = arma::find(V.row(miss_row) == 1);
		int miss_col = coordinates.at(1);
		uvec h = arma::find(W.row(miss_col) == 1);

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(_sigmas(k(0), h(0)), _mus(k(0), h(0)));
		double sample = distribution(generator);
		this->_xsep(miss_row, miss_col) = sample;


	}
}

void Gaussian::Mstep(uvec rowind, uvec colind, int k, int h, bool init) {
	rowvec datablock_kh = this->getDatablockkh(rowind, colind);
	_mus(k, h) = mean(datablock_kh);
	_sigmas(k, h) = stddev(datablock_kh);

	//_mus.print();
	return;
}

void Gaussian::MstepVW(const mat& V, const mat& W, bool init) {
	
	for (int k = 0; k < _kr; k++)
	{
		for (int h = 0; h < _kc; h++) {
			uvec rowind = find(V.col(k) == 1);
			uvec colind = find(W.col(h) == 1);
			mat datablock_kh = _xsep.submat(rowind, colind);
			if(datablock_kh.n_elem){
				_mus(k, h) = mean(vectorise(datablock_kh));
				_sigmas(k, h) = stddev(vectorise(datablock_kh));
			}
			
		}
	}
	return;
}

void Gaussian::MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample){
	for (int k = 0; k < _kr; k++)
	{
		for (int h = 0; h < _kc; h++) {
			uvec rowind = find(Vsample.col(k) == 1);
			uvec colind = find(Wsample.col(h) == 1);
			mat datablock_kh = xsample.submat(rowind, colind);
			
			_mus(k, h) = mean(vectorise(datablock_kh));
			_sigmas(k, h) = stddev(vectorise(datablock_kh));
		}
	}
	return;
}

void Gaussian::fillParameters(int iteration) {
	this->_allmus.slice(iteration) = this->_mus;
	this->_allsigmas.slice(iteration) = this->_sigmas;
	return;
}

void Gaussian::getBurnedParameters(int burn) {
	this->_resmus = mean(this->_allmus.slices(burn, _nbSEM - 1), 2);
	this->_ressigmas = mean(this->_allsigmas.slices(burn, _nbSEM - 1), 2);
	this->_mus = this->_resmus;
	this->_sigmas = this->_ressigmas;

	return;
}

void Gaussian::printResults() {
	this->_sigmas.print();
	this->_mus.print();
	return;
}

List Gaussian::returnResults() {
	List gaussianRes = List::create(Rcpp::Named("sigmas") = this->_sigmas,
									Rcpp::Named("mus") = this->_mus);
	return(gaussianRes);
}

List Gaussian::returnParamsChain() {
	List gaussianRes = List::create(Rcpp::Named("sigmas") = this->_allsigmas,
									Rcpp::Named("mus") = this->_allmus);
	return(gaussianRes);
}

void Gaussian::putParamsToZero() {
	return;
}

double Gaussian::computeICL(int i, int d, int k, int h) {
	double result = 0;
	if(i==0 && d==0 && k==0 && h==0){
		// did not divided by two because there are two parameter mu and sigma
		result = - _kc*_kr * log(_N*_J); 
	}
	double tocompute = 1 / (_sigmas(k, h)*std::sqrt(2* M_PI))*std::exp(-0.5*(std::pow((_xsep(i,d)-_mus(k,h))/ _sigmas(k, h),2)));
	if (!(tocompute > 0)) {
		tocompute = 1e-300;
	}
	double density = log(tocompute);
	result += density;
	return(result);
}

