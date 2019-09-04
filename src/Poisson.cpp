#include "Poisson.h"


Poisson::Poisson(mat& xsep, int kr, int kc, int nbSEM)
	:Distribution(xsep, kr, kc, nbSEM)
{
	this->_name = "Poisson";
	this->_mus = zeros(_N);
	this->_nus = zeros(_J);
	this->_gammas = zeros(_kr, _kc);
	this->_resmus = zeros(_N);
	this->_resnus = zeros(_J);
	this->_resgammas = zeros(_kr, _kc);
	this->_allmus = zeros(_N, _nbSEM);
	this->_allnus = zeros(_J, _nbSEM);
	this->_allgammas = zeros(_kr, _kc, _nbSEM);

	this->missingValuesInit();

	_mus = conv_to<vec>::from(sum(_xsep, 1));
	_nus = conv_to<vec>::from(sum(_xsep, 0));

	this->_constant = zeros(_N, _J);
	_constant.zeros();
	this->_musnus = zeros(_N, _J);
	_musnus.zeros();
	for (size_t i = 0; i < _N; i++)
	{
		for (size_t d = 0; d < _J; d++)
		{
			_constant(i,d) += (_xsep(i, d) * log(_mus(i) * _nus(d)) - logfactorial(_xsep(i, d)));
			_musnus(i,d) = _mus(i) * _nus(d);
		}
	}

}

Poisson::Poisson()
{
}


Poisson::~Poisson()
{
}

void Poisson::missingValuesInit() {
	for (int imiss = 0; imiss < _miss.size(); imiss++) {
		mt19937 gen(_rd());
		double eqprob = (double)1 / 5;
		vec vecprob(5, fill::ones);
		vecprob = vecprob*eqprob;
		discrete_distribution<> d(vecprob.begin(), vecprob.end()); // maybe a problem?
		int sample = d(gen);
		_xsep(_miss.at(imiss)[0], _miss.at(imiss)[1]) = sample + 1;
	}
}


TabProbsResults Poisson::SEstep(const mat& V, const mat& W) {
	TabProbsResults result = TabProbsResults(_N, _kr, _J, _kc);

	for (size_t i = 0; i < _N; i++)
	{

		for (size_t k = 0; k < _kr; k++)
		{
			
			for (size_t d = 0; d < _J; d++)
			{

				for (size_t h = 0; h < _kc; h++)
				{
					
					double density1 = -_mus(i) * _nus(d) * _gammas(k, h) + _xsep(i, d) * log(_mus(i) * _nus(d) * _gammas(k, h)) - logfactorial(_xsep(i, d));
					result._tabprobaV(i, k) = result._tabprobaV(i, k) +
						W(d, h) * density1;
					result._tabprobaW(d, h) = result._tabprobaW(d, h) +
						V(i, k) * density1;
				}
			}
		}
	}
	return result;
}

mat Poisson::SEstepRow(const mat& W) {
	mat result(_N, _kr);
	result.zeros();
	result = -( _musnus * W ) * _gammas.t() + ( _xsep * W ) * (log(_gammas)).t();
	mat toadd =  (_constant * W);
	// TODO change that! so taht there is no for loop
	for (size_t i = 0; i < _N; i++)
	{
		result.row(i)  += sum(toadd.row(i));
	}
	

	return result;
}


mat Poisson::SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample){
	mat result(_N, _kr);
	result.zeros();

	mat xsample = _xsep.cols(colSample);
	colvec mussamples = sum(xsample,1);

	
	for (int d = 0; d < Wsample.n_rows; d++)
	{
		for (int h = 0; h < _kc; h++)
		{

			if(Wsample(d,h)==1){
				for (int i = 0; i < _N; i++)
				{

					for (int k = 0; k < _kr; k++)
					{
						double density1 = -_mus(i) * _nus(colSample(d)) * _gammas(k, h) + xsample(i, d) * log(_mus(i) * _nus(colSample(d)) * _gammas(k, h)) - logfactorial(xsample(i, d));
						result(i, k) += density1;

					}
				}
			}
			
		}
	}

	return result;
}

mat Poisson::SEstepCol(const mat& V) {
	mat result(_J, _kc);
	result.zeros();

	/*for (size_t k = 0; k < _kr; k++)
	{
		for (size_t i = 0; i < _N; i++)
		{
			if(V(i, k)==1){
				for (size_t d = 0; d < _J; d++)
				{
					for (size_t h = 0; h < _kc; h++)
					{
						double density1 = -_mus(i) * _nus(d) * _gammas(k, h) + _xsep(i, d) * log(_mus(i) * _nus(d) * _gammas(k, h)) - logfactorial(_xsep(i, d));
						result(d, h) += density1;
					}
				}
			}
		}
	}*/

	result = -( _musnus.t() * V ) * _gammas + ( _xsep.t() * V ) * (log(_gammas));
	mat toadd =  (_constant.t() * V);
	// TODO change that! so taht there is no for loop
	for (size_t i = 0; i < _J; i++)
	{
		result.row(i)  += sum(toadd.row(i));
	}

	return result;
}

LogProbs Poisson::SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec) {
	LogProbs result(0, 0);

	double density1 = -sumi * sumd * _gammas(k, h) + x_id * log(sumi * sumd * _gammas(k, h)) - logfactorial(x_id);

	double density2 = -sumi * sumd * _gammas(k, h) + x_id * log(sumi * sumd * _gammas(k, h)) - logfactorial(x_id);

	result._row = density1;
	result._col = density2;

	return result;
}

void Poisson::imputeMissingData(const mat& V, const mat& W) {


	for (int imiss = 0; imiss < _miss.size(); imiss++) {

		vector<int> coordinates = _miss.at(imiss);
		int miss_row = coordinates.at(0);
		uvec k = arma::find(V.row(miss_row) == 1);
		int miss_col = coordinates.at(1);
		uvec h = arma::find(W.row(miss_col) == 1);

		std::default_random_engine generator;
		std::poisson_distribution<int> distribution(_mus(miss_row)*_nus(miss_col)*_gammas(k(0), h(0)));
		int sample = distribution(generator);
		this->_xsep(miss_row, miss_col) = sample;
	}
}

void Poisson::Mstep(uvec rowind, uvec colind, int k, int h, bool init) {


	//computing gammas

	mat tmp_xk = _xsep.rows(rowind);
	mat tmp_xh = _xsep.cols(colind);
	mat datablock_kh = this->getDatablockkh(rowind, colind);
	_gammas(k, h) = accu(datablock_kh) / (accu(tmp_xk)*accu(tmp_xh));



	return;
}

void Poisson::MstepVW(const mat& V, const mat& W, bool init) {


	//computing gammas
	for (int k = 0; k < _kr; k++)
	{
		uvec rowind = find(V.col(k) == 1);
		mat tmp_xk = _xsep.rows(rowind);

		for (int h = 0; h < _kc; h++) {		

			uvec colind = find(W.col(h) == 1);
			mat tmp_xh = _xsep.cols(colind);

			mat datablock_kh = _xsep.submat(rowind, colind);
			
			_gammas(k, h) = accu(datablock_kh) / (accu(tmp_xk)*accu(tmp_xh));

		}
	}


	return;
}

void Poisson::MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample){
	for (int k = 0; k < _kr; k++)
	{
		uvec rowind = find(Vsample.col(k) == 1);
		mat tmp_xk = xsample.rows(rowind);

		for (int h = 0; h < _kc; h++) {		

			uvec colind = find(Wsample.col(h) == 1);
			mat tmp_xh = xsample.cols(colind);

			mat datablock_kh = xsample.submat(rowind, colind);
			
			_gammas(k, h) = accu(datablock_kh) / (accu(tmp_xk)*accu(tmp_xh));

		}
	}	
	return;
}

void Poisson::fillParameters(int iteration) {
	this->_allgammas.slice(iteration) = this->_gammas;
	this->_allmus.col(iteration) = this->_mus;
	this->_allnus.col(iteration) = this->_nus;
	return;
}

void Poisson::getBurnedParameters(int burn) {
	this->_resmus = conv_to<vec>::from(mean(this->_allmus.cols(burn, _nbSEM - 1), 0));
	this->_resnus = conv_to<vec>::from(mean(this->_allnus.cols(burn, _nbSEM - 1), 0));
	this->_resgammas = mean(this->_allgammas.slices(burn, _nbSEM - 1), 2);

	// not useful for mus and nus
	this->_gammas = this->_resgammas;
	//this->_lambdas = this->_reslambdas;
	return;
}

void Poisson::printResults() {
	this->_gammas.print();

	return;
}

List Poisson::returnResults() {
	List poissonRes = List::create(Rcpp::Named("gamma") = _gammas);
	return(poissonRes);
}

List Poisson::returnParamsChain() {
	List poissonRes = List::create(Rcpp::Named("gamma") = _allgammas);
	return(poissonRes);
}

void Poisson::putParamsToZero() {
	//this->_lambdas = zeros(_N, _J);
}

double Poisson::computeICL(int i, int d, int k, int h) {
	double result = 0;
	if(i==0 && d==0 && k==0 && h==0){
		result = - _kc*_kr/2 * log(_N*_J);
	}
	double density = -_mus(i) * _nus(d) * _gammas(k, h) + _xsep(i, d) * log(_mus(i) * _nus(d) * _gammas(k, h)) - logfactorial(_xsep(i, d));
	result += density;
	return(result);
}


double Poisson::factorial(int n)
{
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

double Poisson::logfactorial(int n)
{
	if (n == 0) {
		return 0;
	}
	return (n == 1) ? 0 : logfactorial(n - 1) + log(n);
}
