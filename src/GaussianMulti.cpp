#include "GaussianMulti.h"


GaussianMulti::GaussianMulti(cube xsepC, int kr, int kc, int nbSEM)
{
	this->_name = "GaussianMulti";

	this->_nbSEM = nbSEM;
	this->_xsepC = xsepC;
	this->_nbDim = _xsepC.n_slices;
	this->_kr = kr;
	this->_kc = kc;
	this->_N = xsepC.n_rows;
	this->_J = xsepC.n_cols;

	this->_mus = zeros(_kr, _kc*_nbDim);
	this->_sigmas = zeros(_kr*_nbDim, _kc*_nbDim);
	this->_resmus = zeros(_kr, _kc*_nbDim);
	this->_ressigmas = zeros(_kr*_nbDim, _kc*_nbDim);
	this->_allmus = zeros(_kr, _kc*_nbDim, _nbSEM);
	this->_allsigmas = zeros(_kr*_nbDim, _kc*_nbDim, _nbSEM);

}
GaussianMulti::GaussianMulti()
{
}


GaussianMulti::~GaussianMulti()
{
}

void GaussianMulti::missingValuesInit() {
	return;
}


TabProbsResults GaussianMulti::SEstep(const mat& V, const mat& W) {
	TabProbsResults result = TabProbsResults(_N, _kr, _J, _kc);		for (int i = 0; i < _N; i++)
	{

		for (int k = 0; k < _kr; k++)
		{

			for (int d = 0; d < _J; d++)
			{

				for (int h = 0; h < _kc; h++)
				{
					vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
					mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
					// consider using armadillo function normpdf(X,M,S)
					vec xcomp = _xsepC.tube(i, d);
					double density = this->densityMulti(xcomp, covMat_kh, mu_kh);

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

mat GaussianMulti::SEstepRow(const mat& W) {
	mat result(_N, _kr);
	result.zeros();
	for (int i = 0; i < _N; i++)
	{

		for (int k = 0; k < _kr; k++)
		{

			for (int d = 0; d < _J; d++)
			{

				for (int h = 0; h < _kc; h++)
				{
					vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
					mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
					// consider using armadillo function normpdf(X,M,S)
					vec xcomp = _xsepC.tube(i, d);
					double density = this->densityMulti(xcomp, covMat_kh, mu_kh);

					result(i, k) += W(d, h) * density;

				}
			}
		}
	}
	return(result);
}

mat GaussianMulti::SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample){
	mat result(_N, _kr);
	result.zeros();
	
	for (int d = 0; d < _J; d++)
	{

		// we can't build an xsampleC because subcube does not exist, so we just test if 
		// the column is in the sample
		if(any(colSample == d)){
			for (int h = 0; h < _kc; h++)
			{

				if(Wsample(d,h)==1){
					for (int i = 0; i < _N; i++)
					{

						for (int k = 0; k < _kr; k++)
						{
							vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
							mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
							// consider using armadillo function normpdf(X,M,S)
							vec xcomp = _xsepC.tube(i, d);
							double density = this->densityMulti(xcomp, covMat_kh, mu_kh);

							result(i, k) +=  density;

						}
					}
				}
				
			}
		}
		
	}

	return result;
}

mat GaussianMulti::SEstepCol(const mat& V) {
	mat result(_J, _kc);
	result.zeros();
	for (int i = 0; i < _N; i++)
	{

		for (int k = 0; k < _kr; k++)
		{

			for (int d = 0; d < _J; d++)
			{

				for (int h = 0; h < _kc; h++)
				{
					vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
					mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
					// consider using armadillo function normpdf(X,M,S)
					vec xcomp = _xsepC.tube(i, d);
					double density = this->densityMulti(xcomp, covMat_kh, mu_kh);

					result(d, h) += V(i, k) * density;
				}
			}
		}
	}
	return(result);
}

LogProbs GaussianMulti::SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec) {
	LogProbs result(0, 0);

	vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
	mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
	double density = this->densityMulti(x_id_vec, covMat_kh, mu_kh);
	result._row = density;
	result._col = density;

	return(result);
}

void GaussianMulti::MstepVW(const mat& V, const mat& W, bool init) {

	for (int k = 0; k < _kr; k++)
	{
		for (int h = 0; h < _kc; h++) {
			uvec rowind = find(V.col(k) == 1);
			uvec colind = find(W.col(h) == 1);


			/* not possible to find the subcube, so building a matrix
			that have the x_ij in lines
			*/
			mat datablock_kh = zeros(rowind.n_elem*colind.n_elem, _nbDim);
			int nbi = 0;
			for (int rowindi = 0; rowindi < rowind.n_elem; rowindi++) {
				for (int colindi = 0; colindi < colind.n_elem; colindi++) {
					vec tmp_kh = _xsepC.tube(rowind(rowindi), colind(colindi));
					datablock_kh.row(nbi) = tmp_kh.t();
					nbi++;
				}
			}
			
			this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)) = mean(datablock_kh,0);

			this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1)) = 
				cov(datablock_kh);

		}
	}
	return;
}

void GaussianMulti::MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample){
	// TODO: not possible for now (think about it)
	return;
}

void GaussianMulti::imputeMissingData(const mat& V, const mat& W) {
	return;
}

void GaussianMulti::Mstep(uvec rowind, uvec colind, int k, int h, bool init) {


	/* not possible to find the subcube, so building a matrix
	that have the x_ij in lines
	*/
	mat datablock_kh = zeros(rowind.n_elem*colind.n_elem, _nbDim);
	int nbi = 0;
	for (int rowindi = 0; rowindi < rowind.n_elem; rowindi++) {
		for (int colindi = 0; colindi < colind.n_elem; colindi++) {
			vec tmp_kh = _xsepC.tube(rowind(rowindi), colind(colindi));
			datablock_kh.row(nbi) = tmp_kh.t();
			nbi++;
		}
	}

	this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)) = mean(datablock_kh, 0);

	this->_sigmas.submat(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1)) =
		cov(datablock_kh);


	return;
}

void GaussianMulti::fillParameters(int iteration) {
	this->_allmus.slice(iteration) = this->_mus;
	this->_allsigmas.slice(iteration) = this->_sigmas;
	return;
}

void GaussianMulti::getBurnedParameters(int burn) {
	this->_resmus = mean(this->_allmus.slices(burn, _nbSEM - 1), 2);
	this->_ressigmas = mean(this->_allsigmas.slices(burn, _nbSEM - 1), 2);
	this->_mus = this->_resmus;
	this->_sigmas = this->_ressigmas;
	return;
}


void GaussianMulti::printResults() {
	_mus.print();
	_sigmas.print();
	return;
}

List GaussianMulti::returnResults() {
	List gaussianRes = List::create(Rcpp::Named("sigmas") = this->_sigmas,
									Rcpp::Named("mus") = this->_mus);
	return(gaussianRes);
}

List GaussianMulti::returnParamsChain() {
	List gaussianRes = List::create(Rcpp::Named("sigmas") = this->_allsigmas,
									Rcpp::Named("mus") = this->_allmus);
	return(gaussianRes);
}

void GaussianMulti::putParamsToZero() {
	return;
}

double GaussianMulti::computeICL(int i, int d, int k, int h) {
	double result = 0;
	if(i==0 && d==0 && k==0 && h==0){
		// did not divided by two because there are two parameter mu and sigma
		result = - _kc * _kr * (_nbDim + _nbDim * (_nbDim-1)/2 ) / 2 * log(_N*_J); 
	}
	vec mu_kh = conv_to<vec>::from(this->_mus.submat(span(k, k), span(h*_nbDim, h*_nbDim + _nbDim - 1)));
	mat covMat_kh = this->_sigmas(span(k*_nbDim, k*_nbDim + _nbDim - 1), span(h*_nbDim, h*_nbDim + _nbDim - 1));
	// consider using armadillo function normpdf(X,M,S)
	vec xcomp = _xsepC.tube(i, d);
	double density = this->densityMulti(xcomp, covMat_kh, mu_kh);
	result += log(density);
	return(result);
}

mat GaussianMulti::colkmeans() {
	mat result;
	return(result);
}

// see http://gallery.rcpp.org/articles/dmvnorm_arma/ for implementation
double GaussianMulti::densityMulti(vec x, mat covMat, vec meanVec, bool logd) {
	double result = 0;
	const double logSqrt2Pi = 0.5*std::log(2 * M_PI);

	mat rooti = trans(inv(trimatu(chol(covMat))));
	double rootisum = sum(log(rooti.diag()));
	vec tmp = rooti * (x - meanVec);
	result = logSqrt2Pi - 0.5 * sum(tmp%tmp) + rootisum;
	if (logd == false) {
		result = exp(result);
	}

	return(result);
}
