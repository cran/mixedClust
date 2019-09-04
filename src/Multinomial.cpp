#include "Multinomial.h"

Multinomial::Multinomial(mat& xsep, int kr, int kc, int m, int nbSEM)
	:Distribution(xsep, kr, kc, nbSEM)
{
	this->_name = "Multinomial";
	this->_m = m;
	this->_alphas = zeros(_kr, _kc, _m);

	// the stocked parameters for each iteration
	for (size_t sem = 0; sem < _nbSEM; sem++)
	{
		cube alpha_tmp = zeros(_kr, _kc, _m);
		this->_allalphas.push_back(alpha_tmp);
	}

	// the resulting parameters
	this->_resalphas = zeros(_kr, _kc, _m);


}

Multinomial::Multinomial()
{

}


Multinomial::~Multinomial()
{
}

void Multinomial::missingValuesInit() {
	for (int imiss = 0; imiss < _miss.size(); imiss++) {
		mt19937 gen(_rd());
		double eqprob = (double)1 / _m;
		vec vecprob(_m, fill::ones);
		vecprob = vecprob*eqprob;
		discrete_distribution<> d(vecprob.begin(), vecprob.end()); // maybe a problem?
		int sample = d(gen);
		_xsep(_miss.at(imiss)[0], _miss.at(imiss)[1]) = sample + 1;
	}
	return;
}


TabProbsResults Multinomial::SEstep(const mat& V, const mat& W)
{
	TabProbsResults result = TabProbsResults(_N, _kr, _J, _kc);
	for (size_t i = 0; i < _N; i++)
	{
		for (size_t k = 0; k < _kr; k++)
		{
			for (size_t d = 0; d < _J; d++)
			{
				for (size_t h = 0; h < _kc; h++)
				{
					for (size_t cat = 1; cat <= this->_m; cat++)
					{
						if (_xsep(i, d)==cat) {
							double local_alpha = _alphas(k, h, (cat-1)); 
							double log_local_alpha = 0;
							if (local_alpha != 0) {
								log_local_alpha = log(_alphas(k, h, (cat-1)));
							}
							else {
								log_local_alpha = -30; // TODO : not very precise?
							}
							result._tabprobaV(i, k) = result._tabprobaV(i, k) +
								W(d, h) * log_local_alpha;
							result._tabprobaW(d, h) = result._tabprobaW(d, h) +
								V(i, k) * log_local_alpha;
							
						}
					}
				}
			}
		}
	}
	return result;
}


mat Multinomial::SEstepRow(const mat& W)
{
	mat result(_N, _kr);
	result.zeros();

	for (size_t d = 0; d < _J; d++)
	{
		for (size_t h = 0; h < _kc; h++)
		{
			if(W(d, h) == 1){
				for (size_t i = 0; i < _N; i++)
				{
					for (size_t k = 0; k < _kr; k++)
					{
						
						int cat = _xsep(i,d);
						double local_alpha = _alphas(k, h, (cat-1)); 
						double log_local_alpha = 0;
						if (local_alpha != 0) {
							log_local_alpha = log(_alphas(k, h, (cat-1)));
						}
						else {
							log_local_alpha = log(0.01); // TODO : not very precise?
						}
						result(i, k) += log_local_alpha;
					}
				}
			}
		}
	}
	return result;
}

mat Multinomial::SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample){
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
						int cat = xsample(i,d);
						double local_alpha = _alphas(k, h, (cat-1)); 
						double log_local_alpha = 0;
						if (local_alpha != 0) {
							log_local_alpha = log(_alphas(k, h, (cat-1)));
						}
						else {
							log_local_alpha = log(0.01); // TODO : not very precise?
						}
						result(i, k) += log_local_alpha;

					}
				}
			}
			
		}
	}

	return result;
}

mat Multinomial::SEstepCol(const mat& V)
{
	mat result(_J, _kc);
	result.zeros();
	for (size_t i = 0; i < _N; i++)
	{
		for (size_t k = 0; k < _kr; k++)
		{

			if(V(i, k) == 1){
				for (size_t d = 0; d < _J; d++)
				{
					for (size_t h = 0; h < _kc; h++)
					{
						
						int cat = _xsep(i,d);
						double local_alpha = _alphas(k, h, (cat-1)); 
						double log_local_alpha = 0;
						if (local_alpha != 0) {
							log_local_alpha = log(_alphas(k, h, (cat-1)));
						}
						else {
							log_local_alpha = log(0.01); // TODO : not very precise?
						}
						result(d, h) += log_local_alpha;
						
					}
				}
			}
			
		}
	}
	return result;
}

LogProbs Multinomial::SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec)
{
	LogProbs result(0, 0);
	for (size_t cat = 1; cat <= this->_m; cat++)
	{
		if (x_id == cat) {
			double local_alpha = this->_alphas.tube(k, h)(cat - 1);
			double log_local_alpha = 0;
			if (local_alpha != 0) {
				log_local_alpha = log(this->_alphas.tube(k, h)(cat - 1));
			}
			else {

				log_local_alpha = -100; // TODO : not very precise?
			}
			result._row = log_local_alpha;
			result._col=  log_local_alpha;

		}
	}

	return result;
}

void Multinomial::imputeMissingData(const mat& V, const mat& W) {
	for (int i = 0; i < _miss.size(); i++)
	{
		int row = (_miss[i])[0];
		int col = (_miss[i])[1];

		// This part could be done wit find()
		rowvec vrow = V.row(row);
		int rowcluster=-1;
		for (int ivrow = 0; ivrow < vrow.size(); ivrow++)
		{
			if (vrow(ivrow) == 1) {
				rowcluster = ivrow;
				break;
			}
		}
		rowvec wrow = W.row(col);
		int colcluster=-1;
		for (int iwrow = 0; iwrow < wrow.size(); iwrow++)
		{
			if (wrow(iwrow) == 1) {
				colcluster = iwrow;
				break;
			}
		}
		// end of "This part could be done with find()"
		vec alpha_tmp = _alphas.tube(rowcluster, colcluster);
		mt19937 gen(_rd());
		discrete_distribution<> d(alpha_tmp.begin(), alpha_tmp.end()); // maybe a problem?
		int sample = d(gen);
		_xsep(row, col) = sample + 1;
	}
}


void Multinomial::Mstep(uvec rowind, uvec colind, int k, int h, bool init)
{

	rowvec datablock_kh = this->getDatablockkh(rowind, colind);
	this->_alphas.tube(k, h) = this->getAlpha(datablock_kh);
	return;
}

void Multinomial::MstepVW(const mat& V, const mat& W, bool init)
{
	for (int k = 0; k < _kr; k++)
	{
		for (int h = 0; h < _kc; h++) {
			mat datablock_kh_mat;
			uvec rowind = find(V.col(k) == 1);
			uvec colind = find(W.col(h) == 1);
			datablock_kh_mat = _xsep.submat(rowind, colind);
			rowvec datablock_kh = conv_to<rowvec>::from(vectorise(datablock_kh_mat));
			this->_alphas.tube(k,h) = this->getAlpha(datablock_kh);
		}
	}	
	return;
}

void Multinomial::MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample){
	
	for (int k = 0; k < _kr; k++)
	{

		for (int h = 0; h < _kc; h++) {

			uvec rowind = find(Vsample.col(k) == 1);
			uvec colind = find(Wsample.col(h) == 1);
			mat  datablock_kh_mat = xsample.submat(rowind, colind);
			rowvec datablock_kh = conv_to<rowvec>::from(vectorise(datablock_kh_mat));
			this->_alphas.tube(k,h) = this->getAlpha(datablock_kh);
		}
	}	
	return;
}

void Multinomial::fillParameters(int iteration) {
	this->_allalphas.at(iteration) = this->_alphas;
	return;
}

void Multinomial::getBurnedParameters(int burn) {
	cube result = zeros(_kr, _kc, _m);
	for (int i = burn; i < _nbSEM; i++) {
		for (int k = 0; k < _kr; k++) {
			for (int h = 0; h < _kc; h++) {
				for (int mu = 0; mu < _m; mu++) {
					result(k, h, mu) += this->_allalphas.at(i)(k, h, mu);
				}
			}
		}
	}
	this->_resalphas = result / (_nbSEM - burn);
	this->_alphas = this->_resalphas;
	return;
}

void Multinomial::printResults() {
	this->_alphas.print();
}

List Multinomial::returnResults() {
	List multiRes = List::create(Rcpp::Named("betas") = _alphas);
	return(multiRes);
}

List Multinomial::returnParamsChain() {
	List multiRes = List::create(Rcpp::Named("betas") = _allalphas);
	return(multiRes);
}

void Multinomial::putParamsToZero() {
	return;
}

double Multinomial::computeICL(int i, int d, int k, int h) {
	double result = 0;
	if(i==0 && d==0 && k==0 && h==0){
		// did not divided by two because there are two parameter mu and sigma
		result = - _kc * _kr * ( _m - 1 ) / 2  * log(_N*_J); 
	}
	for (size_t cat = 1; cat <= this->_m; cat++){
		if(_xsep(i,d) == cat){
			double local_alpha = _alphas(k, h, (cat-1));
			double log_local_alpha = 0;
			if (local_alpha != 0) {
				log_local_alpha = log(_alphas(k, h, (cat-1)));
			}
			else {
				log_local_alpha = -100; // TODO : not very precise?
			}
			result += log_local_alpha;
		}
	}

	/*
*/
	return(result);
}

vec Multinomial::getAlpha(rowvec block_kh) {
	int tot = block_kh.n_elem;
	vec result(_m);
	for (int i = 0; i < _m; i++)
	{
		uvec values = find(block_kh == (i + 1));
		result(i) = (double)values.n_elem / tot;
	}

	return result;
}


void Multinomial::printAlphas() {
	this->_alphas.print();
}




