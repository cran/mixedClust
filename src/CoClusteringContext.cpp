#include "CoClusteringContext.h"

CoClusteringContext::CoClusteringContext(arma::mat& x, std::vector< arma::urowvec > dlist,
	std::vector< std::string > distrib_names, int kr, std::vector< int > kc,
	std::string init, int nbSEM, int nbSEMburn, int nbindmini, std::vector< int > m,
	arma::cube functionalData, vector<int> zrinit, std::vector<int> zcinit,
	vector<double> percentRandomB, vector<double> percentRandomP) {

	// attributes that are directly instanciated
	this->_x = x;
	this->_Nr = _x.n_rows;
	this->_dlist = dlist;
	this->_distrib_names = distrib_names;
	this->_kr = kr;
	this->_kc = kc;
	this->_m = m;
	this->_init = init;
	this->_nbSEM = nbSEM;
	this->_nbSEMburn = nbSEMburn;
	this->_nbindmini = nbindmini;

	// percents for complex initialization
	this->_percentRandomB = percentRandomB; 
	this->_percentRandomP = percentRandomP; 



	//attributes to construct
	this->_number_distrib = _distrib_names.size();


	vector<vector<int>> tmp_zcvec(_number_distrib);
	vector<rowvec> tmp_rhovec(_number_distrib);
	vector<rowvec> tmp_resrhovec(_number_distrib);


	

	this->_zrinit = zrinit;
	this->_zcinit = zcinit;

	int im = 0;


	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		// instanciate Distribution :
		string distrib_name = _distrib_names.at(idistrib);
		// define xsep
		mat xsep; 
		if (!(distrib_name == "Functional")) {
			xsep = _x.cols(_dlist.at(idistrib));
		}
		// here, should be done with a map, but I can't for now
		
		if (distrib_name == "Bos") {
			unsigned int iterordiEM = 50;
			this->_distrib_objects.push_back(new Bos(xsep, _kr, _kc.at(idistrib), _m[im], this->_nbSEM, iterordiEM));
			im++;
		}
		if (distrib_name == "Poisson") {
			this->_distrib_objects.push_back(new Poisson(xsep, _kr, _kc.at(idistrib), this->_nbSEM));
		}
		if (distrib_name == "Functional") {
			this->_distrib_objects.push_back(new GaussianMulti(functionalData, _kr, _kc.at(idistrib), _nbSEM));
		}
		if (distrib_name == "Gaussian") {
			this->_distrib_objects.push_back(new Gaussian(xsep, _kr, _kc.at(idistrib), this->_nbSEM));
		}
		if (distrib_name == "Multinomial") {
			this->_distrib_objects.push_back(new Multinomial(xsep, _kr, _kc.at(idistrib), _m[im], this->_nbSEM));
			im++;
		}



		if (!(distrib_name == "Functional")) {
			this->_Jc.push_back(_dlist.at(idistrib).size());
		}
		else if (!(functionalData.n_slices == 1 && functionalData.n_rows == 1 && functionalData.n_cols == 1)) {
			this->_Jc.push_back(functionalData.n_cols);
		}

		vector<int> tmp_zc(_Jc.at(idistrib));
		std::fill(tmp_zc.begin(), tmp_zc.end(), 0);
		tmp_zcvec[idistrib] = tmp_zc;
		rowvec tmp_rho(_kc.at(idistrib), fill::zeros);
		tmp_rhovec.push_back(tmp_rho);
		rowvec tmp_resrho(_kc.at(idistrib), fill::zeros);
		tmp_resrhovec.push_back(tmp_rho);


		// probas and logprobas tables
		mat tmp_probaW(_Jc[idistrib], _kc.at(idistrib), fill::zeros);
		this->_probaW.push_back(tmp_probaW);
		mat tmp_logprobaW(_Jc[idistrib], _kc.at(idistrib), fill::zeros);
		this->_logprobaW.push_back(tmp_probaW);
		mat tmp_W(_Jc[idistrib], _kc.at(idistrib), fill::zeros);
		this->_W.push_back(tmp_W);

		mat tmp_zcchain(_nbSEM, _Jc[idistrib], fill::zeros);
		this->_zcchain.push_back(tmp_zcchain);

	}

	//this->_Jc = tmp_Jc;
	//this->_probaW = tmp_probaWvec;
	//this->_logprobaW = tmp_logprobaWvec;
	this->_zc = tmp_zcvec;
	this->_rho = tmp_rhovec;
	this->_resrho = tmp_resrhovec;
	//this->_W = tmp_Wvec;

	//this->_zcchain = tmp_zcchainvec;

	// attributes regarding lines
	
	if (!(functionalData.n_slices == 1 && functionalData.n_rows == 1 && functionalData.n_cols == 1) && _number_distrib == 1) {
		this->_Nr = functionalData.n_rows;
	}

	this->_zrchain = zeros(_nbSEM,_Nr);

	mat tmp_probaV(_Nr, _kr, fill::zeros);
	this->_probaV = tmp_probaV;
	mat tmp_logprobaV(_Nr, _kr, fill::zeros);
	this->_logprobaV = tmp_probaV;
	vector<int> tmp_zr(this->_Nr);
	std::fill(tmp_zr.begin(), tmp_zr.end(), 0);
	this->_zr = tmp_zr;


	mat tmp_V(_Nr, _kr, fill::zeros);
	this->_V = tmp_V;
	rowvec tmp_gamma(_kr);
	std::fill(tmp_gamma.begin(), tmp_gamma.end(), 0);
	this->_gamma = tmp_gamma;
	rowvec tmp_resgamma(_kr);
	std::fill(tmp_resgamma.begin(), tmp_resgamma.end(), 0);
	this->_resgamma = tmp_resgamma;

	// filling the parameters on SEM iterations
	vector<vector<rowvec>> tmp_allrho;
	vector<rowvec> tmp_allgamma(_nbSEM);
	for (int isem = 0; isem < _nbSEM; isem++) {
		rowvec tmp_gam(_kr);
		std::fill(tmp_gam.begin(), tmp_gam.end(), 0);
		tmp_allgamma.push_back(tmp_gam);

		vector<rowvec> tmp_rho2(_number_distrib);
		for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
			rowvec tmp_rh(_kc[idistrib]);
			std::fill(tmp_rh.begin(), tmp_rh.end(), 0);
			tmp_rho2.push_back(tmp_rh);
		}
		tmp_allrho.push_back(tmp_rho2);
	}
	this->_allrho = tmp_allrho;
	this->_allgamma = tmp_allgamma;
}

CoClusteringContext::CoClusteringContext()
{
}


CoClusteringContext::~CoClusteringContext()
{
	for (int i = 0; i < _distrib_objects.size(); i++)
    {
        delete _distrib_objects[i]; // this is needed to free the memory
    }
    _distrib_objects.clear();

}

void CoClusteringContext::disposeObject() { delete this; }

bool CoClusteringContext::initialization() {
	//arma_rng::set_seed_random();
	// TODO : enlever discrete_distribution<> d(vec.begin(), vec.end()); ??
	if (_init == "random" || _init=="randomBurnin") {
		// partitions V
		vector<double> vec(_kr);
		double prob = (double)1 / _kr;
		std::fill(vec.begin(), vec.end(), prob);
		discrete_distribution<> d(vec.begin(), vec.end());
		this->_V.zeros();
		for (int i = 0; i<_Nr; ++i) {
			// random!
			mt19937 gen(_rd());
			int sample = d(gen);
			//int sample = randi(1, distr_param(0, (_kr - 1)))(0);
			this->_V(i, sample) = 1;
		}
		// updating gamma
		this->_gamma = this->getMeans(this->_V);
		for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
		{
			this->_W.at(idistrib).zeros();
			/*vector<double> vec(_kc.at(idistrib));
			double prob = (double)1 / _kc.at(idistrib);
			std::fill(vec.begin(), vec.end(), prob);
			discrete_distribution<> d(vec.begin(), vec.end());*/
			arma::vec probas(_kc[idistrib]);
			probas.ones();
			probas = probas/_kc[idistrib];
			discrete_distribution<> d(probas.begin(), probas.end());
			for (int i = 0; i<_Jc.at(idistrib); ++i) {
				// random!
				mt19937 gen(_rd());
				int sample = d(gen);
				//int sample = randi(1, distr_param(0, (_kc[idistrib] - 1)))(0);
				this->_W.at(idistrib)(i, sample) = 1;
			}

			this->_distrib_objects[idistrib]->MstepVW(this->_V, this->_W.at(idistrib), true);

			// updating rho 
			this->_rho.at(idistrib) = this->getMeans(this->_W.at(idistrib));
			//_rho.at(idistrib).print();
		}

		return true;

	}
	if (_init == "randomParams") {

		double percentR = _percentRandomP[0];
		double percentC = _percentRandomP[1];



		int Nsample = ceil(percentR * _Nr);
		vector<int> Jsample(_number_distrib);


		// define Vsample and the selected row indexes
		vector<double> vec(_kr);
		double prob = (double)1 / _kr;
		std::fill(vec.begin(), vec.end(), prob);
		discrete_distribution<> d(vec.begin(), vec.end());
		mat Vsample = zeros(Nsample, _kr);
		uvec rowSample(Nsample);
		for (int i = 0; i<Nsample; ++i) {
			int sample = randi(1, distr_param(0, (_kr - 1)))(0);
			Vsample(i, sample) = 1;
			int row = randi(1, distr_param(0, (_Nr - 1)))(0);
			rowSample(i) = row;
		}
		// updating gamma
		this->_gamma = this->getMeans(Vsample);
		vector<mat> Wsamples(_number_distrib);
		vector<uvec> colSamples(_number_distrib);

		for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
		{

			Jsample[idistrib] = ceil(percentC * _Jc[idistrib]);
			mat Wsample = zeros(Jsample[idistrib],_kc[idistrib]);

			arma::vec probas(_kc[idistrib]);
			probas.ones();
			probas = probas/_kc[idistrib];
			discrete_distribution<> d(probas.begin(), probas.end());
			uvec colSample(Jsample[idistrib]);

			for (int i = 0; i<Jsample[idistrib]; ++i) {

				int sample = randi(1, distr_param(0, (_kc[idistrib] - 1)))(0);
				Wsample(i, sample) = 1;
				int col = randi(1, distr_param(0, (_Jc[idistrib] - 1)))(0);
				colSample(i) = col;
			}

			Wsamples[idistrib] = Wsample;
			colSamples[idistrib] = colSample;

			this->_distrib_objects[idistrib]->initParams(rowSample, colSample, Vsample, Wsample);
			// updating rho 
			this->_rho.at(idistrib) = this->getMeans(Wsample);


		}

		this->printResults();


		this->SEstepRowRandomParamsInit(Wsamples, colSamples);
		this->sampleV();

		this->SEstepCol();
		this->sampleW();


		for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
				this->_distrib_objects[idistrib]->MstepVW(this->_V, this->_W.at(idistrib), true);
		}

		return true;
	}
	if (_init == "kmeans") {

		bool verif = false;
		int restart_init = 0;
		while (verif == false && restart_init < 15) {

			this->_V = this->kmeansi();
			//updating gamma
			this->_gamma = this->getMeans(this->_V);
			for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
				this->_W.at(idistrib) = this->_distrib_objects[idistrib]->colkmeans();
				//updating distribution parameters;
				// updating rho 
				this->_rho.at(idistrib) = this->getMeans(this->_W.at(idistrib));
			}
			verif = this->verif();
			if (!verif) {
				restart_init++;
			}
		}

		if(verif){
			for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
				this->_distrib_objects[idistrib]->MstepVW(this->_V, this->_W.at(idistrib), true);
			}
			return true;
		}
		else{
			return false;
		}

	}
	if(_init == "provided"){
		this->_V.zeros();
		for(int i = 0; i<_Nr; i++){
			this->_V(i,(_zrinit[i]-1)) = 1;
		}
		this->_gamma = this->getMeans(this->_V);

		int iteration = 0;
		for(int idistrib = 0; idistrib < _number_distrib ; idistrib++){
			this->_W.at(idistrib).zeros();
			for(int d = 0; d < _Jc.at(idistrib); d++){
				this->_W.at(idistrib)(d,(_zcinit[iteration]-1)) = 1;
				iteration++;
			}
			this->_distrib_objects[idistrib]->MstepVW(this->_V, this->_W.at(idistrib), true);
			this->_rho.at(idistrib) = this->getMeans(this->_W.at(idistrib));
		}
		return true;
	}
	return false;
}

string CoClusteringContext::getInitialization(){
	return this->_init;
}

void CoClusteringContext::missingValuesInit() {
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->missingValuesInit();
	}
	return;
}

void CoClusteringContext::Mstep() {
	this->_gamma = this->getMeans(this->_V);
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int k = 0; k < _kr; k++) {
			for (int h = 0; h < _kc[idistrib]; h++) {
				uvec rowind = find(this->_V.col(k) == 1);
				uvec colind = find(this->_W[idistrib].col(h) == 1);
				this->_distrib_objects[idistrib]->Mstep(rowind, colind, k, h, false);
			}
		}
		this->_rho.at(idistrib) = this->getMeans(this->_W.at(idistrib));
	}
}

void CoClusteringContext::MstepVW() {	
	this->_gamma = this->getMeans(this->_V);

	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->MstepVW(_V, _W.at(idistrib), false);
		this->_rho.at(idistrib) = this->getMeans(this->_W.at(idistrib));
		
	}
}



void CoClusteringContext::SEstep()
{
	// Computing the log-probabilites
	this->_logprobaV.zeros();
	this->_logprobaV.each_row() += log(this->_gamma);  

	for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
	{
		this->_logprobaW.at(idistrib).zeros();
		this->_logprobaW.at(idistrib).each_row() += log(this->_rho.at(idistrib));
		
		TabProbsResults result(_Nr, _kr, _Jc.at(idistrib), _kc.at(idistrib));
		result = _distrib_objects[idistrib]->SEstep(_V, _W.at(idistrib));
		this->_logprobaV += result._tabprobaV;
		this->_logprobaW.at(idistrib) += result._tabprobaW;
	}


	// Computing the probabilites
	for (int i = 0; i < _Nr; i++) {
		for (int k = 0; k < _kr; k++) {
			this->_probaV(i, k) = exp(this->_logprobaV(i, k) - logsum(_logprobaV.row(i)));
		}
	}
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			for (int h = 0; h < _kc.at(idistrib); h++) {
				this->_probaW.at(idistrib)(d, h) = exp(this->_logprobaW.at(idistrib)(d, h) - logsum(this->_logprobaW.at(idistrib).row(d)));
			}
		}
	}

}

void CoClusteringContext::SEstepRow()
{
	// Computing the log-probabilites
	this->_logprobaV.zeros();
	this->_logprobaV.each_row() += log(this->_gamma);  


	for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
	{
		mat result(_Nr, _kr);
		result.zeros();
		result = _distrib_objects[idistrib]->SEstepRow(_W.at(idistrib));
		this->_logprobaV += result;
	}

	// Computing the probabilites
	for (int i = 0; i < _Nr; i++) {
		for (int k = 0; k < _kr; k++) {
			this->_probaV(i, k) = exp(this->_logprobaV(i, k) - logsum(_logprobaV.row(i)));
		}
	}

}

void CoClusteringContext::SEstepRowRandomParamsInit(vector<mat>& Wsamples, vector<uvec>& colSamples){
	this->_logprobaV.zeros();
	this->_logprobaV.each_row() += log(this->_gamma);  

	for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
	{
		mat result(_Nr, _kr);
		result.zeros();
		result = _distrib_objects[idistrib]->SEstepRowRandomParamsInit(Wsamples[idistrib], colSamples[idistrib]);
		this->_logprobaV += result;

	}
	// Computing the probabilites
	for (int i = 0; i < _Nr; i++) {
		for (int k = 0; k < _kr; k++) {
			this->_probaV(i, k) = exp(this->_logprobaV(i, k) - logsum(_logprobaV.row(i)));
		}
	}
}

void CoClusteringContext::SEstepCol()
{
	// Computing the log-probabilites

	for (int idistrib = 0; idistrib < _number_distrib; idistrib++)
	{
		this->_logprobaW.at(idistrib).zeros();
		this->_logprobaW.at(idistrib).each_row() += log(this->_rho.at(idistrib));

		mat result(_Jc.at(idistrib), _kc.at(idistrib));
		result = _distrib_objects[idistrib]->SEstepCol(_V);
		this->_logprobaW.at(idistrib) += result;

	}


	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			for (int h = 0; h < _kc.at(idistrib); h++) {
				this->_probaW.at(idistrib)(d, h) = exp(this->_logprobaW.at(idistrib)(d, h) - logsum(this->_logprobaW.at(idistrib).row(d)));
			}
		}
	}

}


void CoClusteringContext::sampleVW() {
	// Sampling V and W

	this->_V.zeros();
	for (int i = 0; i < _Nr; i++) {
		// random!
		rowvec vec = _probaV.row(i);
		discrete_distribution<> dis(vec.begin(), vec.end());
		//int sample = dis(gen);
		mt19937 gen(_rd());
		int sample = dis(gen);
		this->_V(i, sample) = 1;
	}
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_W.at(idistrib).zeros();
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			//random!
			rowvec vec = _probaW.at(idistrib).row(d);
			discrete_distribution<> dis(vec.begin(), vec.end());
			mt19937 gen(_rd());
			int sample = dis(gen);
			this->_W.at(idistrib)(d, sample) = 1;
		}

	}
	return;
}
void CoClusteringContext::sampleV() {
	// Sampling V and W

	this->_V.zeros();
    for (int i = 0; i < _Nr; i++) {
		// random!
		rowvec vec = _probaV.row(i);
		discrete_distribution<> dis(vec.begin(), vec.end());
		mt19937 gen(_rd());
		int sample = dis(gen);
		this->_V(i, sample) = 1;
	}
	return;
}

void CoClusteringContext::sampleW() {
	// Sampling W
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_W.at(idistrib).zeros();
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			rowvec vec = _probaW.at(idistrib).row(d);
			discrete_distribution<> dis(vec.begin(), vec.end());
			mt19937 gen(_rd());
			int sample = dis(gen);
			this->_W.at(idistrib)(d, sample) = 1;
		}
	}
	
	return;
}

void CoClusteringContext::sampleVWStock() {
	// Sampling V and W

	mat countV = zeros(_Nr, _kr);
	vector<mat> countW(_number_distrib);
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			mat countWid(_Jc.at(idistrib), _kc.at(idistrib), fill::zeros);
			countW[idistrib] = countWid;
		}
	}
	for (int iter = 0; iter < _nbSEM; iter++) {
		this->_V.zeros();
		for (int i = 0; i < _Nr; i++) {
			// random!
			rowvec vec = _probaV.row(i);
			discrete_distribution<> dis(vec.begin(), vec.end());
			mt19937 gen(_rd());
			int sample = dis(gen);
			this->_V(i, sample) = 1;
			countV(i, sample) += 1;
		}
		for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
			this->_W.at(idistrib).zeros();
			for (int d = 0; d < _Jc.at(idistrib); d++) {
				rowvec vec = _probaW.at(idistrib).row(d);
				discrete_distribution<> dis(vec.begin(), vec.end());
				mt19937 gen(_rd());
				int sample = dis(gen);
				this->_W.at(idistrib)(d, sample) = 1;
				countW.at(idistrib)(d, sample) += 1;

			}
		}
	}

	//determinging final partitions
	this->_V.zeros();
	for (int i = 0; i < _Nr; i++) {
		int maxind = countV.row(i).index_max();
		this->_V(i, maxind) = 1;
	}
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_W.at(idistrib).zeros();
		for (int d = 0; d < _Jc.at(idistrib); d++) {
			int maxind = countW.at(idistrib).row(d).index_max();
			this->_W.at(idistrib)(d, maxind) = 1;
		}
	}

	return;
}

void CoClusteringContext::imputeMissingData() {
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->imputeMissingData(this->_V, this->_W.at(idistrib));
	}
}

bool CoClusteringContext::verif() {
	bool result = true;
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		result = this->_distrib_objects[idistrib]->verif(_V, _W.at(idistrib), _nbindmini);

		if (result == false) {			
			return result;
		}
	}
	return result;
}

vector<vector<int>> CoClusteringContext::verification() {
	vector<vector<int>> result;
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		int verifD = this->_distrib_objects[idistrib]->verification(_V, _W.at(idistrib), _nbindmini);

		if (!(verifD==-1)) {			
			vector<int> newline(2);
			newline.at(0) = idistrib;
			newline.at(1) = verifD; 
			result.push_back(newline);
		}
		
	}
	return result;
}


bool CoClusteringContext::verificationRows() {
	vector<vector<int>> result;
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		int verifD = this->_distrib_objects[idistrib]->verification(_V, _W.at(idistrib), _nbindmini);

		if (!(verifD==-2)) {			
			vector<int> newline(2);
			newline.at(0) = idistrib;
			newline.at(1) = verifD; 
			result.push_back(newline);
		}
		
	}

	return (result.size()>0) ? true : false;
}

bool CoClusteringContext::verificationCols() {
	vector<vector<int>> result;
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		int verifD = this->_distrib_objects[idistrib]->verification(_V, _W.at(idistrib), _nbindmini);

		//verifD=-1 means it's all right, verifID=-2 means there is degenrancy in lines
		if (!(verifD==-1) && !(verifD==-2)) {			
			vector<int> newline(2);
			newline.at(0) = idistrib;
			newline.at(1) = verifD; 
			result.push_back(newline);
		}
		
	}
	return (result.size()>0) ? true : false;
}

void CoClusteringContext::noColDegenerancy(vector<vector<int>> distrib_col){
	double percent = _percentRandomB[1]/100;

	for(int nb_degen = 0; nb_degen<distrib_col.size(); nb_degen++){
		int idistrib = distrib_col.at(nb_degen)[0];
		// VorW va determiner si le pb est en ligne ou en colonen
		int VorW = distrib_col.at(nb_degen)[1];

		if(!(VorW==-2)){
			int nbToSample = ceil(percent*_Jc[idistrib]);
			std::random_device rdtest;     // only used once to initialise (seed) engine
			std::mt19937 rng(rdtest());    // random-number engine used (Mersenne-Twister in this case)
			std::uniform_int_distribution<int> uniW(0,(int)(_Jc[idistrib]-1)); // guaranteed unbiased
			std::uniform_int_distribution<int> unikc(0,(int)(_kc[idistrib]-1));
			for(int i = 0; i<nbToSample; i++){
				int column = uniW(rng);
				rowvec newSample(_kc[idistrib]);
				newSample.zeros();
				(this->_W[idistrib]).row(column) = newSample;

				int cluster = unikc(rng);
				this->_W[idistrib](column, cluster) = 1;

			}
		}

	}
	
}

void CoClusteringContext::noRowDegenerancy(vector<vector<int>> distrib_col){
	double percent = _percentRandomB[0]/100;
	int count = 0;
	for(int nb_degen = 0; nb_degen<distrib_col.size(); nb_degen++){
		//int idistrib = distrib_col.at(nb_degen)[0];
		// VorW va determiner si le pb est en ligne ou en colonen
		int VorW = distrib_col.at(nb_degen)[1];

		if(VorW==-2){
			count++;
			int nbToSample = ceil(percent*_Nr);
			std::random_device rdtest;     // only used once to initialise (seed) engine
			std::mt19937 rng(rdtest());    // random-number engine used (Mersenne-Twister in this case)
			std::uniform_int_distribution<int> uniW(0,(int)(_Nr-1)); // guaranteed unbiased
			std::uniform_int_distribution<int> unikr(0,(int)(_kr-1));
			for(int i = 0; i<nbToSample; i++){
				int line = uniW(rng);
				rowvec newSample(_kr);
				newSample.zeros();
				(this->_V).row(line) = newSample;

				int cluster = unikr(rng);
				this->_V(line, cluster) = 1;

			}
		}
		if(count>0) return;
	}
	
}

void CoClusteringContext::fillParameters(int iteration) {
	this->_allgamma.at(iteration) = this->_gamma;
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_allrho.at(iteration).at(idistrib) = this->_rho.at(idistrib);
		this->_distrib_objects[idistrib]->fillParameters(iteration);
	}
}

void CoClusteringContext::fillLabels(int iteration) {
	//_V.print();
	for(int i = 0; i<_Nr; i++){
		uvec tmp = find(_V.row(i)==1);
		int label = tmp(0);
		_zrchain(iteration, i) = label;
	}
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for(int d = 0; d<_Jc.at(idistrib); d++){
			uvec tmp = find(_W.at(idistrib).row(d)==1);
			int label = tmp(0);
			_zcchain.at(idistrib)(iteration,d) = label;
		}
	}
}

void CoClusteringContext::getBurnedParameters() {
	// gammas:
	rowvec gamma_result = conv_to<rowvec>::from(zeros(_kr));
	for (int i = _nbSEMburn; i < _nbSEM; i++) {
		for (int k = 0; k < _kr; k++) {
			gamma_result(k) += this->_allgamma.at(i)(k);
		}
	}
	this->_resgamma = gamma_result / (_nbSEM - _nbSEMburn);
	this->_gamma = this->_resgamma;

	// rhos:
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {

		rowvec resrho_distrib = conv_to<rowvec>::from(zeros(_kc[idistrib]));
		for (int i = _nbSEMburn; i < _nbSEM; i++) {
			for (int h = 0; h < _kc[idistrib]; h++) {
				resrho_distrib(h) += this->_allrho.at(i).at(idistrib)(h);
			}
		}
		this->_resrho.at(idistrib) = resrho_distrib / (_nbSEM - _nbSEMburn);
		this->_rho.at(idistrib) = this->_resrho.at(idistrib);
	}

	// distributions parameters
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->getBurnedParameters(this->_nbSEMburn);
	}
}

void CoClusteringContext::printResults() {
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->printResults();
	}
	_gamma.print();
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		_rho.at(idistrib).print();
	}
}

void CoClusteringContext::returnResults() {
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->returnResults();
	}
	_resgamma.print();
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		_resrho.at(idistrib).print();
	}
}

void CoClusteringContext::putParamsToZero() {
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		this->_distrib_objects[idistrib]->putParamsToZero();
	}
}

S4 CoClusteringContext::returnCoclustering() {
	S4 x("ResultCoclust");

    // partitions:
    x.slot("V")  = _V;
    List resultW(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		resultW[idistrib] = this->_W[idistrib];
	}
    x.slot("W") = resultW;


    // labels:
    vec zr = zeros(_Nr);
    for(int i=0; i<_Nr; i++){
    	uvec k = find(_V.row(i)==1);
    	zr(i) = k(0)+1;
    }
    x.slot("zr")  = zr;


    List resultzc(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	vec zc = zeros(_Jc[idistrib]);
    	for(int d=0; d<_Jc[idistrib]; d++){
    		uvec h = find(_W[idistrib].row(d)==1);
    		zc(d) = h(0)+1;
    	}
		resultzc[idistrib] = zc;
	}
	x.slot("zc") = resultzc;

	//labels chain
	x.slot("zrchain") = _zrchain;

	List zcchain(_number_distrib);
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		zcchain[idistrib] = _zcchain[idistrib];
	}
	x.slot("zcchain") = zcchain;


	// parameters: 
	List resultAlpha(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	
    	resultAlpha[idistrib] =	this->_distrib_objects[idistrib]->returnResults();
    	
	}
	x.slot("params") = resultAlpha;

	// parameters chain: 
	List resultAlphaChain(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	
    	resultAlphaChain[idistrib] =	this->_distrib_objects[idistrib]->returnParamsChain();
    	
	}
	x.slot("paramschain") = resultAlphaChain;

	//xhat
	List resultXhat(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	
    	resultXhat[idistrib] =	this->_distrib_objects[idistrib]->returnXhat();
    	
	}
	x.slot("xhat") = resultXhat;

	// mixing proportions: 
	x.slot("pi")  = this->_resgamma;
	List resultRho(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	
    	resultRho[idistrib] = this->_resrho[idistrib];
    	
	}
	x.slot("rho") = resultRho;

	List resultAllGamma(_nbSEM);
    for (int i = 0; i < _nbSEM; i++) {
    	
    	resultAllGamma[i] = this->_allgamma.at(i);
    	
	}
	x.slot("pichain") = resultAllGamma;
	// TODO : verifier que _allgamma est de la bonne dimension.

	List resultRhoChain(_number_distrib);
    for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
    	List resrho_distrib(_nbSEM);
    	for (int i = 0; i < _nbSEM; i++) {
    		resrho_distrib[i] = this->_allrho.at(i).at(idistrib);
    	}
    	resultRhoChain[idistrib] = resrho_distrib;
    	
	}
	x.slot("rhochain") = resultRhoChain;

	/*
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {

		rowvec resrho_distrib = conv_to<rowvec>::from(zeros(_kc[idistrib]));
		for (int i = _nbSEMburn; i < _nbSEM; i++) {
			for (int h = 0; h < _kc[idistrib]; h++) {
				resrho_distrib(h) += this->_allrho.at(i).at(idistrib)(h);
			}
		}
		this->_resrho.at(idistrib) = resrho_distrib / (_nbSEM - _nbSEMburn);
		this->_rho.at(idistrib) = this->_resrho.at(idistrib);
	}
	*/

	//icl
	x.slot("icl") = this->_icl;
	

    return(x);
	
}

double CoClusteringContext::computeICL() {
	double result = 0;
	result += -(_kr - 1) / 2 * log(_Nr);
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		result += -(_kc[idistrib] - 1) / 2 * log(_Jc[idistrib]);
	}


	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int d = 0; d < _Jc[idistrib]; d++)
		{
			for (int h = 0; h < _kc[idistrib]; h++)
			{
				if(_W[idistrib](d, h)==1){
					for (int i = 0; i < _Nr; i++)
					{
						for (int k = 0; k < _kr; k++)
						{
							if(_V(i, k)==1){
								result += _distrib_objects[idistrib]->computeICL(i, d, k, h);
							}
							
						}
					}
				}
				
			}
		}
	}

	
	
	for (int k = 0; k < _kr; k++) {
		result += accu(this->_V.col(k))*log(_resgamma(k));
	}
	
	for (int idistrib = 0; idistrib < _number_distrib; idistrib++) {
		for (int h = 0; h < _kc[idistrib]; h++) {
			result += accu(this->_W[idistrib].col(h))*log(_resrho[idistrib](h));
		}	
	}

	this->_icl = result;
	return(result);
}

/*======================================UTILS======================================*/


rowvec CoClusteringContext::getMeans(mat VorW) {
	rowvec result;
	result.zeros(VorW.n_cols);
	for (int i = 0; i < VorW.n_cols; i++)
	{
		colvec column = VorW.col(i);
		result(i) = mean(column);
	}
	return result;
}

double CoClusteringContext::logsum(rowvec logx) {
	// on fait ca pour l'erreur error: sort(): detected NaN
	logx.replace(datum::nan, -100000);
	if (logx.size() == 1) {
		return logx(0);
	}
	double result = 0;
	logx = sort(logx, "descend");
	double tmp = 1;
	for (int i = 1; i < logx.n_elem; i++) {
		tmp += exp(logx(i) - logx(0));
	}
	result = logx(0) + log(tmp);
	return(result);
}

mat CoClusteringContext::kmeansi() {

	mat result(_Nr, _kr);
	result.zeros();

	mat means;
	bool status = arma::kmeans(means, _x.t(), _kr, random_subset, 2, false);
	if (status == false)
	{
		return(result);
	}
	for (int i = 0; i < _Nr; i++) {
		int num_clust = -1;
		double dst_old = -1;
		double dst = -1;
		int leng = std::accumulate(_Jc.begin(), _Jc.end(), 0);
		for (int k = 0; k < _kr; k++) {
			vec a(leng);
			vec b(leng);
			for (int ireconstruct = 0; ireconstruct < means.col(k).n_elem; ireconstruct++) {
				a(ireconstruct) = means.col(k)(ireconstruct);
				b(ireconstruct) = _x.row(i)(ireconstruct);
			}
			dst = this->getDistance(a, b);
			if (dst_old < 0 || dst < dst_old) {
				dst_old = dst;
				num_clust = k;
			}
		}

		result(i, num_clust) = 1;
	}
	return result;
}

double CoClusteringContext::getDistance(vec &a, vec &b) {
	vec temp = a - b;
	return arma::norm(temp);

}

