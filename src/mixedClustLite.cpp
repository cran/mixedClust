// mixedData.cpp : définit le point d'entrée pour l'application console.
//


#include "Distribution.h"
#include "CoClusteringContext.h"


#include "LogProbs.h"
#include "TabProbsResults.h"
#include <iostream>
#include <vector>
//#include "Rcpp.h"

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppProgress)]]
#include <progress.hpp>
#include <progress_bar.hpp>



// [[Rcpp::depends(RcppArmadillo)]] 
//#include <armadillo>
#include <limits>
#include <cmath>
#include <list>
#include <typeinfo>
#include <iostream>
#include <initializer_list> 
#include <vector>
#include <numeric>




using namespace std;
using namespace arma;
using namespace Rcpp;


//[[Rcpp::plugins(cpp11)]]


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
S4 coclust(NumericMatrix& xMat, std::vector<unsigned int> myList, std::vector<std::string> distrib_names,
	int kr, std::vector<int> kc, std::string init, int nbSEM, int nbSEMburn, int nbRepeat, int nbindmini, 
	const std::vector<int> m, NumericVector functionalDataV, std::vector<int> zrinit, std::vector<int> zcinit, 
	std::vector<double> percentRandomB, std::vector<double> percentRandomP)
{
    

	Progress p(nbSEM, true);

	Rcpp::IntegerVector x_dims = functionalDataV.attr("dim");
	arma::cube functionalData(functionalDataV.begin(), x_dims[0], x_dims[1], x_dims[2]);
	//functionalData.print();


	arma::mat x(xMat.begin(), xMat.nrow(), xMat.ncol(), false ) ;


	int dlistSize = myList.size();
	vector<urowvec> dlist(dlistSize);


	arma::urowvec tmp;
	for(int i=0; i<dlistSize;i++){
		if(i==(dlistSize-1)){
			tmp = linspace<urowvec>(myList[i], x.n_cols-1, x.n_cols - myList[i]);
			dlist[i] = tmp;
		}
		else{
			tmp = linspace<urowvec>(myList[i], myList[i+1]-1, myList[i+1] - myList[i]);
			dlist[i] = tmp;
		}

	}

	CoClusteringContext context(x, dlist, distrib_names, kr, kc, 
		init, nbSEM, nbSEMburn, nbindmini, m, functionalData, 
		zrinit, zcinit, percentRandomB, percentRandomP);

	context.missingValuesInit();

	string initialization = context.getInitialization();


	bool verif = context.initialization();

		
	if (!verif) {
		S4 t("ResultCoclust");
		return t;
	}

	
	context.imputeMissingData();

	context.fillParameters(0);
	if(initialization != "randomParams"){
		context.fillLabels(0);
	}


	//context.printResults();
	

	// << "after fillParameters and fillLabels" << endl;
	
	for (int iter = 0; iter < nbSEM; iter++) {
		p.increment();
		verif = false;




			for(int repeat=0; repeat<nbRepeat; repeat++){
				
					context.SEstepRow();
					context.sampleV();
					
					verif = context.verif();
					int restart = 0;

					if(initialization!="randomBurnin" && initialization!="randomParams"){
						while (verif == false && restart < 50) {
							context.sampleV();
							restart++;		
							verif = context.verif();
						}
					}
					if (!verif) {
						if((initialization!="randomBurnin" && initialization!="randomParams")||iter>nbSEMburn){
							S4 t("ResultCoclust");
							return t;
						}
						else{
							// todo implement verification for cols
							vector<vector<int>> res = context.verification();
							context.noRowDegenerancy(res);
							context.MstepVW();
							verif = context.verif();
						}
					}
					else{
						context.MstepVW();
					}

			}

			//context.printResults();

			//
			for(int repeat=0; repeat<nbRepeat; repeat++){
				
					context.SEstepCol();
					context.sampleW();
					
					verif = context.verif();
					int restart = 0;

					if(initialization!="randomBurnin" && initialization!="randomParams"){
						while (verif == false && restart < 50) {
							context.sampleW();
							restart++;		
							verif = context.verif();
						}
					}
					if (!verif) {
						if((initialization!="randomBurnin" && initialization!="randomParams")||iter>nbSEMburn){
							S4 t("ResultCoclust");
							return t;
						}
						else{
							// todo implement verification for cols
							vector<vector<int>> res = context.verification();
							context.noColDegenerancy(res);
							context.MstepVW();
							verif = context.verif();
						}
					}
					else{
						context.MstepVW();
					}

			}
		
		
		
		if (!verif) {
			S4 t("ResultCoclust");
			return t;
		}
		else {
			context.imputeMissingData();
		}
		if(iter>0 || initialization=="randomParams"){
			context.fillParameters(iter);
			context.fillLabels(iter);
		}


			
	} // end iter
	
	
	context.getBurnedParameters();
	
	
	context.SEstepRow();
	context.SEstepCol();  // TODO : remettre // added so that probaV is actualized after burned params
	context.sampleVWStock(); // TODO : remettre
	context.computeICL();


	
    S4 t = context.returnCoclustering();
    //context.disposeObject();
    return(t);
}



double logsum(rowvec logx) {
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
double logsum(rowvec logx);




