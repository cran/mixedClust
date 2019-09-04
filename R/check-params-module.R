checkParamsCoclust <- function(x, distrib_names, init, nbSEM, nbSEMburn, functionalData){
	##### generic things #######
	if(any(distrib_names=="SparsePoisson")) stop('"SparseGaussian" is not supported with clustering')
	generical(x, distrib_names, init, nbSEM, nbSEMburn, functionalData)
}




######################## UTILS ######################
generical <- function(x, distrib_names, init, nbSEM, nbSEMburn, functionalData){
	tmp_comp = distrib_names %in% c("Bos","Gaussian","Poisson","Multinomial","Functional",
		"SparsePoisson","SparsePoissonExtension","SparseGaussian")
	if(any(distrib_names==FALSE)) stop('distribution names should be among "Bos","Gaussian","Poisson","Multinomial", "SparsePoisson" or "Functional".')
	if(!length(x)>1 && !all(distrib_names=="Functional")) stop('Error with data matrix.')
	if(!any(init=="kmeans") & !any(init=="random") & !any(init=="provided") & !any(init=="randomBurnin")) 
		stop('init argument should be "kmeans", "random", "randomBurnin", or "provided".')
	if(nbSEM<=nbSEMburn) stop('nbSEMburn must be inferior to nbSEM.')

	if( length(functionalData) > 1 ){ # in case there are functional data
		# if there are functionalData but not in distrib_names
		if(!any(distrib_names == "Functional")) stop('functionalData should not be given as an argument if there is no fuctional data.')
		if(init=="kmeans") stop('When functional data is involved, a random init must be used.')
	}
	else{ # in case there is no functional data
		if(any(distrib_names == "Functional")) stop('functionalData should be given as an argument.')
	}

}

