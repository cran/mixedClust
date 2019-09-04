mixedCoclust <- function(x=matrix(0,nrow=1,ncol=1), idx_list=c(1), distrib_names,
 kr, kc, init, nbSEM, nbSEMburn, nbRepeat=1, nbindmini, m=0, functionalData=array(0, c(1,1,1)), 
 zrinit= 0 , zcinit=0, percentRandomB=0, percentRandomP=0){

 	idx_list = idx_list - 1
	checkParamsCoclust(x, distrib_names, init, nbSEM, nbSEMburn, functionalData)
	if(length(functionalData)>1){
		functionalData = transformFDA(functionalData)
	}
	#print(functionalData)
	res <- coclust(xMat=x, myList=idx_list, distrib_names, kr, kc, init, nbSEM, 
		nbSEMburn, nbRepeat, nbindmini, m=m, functionalDataV=functionalData,
		zrinit=zrinit, zcinit=zcinit, percentRandomB=percentRandomB, percentRandomP=percentRandomP)
	if(length(res@icl)==0){
		warning('Error: probably empty clusters')
	}
	return(res)
}