transformFDA <- function(functionalData){

	nrows=dim(functionalData)[1]
	ncols=dim(functionalData)[2]
	nslices=dim(functionalData)[3]
	# functionalData is an array with three dimensions, we change it in an aray of 2 dim
	functionalData2 <- matrix(0, nrow= nrows*ncols, ncol=nslices)
	iter = 1
	for(i in 1:nrows){
		for(j in 1:ncols){
			functionalData2[iter,] = functionalData[i,j,]
			iter=iter+1
		}
	}
	base.t <- fda::create.fourier.basis(c(0, 1), nbasis=31) 
	lent<-50
	tt<-seq(0,1,len=lent)
	fd <- fda::smooth.basis(tt, t(functionalData2),base.t)$fd
	acpf <- fda::pca.fd(fd, nharm=10, centerfns=FALSE)
	
	resultCube = array(0, c(nrows, ncols, 5))
	iter = 1
	for(i in 1:nrows){
		for(j in 1:ncols){
			resultCube[i,j,] = acpf$scores[iter,1:5]
			iter=iter+1
		}
	}
	return(resultCube)

}