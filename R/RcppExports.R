# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

coclust <- function(xMat, myList, distrib_names, kr, kc, init, nbSEM, nbSEMburn, nbRepeat, nbindmini, m, functionalDataV, zrinit, zcinit, percentRandomB, percentRandomP) {
    .Call('_mixedClust_coclust', PACKAGE = 'mixedClust', xMat, myList, distrib_names, kr, kc, init, nbSEM, nbSEMburn, nbRepeat, nbindmini, m, functionalDataV, zrinit, zcinit, percentRandomB, percentRandomP)
}

