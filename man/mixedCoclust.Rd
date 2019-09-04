\name{mixedCoclust}
\alias{mixedCoclust}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Function to perform a co-clustering
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
This function performs a co-clustering on heterogeneous data sets by using the Multiple Latent Block model (cf references for further  details). 
}
\usage{
mixedCoclust(x=matrix(0,nrow=1,ncol=1), idx_list=c(1), distrib_names,
          kr, kc, init, nbSEM, nbSEMburn, nbRepeat=1, nbindmini, m=0, 
          functionalData=array(0, c(1,1,1)), zrinit= 0 , zcinit=0, 
          percentRandomB=0, percentRandomP=0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
  %%     ~~Describe \code{x} here~~
  Data matrix, of dimension N*Jtot. The features with same type should be aside. The missing values should be coded as NA.
  }
  \item{idx_list}{
  %%     ~~Describe \code{x} here~~
  Vector of length D. This argument is useful when variables are of different types. Element d should indicate where the variables of type d begins in matrix x.
  }
  \item{distrib_names}{
  Vector of length D. indicates the type of distribution to use. Must be among "Gaussian", "Multinomial", "BOS", "Poisson" or "Functional". Functional data must always be at the end.
  }
  \item{kr}{
  %%     ~~Describe \code{kr} here~~
  Number of row classes.
  }
  \item{kc}{
  %%     ~~Describe \code{kc} here~~
  Vector of length D. d^{th} element indicates the number of column clusters. 
  }
  \item{m}{
  %%     ~~Describe \code{m} here~~
  Vector of length D. d^{th} element defines the ordinal and categorical data's number of levels.
  }
  \item{functionalData}{
    Data tensor of dimension N*J*T.
  }
  \item{nbSEM}{
  %%     ~~Describe \code{nbSEM} here~~
  Number of SEM-Gibbs iterations realized to estimate parameters.
  }
  \item{nbSEMburn}{
  %%     ~~Describe \code{nbSEMburn} here~~
  Number of SEM-Gibbs burning iterations for estimating parameters. This parameter must be inferior to nbSEM.
  }
  \item{nbRepeat}{
  %%     ~~Describe \code{nbSEMburn} here~~
  Number of times sampling on rows and on colums will be done at each SEM-Gibbs iteration.
  }
  \item{nbindmini}{
  %%     ~~Describe \code{nbindmini} here~~
  Minimum number of cells belonging to a block.
  }
  \item{init}{
  %%     ~~Describe \code{init} here~~
  String that indicates the kind of initialisation. Must be one of th following words :  "kmeans", "random", "provided", "randomParams" or "randomBurnin".
  }
  \item{zrinit}{
  Vector of length N. When init="provided", indicates the labels of each row.
  }
  \item{zcinit}{
  Vector of length Jtot. When init="provided", indicates the labels of each column.
  }
  \item{percentRandomB}{
  Vector of length 2. Indicates the percentage of resampling when init is equal to "randomBurnin".
  }
  \item{percentRandomP}{
  Vector of length 2. Indicates the percentage of resampling when init is equal to "randomParams".
  }
}

%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
\item{@V}{Matrix of dimension N*kr such that V[i,g]=1 if i belongs to cluster g.}
\item{@icl}{ICL value for co-clustering.}\item{@name}{}
\item{@paramschain}{List of length nbSEMburn. For each iteration of the SEM-Gibbs algorithm, the parameters of the blocks are stored.}
\item{@pichain}{List of length nbSEM. Item i is a vector of length kr which contains the row mixing proportions at iteration i.}
\item{@rhochain}{List of length nbSEM. Item i is a list of length D whose d^{th} contains the column mixing proportions of groups of variables d, at iteration i.}
\item{@zc}{List of length D. d^{th} item is a vector of length J[d] representing the columns partitions for the group of variables d.}
\item{@zr}{Vector of length N with resulting row partitions.}
\item{@W}{List of length D. Item d is a matrix of dimension J*kc[d] such that W[j,h]=1 if j belongs to cluster h.}
\item{@m}{Vector of length D. d^{th} element represents the number of levels of d^{th} group of variables.}
\item{@params}{List of length D. d^{th} item represents the blocks paramaters for group of variables d.}
\item{@pi}{Vector of length kr. Row mixing proportions.}
\item{@rho}{List of length D. d^{th} item represents the column mixing proportion for d^{th} group of variables.}
\item{@xhat}{List of length D. d^{th} item represents the d^{th} group of variables dataset, with missing values completed.}
\item{@zrchain}{Matrix of dimension nbSEM*N. Row i represents the row cluster partitions at iteration i.}  
\item{@zrchain}{List of length D. Item d is a matrix of dimension nbSEM*J[d]. Row i represents the column cluster partitions at iteration i.}

}
%%\references{
%% ~put references to the literature/web site here ~
%%}
\author{
%%  ~~who you are~~
Margot Selosse, Julien Jacques, Christophe Biernacki.
}
%%\note{
%%  ~~further notes~~
%%}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
  
    data(M1)
    nbSEM=30
    nbSEMburn=20
    nbindmini=1
    init = "random"

    kr=2
    kc=c(2,2,2)
    m=c(6,3)
    d.list <- c(1,41,81)
    distributions <- c("Multinomial","Gaussian","Bos")
    res <- mixedCoclust(x = M1, idx_list = d.list,distrib_names = distributions,
                        kr = kr, kc = kc, m = m, init = init,nbSEM = nbSEM,
                        nbSEMburn = nbSEMburn, nbindmini = nbindmini)
  
  
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{"mixed-type-data"}% use one of  RShowDoc("KEYWORDS")
\keyword{"co-clustering"}% __ONLY ONE__ keyword per line
