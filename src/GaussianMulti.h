#pragma once
#include "Distribution.h"
#define _USE_MATH_DEFINES
#include <math.h>

class GaussianMulti :
	public Distribution
{
public:
	GaussianMulti(cube xsepC, int kr, int kc, int nbSEM);
	void missingValuesInit();
	TabProbsResults SEstep(const mat& V, const mat& W);
	mat SEstepRow(const mat& W);
	mat SEstepRowRandomParamsInit(mat& Wsample, uvec& colSample);
	mat SEstepCol(const mat& V);
	LogProbs SEstep_predict(int i, int d, int k, int h, double x_id, double sumi, double sumd, vec x_id_vec);
	void imputeMissingData(const mat& V, const mat& W);
	void Mstep(uvec rowind, uvec colind, int k, int h, bool init);
	void MstepVW(const mat& V, const mat& W, bool init);
	void fillParameters(int iteration);
	void getBurnedParameters(int burn);
	void printResults();
	List returnResults();
	List returnParamsChain();
	void putParamsToZero();
	double computeICL(int i, int d, int k, int h);
	void MstepInitRandomParams(mat xsample, mat Vsample, mat Wsample);
	mat colkmeans();
	GaussianMulti();
	~GaussianMulti();
protected:
	double densityMulti(vec x, mat covMat, vec meanVec, bool logd = true);
	cube _xsepC;
	int _nbDim;
	mat _sigmas;
	mat _mus;
	mat _ressigmas;
	mat _resmus;
	cube _allsigmas;
	cube _allmus;
};

/*
-------------------------------------------------------------------
Note on the sigmas and mus:
-------------------------------------------------------------------

+ Since we are in a multivariate case, the sigma of a co-cluster
should be a matrix of dimension hxh (and not a real). To store
the sigmas, we chose to put them aside in a matrix. For example,
if kr = 3, and kc = 2:
------------------------
|   sigma   |   sigma  |
|  block11  |  block12 |
------------------------
|   sigma   |   sigma  |
|  block21  |  block22 |
------------------------
|   sigma   |   sigma  |
|  block31  |  block32 |
------------------------
This matrix will be of dimension (h*kr)x(h*kc).

+ For the same reason, the co-cluster's mu is a vector of length h.
To store it, we chose to represent it as a matrix:
------------------------
|     mu    |     mu   |
|  block11  |  block12 |
------------------------
|     mu    |     mu   |
|  block21  |  block22 |
------------------------
|     mu    |     mu   |
|  block31  |  block32 |
------------------------
This matrix will be of dimension (kr)x(h*kc).

*/

