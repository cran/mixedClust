setClass (
  "xhat",
  representation=representation(
    xhat="matrix"
  )
)
setClass (
  "W",
  representation=representation(
    W="matrix"
  )
)
setClass (
  "zc",
  representation=representation(
    zc="vector"
  )
)
setClass (
  "zcchain",
  representation=representation(
    zcchain="list"
  )
)
setClass (
  "zrchain",
  representation=representation(
    zrchain="vector"
  )
)
setClass (
  "rho",
  representation=representation(
    rho="vector"
  )
)
setClass (
  "rhochain",
  representation=representation(
    rhochain="list"
  )
)
setClass (
  "pichain",
  representation=representation(
    pichain="vector"
  )
)
setClass (
  "params",
  representation=representation(
    params="list"
  )
)
setClass (
  "paramschain",
  representation=representation(
    paramschain="list"
  )
)
setClass (
  "dlist",
  representation=representation(
    dlist="vector"
  )
)
setClass (
  "distrib_names",
  representation=representation(
    distrib_names="character"
  )
)
# Result for co-clustering
setClass (
  "ResultCoclust",
  
  # Defining slot type
  representation = representation (
    V = "matrix",
    zr = "vector",
    pi = "vector",
    icl = "numeric"
  ),
  contains=c("W","zc","rho","params","paramschain","xhat","pichain","rhochain","zrchain","zcchain")
)
