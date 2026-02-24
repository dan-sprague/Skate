module Skate

using SpecialFunctions: loggamma, logbeta
using PositiveFactorizations: cholesky
using LinearAlgebra: dot,diag
using Enzyme

include("bijections.jl")
include("lpdfs.jl")
include("welford.jl")
include("logdensitygenerator.jl")
include("hmc.jl")


export cholesky


export multi_normal_cholesky_lpdf,
       multi_normal_diag_lpdf,
       normal_lpdf,
       cauchy_lpdf,
       exponential_lpdf,
       gamma_lpdf,
       poisson_lpdf,
       binomial_lpdf,
       beta_binomial_lpdf,
       weibull_lpdf,
       neg_binomial_2_lpdf,
       bernoulli_logit_lpdf,
       weibull_logsigma_lpdf,
       categorical_logit_lpdf,
       weibull_logsigma_lccdf,
       lognormal_lodf,
       student_t_lpdf,
       dirichlet_lpdf,
       uniform_lpdf,
       laplace_lpdf,
       logistic_lpdf,
       beta_lpdf,
       lkj_corr_cholesky_lpdf,
       diag_pre_multiply,
       corr_cholesky_transform

end # Skater module