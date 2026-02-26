module Skate

using SpecialFunctions: loggamma, logbeta, erfinv, gamma_inc
using PositiveFactorizations: cholesky
using LinearAlgebra: dot, diag
using Enzyme
using Statistics
using Printf: @sprintf
using Random: randn, randn!

import Base.@kwdef

include("bijections.jl")
include("lpdfs.jl")
include("utilities.jl")
include("welford.jl")
include("logdensitygenerator.jl")
include("chains.jl")
include("hmc.jl")
include("lang.jl")
include("sbc.jl")

export @spec, sample, log_prob, ModelLogDensity,
       SamplerBackend, CPUBackend, XLABackend, _sample_impl,
       sbc, SBCResult, calibrated,
       Chains, samples, mean, ci, thin, min_ess,
       cholesky,
       # constraint types & transforms (used by @spec generated code)
       _qscalar, transform, log_abs_det_jacobian,
       IdentityConstraint, LowerBounded, UpperBounded, Bounded,
       SimplexConstraint, OrderedConstraint,
       simplex_transform, ordered_transform,
       corr_cholesky_transform, corr_cholesky_transform!,
       # log-pdf functions
       log_sum_exp, log_mix,
       multi_normal_cholesky_lpdf,
       multi_normal_cholesky_scaled_lpdf,
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
       lognormal_lpdf,
       student_t_lpdf,
       dirichlet_lpdf,
       uniform_lpdf,
       laplace_lpdf,
       logistic_lpdf,
       beta_lpdf,
       lkj_corr_cholesky_lpdf,
       diag_pre_multiply

end # module Skate
