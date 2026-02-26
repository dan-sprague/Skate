module Skate

using SkateBase
using Enzyme
using Random: randn, randn!, Xoshiro
using LinearAlgebra: dot
using Statistics
using Printf: @sprintf
using SpecialFunctions: gamma_inc

import Base.@kwdef
import SkateBase: mean

include("welford.jl")
include("safe_grads.jl")
include("hmc.jl")
include("lang.jl")
include("sbc.jl")

export @skate, sample, log_prob, ModelLogDensity,
       sbc, SBCResult, calibrated,
       # re-export from SkateBase
       Chains, samples, mean, ci, thin, min_ess,
       cholesky,
       transform, log_abs_det_jacobian,
       IdentityConstraint, LowerBounded, UpperBounded, Bounded,
       SimplexConstraint, OrderedConstraint,
       simplex_transform, ordered_transform,
       corr_cholesky_transform, corr_cholesky_transform!,
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
