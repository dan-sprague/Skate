"""
    PhaseSkate

Scalable, high-performance Bayesian inference in native Julia.
Built for Enzyme LLVM autodifferentiation.

# Core workflow
1. Define a model with `@skate`
2. Build with `make(data)` → `ModelLogDensity`
3. Sample with `sample()`, `sample_mclmc()`, or `sample_adjusted_mclmc()`
4. Inspect results via `Chains`: `mean()`, `ci()`, `min_ess()`
"""
module PhaseSkate

using Enzyme
using Random: randn, randn!, Xoshiro
using LinearAlgebra: dot, mul!, cholesky, Symmetric, I, LowerTriangular, BLAS, diagm
using Statistics
using Printf: @sprintf
using SpecialFunctions: gamma_inc, loggamma, logbeta, erfinv

import Base.@kwdef
import Statistics: mean

include("utilities.jl")
include("bijections.jl")
include("lpdfs.jl")
include("welford.jl")
include("safe_grads.jl")
include("hmc.jl")
include("chains.jl")
include("lang.jl")
include("adjoints.jl")
include("sbc.jl")
include("compile.jl")

"""
    app(; kwargs...)

Launch the PhaseSkate IDE — a terminal-based environment for Bayesian modeling.
Requires Tachikoma.jl to be loaded (`using Tachikoma`).
"""
function app end

"""
    dashboard(model::ModelLogDensity, num_samples; kwargs...) → Chains
    dashboard(chains::Chains; sbc=nothing)

Launch the IDE with a pre-compiled model or existing results.
Requires Tachikoma.jl to be loaded (`using Tachikoma`).
"""
function dashboard end

export app, dashboard, compile,
       @skate, make, sample, log_prob, ModelLogDensity,
       sbc, SBCResult, calibrated,
       Chains, samples, mean, ci, thin, min_ess, diagnostics,
       cholesky,
       transform, log_abs_det_jacobian,
       IdentityConstraint, LowerBounded, UpperBounded, Bounded,
       SimplexConstraint, OrderedConstraint,
       simplex_transform, simplex_transform!, ordered_transform, ordered_simplex_matrix!,
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
       weibull_logsigma_lpdf_sum,
       categorical_logit_lpdf,
       weibull_logsigma_lccdf,
       weibull_logsigma_lccdf_sum,
       normal_lpdf_sum,
       lognormal_lpdf,
       student_t_lpdf,
       dirichlet_lpdf,
       uniform_lpdf,
       laplace_lpdf,
       logistic_lpdf,
       beta_lpdf,
       lkj_corr_cholesky_lpdf,
       correlated_topic_lpdf,
       diag_pre_multiply

end # module PhaseSkate
