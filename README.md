UNDER CONSTRUCTION!!!!! This repo was started for my own benefit and work, but I am interested in building a package for the community to use!

# PhaseSkate - Scalable, High Performance Bayesian Inference In Native Julia

A native Julia implementation of the Stan approach to Bayesian inference. Built *for* [Enzyme](https://github.com/EnzymeAD/Enzyme.jl). Skate along the logjoint with ease!

## Design Philosophy

1. Speed - I built PhaseSkate because I needed a tool that could handle the large, messy and complex data+models of Biology. To do this, PhaseSkate was built **for** Enzyme with minimal external dependencies, with an emphasis on type stability and pure LPDF functions.
2. Clarity - the `@skate` that defines the Skate DSL is a complete description of the information needed by a reader (and the compiler!!) to understand the model. What this means in practice is that the information flowing into the model is not scattered across an analysis script. The `@skate` block, while technical, is visually distinct, cohesive, and defines everything. Stan got this right.

**Note:** DSL not finished yet. Right now in generates some not super pretty function names that I am working on simplifying.

```julia

using Pkg; Pkg.activate(@__DIR__)
using PhaseSkate


@skate MixtureModel begin
    @constants begin ## constants and data
        N::Int
        K::Int
        x::Vector{Float64}
    end

    @params begin ### these parameters will be sampled
        theta = param(Vector{Float64}, K; simplex = true)
        mus = param(Vector{Float64}, K, ordered = true)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin

        
        target += dirichlet_lpdf(theta, 1.0) # mixture weights
        target += multi_normal_diag_lpdf(mus, 0.0, 10.0) # means
        target += normal_lpdf(sigma, 0.0, 5.0) # shared var 

        # Likelihood
        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x[i], mus[j], sigma)
            end
        end
    end
end


K = 2
N = 50
x_data = vcat(randn(25) .- 2.0, randn(25) .+ 2.0)

d = MixtureModel_DataSet(N=N, K=K, x=x_data)
m = make_mixturemodel(d);

samples = sample(m, 1000; ϵ = 0.01, L = 10)

```


### What about Blackjax?

Bob Carpenter from the Stan project has written several times about Jax and the decline of laptop Bayesian inference. Certainly for mega-models, this is probably true. But my intuition tells me that the vast majority of Bayesian inference will continue to be done on local computers.

The hope of this project is that by letting Enzyme LLVM autodifferentiation shine on top of performant Julia code, a new generation of laptop Bayesian inference can be done natively inside Julia without having to work with C++.

Indeed, bayesian inference is an area where the Julia language should shine. HMC is intensive, CPU bound numerical computations of complex gradients and at its heart a physics simulation of a particle flying around phase-space. I built this package because I became increasingly upset that I always found myself returning to Stan.

## Example: Complex, high N, 177 Dimensional Model Survival Model 

Below is a complex joint survival model for tiered data in a genetic disease (model itself still a WIP). Clearly, complex (plate diagram available in this repo for the moment). Remarkably, on 3,000 datapoints PhaseSkate is able to sample **2,000 posterior draws in about 4 minutes**. In stan, the same model takes... substantially longer... to sample (via StanSample.jl, which should just be a wrapper). Stan is amazing and much faster than most alternatives for me, so I am very excited about this if it holds up.

1. The syntax should give Stan immediately. I like it! I think it provides structure and clarity.
2. NO TILDES! I think, contrary to many mathematicians, people in the applied sciences and particularly programmers find this syntax very confusing. It *obscures the fact that really all that is happening is a summation to the log joint function*. I think people will be able to, with greater ease, write more performant code if the abstraction of the tilde removes this confusion.
3. The @for macro -- probably to be renamed. I made this so that I could write broadcast syntax for clarity, while the macro essentially unrolls it into a non allocating for loop. Unlike the `@..` macro available already, I am trying to build this macro to handle multiple accumulators in a single `for i in 1:n` block. We'll see! Right now its working in this case!!


You can run this example for yourself with the joint_alm.jl script to the left with synthetic data (real data is private and protected). There is also a stan implementation. 



```julia

@skate JointALM begin
    @constants begin
        n1::Int
        n2::Int
        p::Int
        n_countries::Int
        MRC_MAX::Int
        tier1_times::Vector{Float64}
        tier1_X::Matrix{Float64}
        tier1_country_ids::Vector{Int}
        tier1_obs_idx::Vector{Int}
        tier1_cens_idx::Vector{Int}
        tier2_times::Vector{Float64}
        tier2_X::Matrix{Float64}
        tier2_country_ids::Vector{Int}
        tier2_obs_idx::Vector{Int}
        tier2_cens_idx::Vector{Int}
        total_mrc_obs::Int
        mrc_scores_flat::Vector{Int}
        mrc_times_flat::Vector{Float64}
        mrc_patient_ids::Vector{Int}
    end

    @params begin
        log_shape::Float64
        log_scale::Float64
        beta_s     = param(Vector{Float64}, p)
        beta_k     = param(Vector{Float64}, p)
        sigma_country_k = param(Float64; lower=0.0)
        sigma_country_s = param(Float64; lower=0.0)
        mu_country_k = param(Vector{Float64}, n_countries)
        mu_country_s = param(Vector{Float64}, n_countries)
        mu_k::Float64
        omega_k    = param(Float64; lower=0.0)
        gamma_k::Float64
        gamma_hill = param(Float64; lower=1.0)
        EC50       = param(Float64; lower=0.0, upper=1.0)
        log_phi::Float64
        P0         = param(Float64; lower=0.0, upper=1.0)
        z_k        = param(Vector{Float64}, n2)
    end

    @logjoint begin
        # ── Derived scalars  ──
        shape = exp(log_shape)
        inv_shape = 1.0 / shape
        phi = exp(log_phi)
        log_P0_ratio = log1p(-P0) - log(P0)
        log_EC50g = gamma_hill * log(EC50)

        # ── Priors ──
        target += normal_lpdf(log_shape, 0.2, 0.5)
        target += normal_lpdf(log_scale, 2.5, 1.0)
        target += multi_normal_diag_lpdf(beta_s, 0.0, 2.0)
        target += multi_normal_diag_lpdf(beta_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_k, 0.0, 0.5)
        target += cauchy_lpdf(sigma_country_s, 0.0, 1.0)
        target += multi_normal_diag_lpdf(mu_country_k, 0.0, sigma_country_k)
        target += multi_normal_diag_lpdf(mu_country_s, 0.0, sigma_country_s)
        target += normal_lpdf(mu_k, log(0.08), 0.5)
        target += cauchy_lpdf(omega_k, 0.0, 0.5)
        target += normal_lpdf(gamma_k, 1.0, 0.5)
        target += normal_lpdf(gamma_hill, 3.0, 1.0)
        target += beta_lpdf(EC50, 4.0, 6.0)
        target += normal_lpdf(log_phi, log(15.0), 0.5)
        target += beta_lpdf(P0, 2.0, 8.0)
        target += multi_normal_diag_lpdf(z_k, 0.0, 1.0)

        # ── Tier 2: log_k_2, log_eff_scale_2 ──
        @for begin
            log_k_2 = mu_k .+ (tier2_X * beta_k) .+ mu_country_k[tier2_country_ids] .+ (omega_k .* z_k)
            log_eff_scale_2 = log_scale .- ((tier2_X * beta_s) .+ mu_country_s[tier2_country_ids] .+ gamma_k .* log_k_2) .* inv_shape
        end

        for idx in tier2_obs_idx
            target += weibull_logsigma_lpdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end
        for idx in tier2_cens_idx
            target += weibull_logsigma_lccdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end

        # ── Tier 1: log_eff_scale_1 ──
        @for begin
            log_k_1 = mu_k .+ (tier1_X * beta_k) .+ mu_country_k[tier1_country_ids]
            log_eff_scale_1 = log_scale .- ((tier1_X * beta_s) .+ mu_country_s[tier1_country_ids] .+ gamma_k .* log_k_1) .* inv_shape
        end

        for idx in tier1_obs_idx
            target += weibull_logsigma_lpdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end
        for idx in tier1_cens_idx
            target += weibull_logsigma_lccdf(tier1_times[idx], shape, log_eff_scale_1[idx])
        end

        # ── MRC likelihood (scalar loop, zero extra alloc) ──
        for i in 1:total_mrc_obs
            k_i = exp(log_k_2[mrc_patient_ids[i]])
            P_t_i = 1.0 / (1.0 + exp(log_P0_ratio - k_i * mrc_times_flat[i]))
            log_Pg_i = gamma_hill * log(max(P_t_i, 1e-9))
            mu_i = clamp(1.0 / (1.0 + exp(log_Pg_i - log_EC50g)), 1e-6, 1.0 - 1e-6)
            a_mrc = mu_i * phi
            b_mrc = (1.0 - mu_i) * phi
            target += beta_binomial_lpdf(mrc_scores_flat[i], MRC_MAX, a_mrc, b_mrc)
        end
    end
end

```
