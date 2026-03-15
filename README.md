# PhaseSkate

High-performance HMC Bayesian inference in native Julia. An implementation of, and love letter to, the Stan approach to PPLs. Except no tildes.

Built for [Enzyme](https://github.com/EnzymeAD/Enzyme.jl).

> Early-stage — API may change. Contributions and feedback welcome!

## Why PhaseSkate?

PhaseSkate started off more as a pretty simple question, really. Given a performant logpdf function, how fast can Julia + Enzyme AD NUTS draw posterior samples? 

Things spiraled when I decided Enzyme had to be able to do static analysis of the gradient on an arbtitrary *lpdf* at compile time, ie no `set_runtime_analysis`. This meant that my logpdf could not be passed through any generic interfaces, which essentially led to the package you see now. In practice, this means that PhaseSkate has its own implementation of Stan's NUTS algorithm, as well as its own implementation of `lpdf` functions, which in the examples you will see are called in the style of Stan. 

**Features**

1. Built for Enzyme and Enzyme only, meaning static analysis on lpdfs at Enzyme compile time. JET.jl is used to catch generics/invalid type signatures in the logjoint function.
2. Stan-like PPL: Stan's PPL is actually quite valuable for writing performant models. PhaseSkate both encourages clarity and type stability in the model definition, and also verifies that all functions inside the logjoint are statically analyzable/return concrete Float64.
3. Live chain traces and diagnostics in a `Tachikoma.jl` powered TUI!
4. Narrow focus -- no discrete sampling! See examples and tutorials for how to sample models with discrete variables, or use the excellent `Turing.jl` ecosystem.

5. `@for` - keep readable broadcast math that gets rewritten under the hood as a direct accumulator into `target`. Allows you to accumulate multiple things sharing a common axis of iteration in a single for loop while keeping things legible :).
```julia
@for begin
    log_k_2 = mu_k .+ (tier2_X * beta_k) .+ mu_country_k[tier2_country_ids] .+ (omega_k .* z_k)
    log_eff_scale_2 = log_scale .- ((tier2_X * beta_s) .+ mu_country_s[tier2_country_ids] .+ gamma_k .* log_k_2) .* inv_shape
end
```

## Installation

```bash
git clone https://github.com/dansprague/PhaseSkate.jl
```

```julia
using Pkg
Pkg.develop(path="path/to/PhaseSkate.jl")
```

## Getting Started

```julia
using PhaseSkate

# Define a model
@skate NormalModel begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        for i in 1:N
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end

# Generate data and fit
y_data = randn(100) .* 2.0 .+ 3.0
d = NormalModelData(N=100, y=y_data)
m = make(d)

ch = sample(m, 2000; warmup=1000, chains=4)

# Inspect results
mean(ch, :mu)        # posterior mean
ci(ch, :mu)          # 95% credible interval
min_ess(ch)          # minimum effective sample size
```

## Example: Gaussian Mixture Model

```julia
@skate MixtureModel begin
    @constants begin
        N::Int
        K::Int
        x::Vector{Float64}
    end
    @params begin
        theta = param(Vector{Float64}, K; simplex = true)
        mus = param(Vector{Float64}, K; ordered = true)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        target += dirichlet_lpdf(theta, 1.0)
        target += multi_normal_diag_lpdf(mus, 0.0, 10.0)
        target += normal_lpdf(sigma, 0.0, 5.0)

        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x[i], mus[j], sigma)
            end
        end
    end
end
```

## Example: Complex Joint Survival Model (177 dimensions)

The `@for` macro converts broadcast syntax into zero-allocation for loops — readable vectorized notation with the performance of hand-written scalar code.

```julia
@skate JointALM begin
    @constants begin
        n1::Int; n2::Int; p::Int; n_countries::Int; MRC_MAX::Int
        tier1_times::Vector{Float64}; tier1_X::Matrix{Float64}
        tier1_country_ids::Vector{Int}
        tier1_obs_idx::Vector{Int}; tier1_cens_idx::Vector{Int}
        tier2_times::Vector{Float64}; tier2_X::Matrix{Float64}
        tier2_country_ids::Vector{Int}
        tier2_obs_idx::Vector{Int}; tier2_cens_idx::Vector{Int}
        total_mrc_obs::Int; mrc_scores_flat::Vector{Int}
        mrc_times_flat::Vector{Float64}; mrc_patient_ids::Vector{Int}
    end

    @params begin
        log_shape::Float64; log_scale::Float64
        beta_s = param(Vector{Float64}, p)
        beta_k = param(Vector{Float64}, p)
        sigma_country_k = param(Float64; lower=0.0)
        sigma_country_s = param(Float64; lower=0.0)
        mu_country_k = param(Vector{Float64}, n_countries)
        mu_country_s = param(Vector{Float64}, n_countries)
        mu_k::Float64
        omega_k = param(Float64; lower=0.0)
        gamma_k::Float64
        gamma_hill = param(Float64; lower=1.0)
        EC50 = param(Float64; lower=0.0, upper=1.0)
        log_phi::Float64
        P0 = param(Float64; lower=0.0, upper=1.0)
        z_k = param(Vector{Float64}, n2)
    end

    @logjoint begin
        shape = exp(log_shape)
        inv_shape = 1.0 / shape
        # ... priors ...

        # @for unrolls broadcasts into zero-allocation scalar loops
        @for begin
            log_k_2 = mu_k .+ (tier2_X * beta_k) .+ mu_country_k[tier2_country_ids] .+ (omega_k .* z_k)
            log_eff_scale_2 = log_scale .- ((tier2_X * beta_s) .+ mu_country_s[tier2_country_ids] .+ gamma_k .* log_k_2) .* inv_shape
        end

        for idx in tier2_obs_idx
            target += weibull_logsigma_lpdf(tier2_times[idx], shape, log_eff_scale_2[idx])
        end
        # ... tier 1 + MRC likelihood ...
    end
end
```

On 3,500 datapoints with 177 parameters, PhaseSkate samples 2,000 posterior draws in ~4 minutes on a laptop.

See `examples/joint_alm.jl` for the complete runnable script.

## Samplers

- **HMC/NUTS** — `sample(model, n; warmup, chains)` — Classic No-U-Turn sampler with dual averaging
- **MCLMC** — `sample_mclmc(model, n; ...)` — Microcanonical Langevin Monte Carlo (Robnik & Seljak, 2023)
- **Adjusted MCLMC** — `sample_adjusted_mclmc(model, n; ...)` — MCLMC with Metropolis-Hastings correction

## Benchmarks

Cross-language benchmarks live in a separate repo: [PhaseSkateBenchmark](https://github.com/dansprague/PhaseSkateBenchmark).

## More Examples

See the `examples/` directory:
- `lda.jl` — Latent Dirichlet Allocation
- `mixture_model.jl` — K=2 Gaussian mixture with simplex weights
- `logistic_regression.jl` — Bayesian logistic regression
- `hierarchical_normal.jl` — Eight Schools (Rubin, 1981)
- `joint_alm.jl` — Complex joint survival model

## Testing

```julia
using Pkg
Pkg.test("PhaseSkate")
```

## License

MIT
