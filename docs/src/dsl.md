# Model DSL

The `@skate` macro defines a complete Bayesian model in a single block with three sections.

## `@constants`

Declares the data/constants that are fixed for a given dataset. These become fields of the generated data struct.

```julia
@constants begin
    N::Int
    K::Int
    X::Matrix{Float64}
    y::Vector{Float64}
    group_ids::Vector{Int}
end
```

Supported types: `Int`, `Float64`, `Vector{Float64}`, `Vector{Int}`, `Matrix{Float64}`.

## `@params`

Declares the parameters to be sampled. Each parameter can be:

**Unconstrained scalar:**
```julia
mu::Float64
```

**Constrained scalar:**
```julia
sigma = param(Float64; lower=0.0)          # positive
rho = param(Float64; lower=-1.0, upper=1.0) # bounded
gamma = param(Float64; lower=1.0)           # lower-bounded
```

**Vectors and matrices:**
```julia
beta = param(Vector{Float64}, P)            # P-dimensional vector
mu_group = param(Vector{Float64}, K)        # K-dimensional vector
B = param(Matrix{Float64}, K, P)            # K×P matrix
```

**Special constraints:**
```julia
theta = param(Vector{Float64}, K; simplex=true)   # K-simplex (sums to 1)
mus = param(Vector{Float64}, K; ordered=true)      # ordered (ascending)
```

## `@logjoint`

The log-joint density function. Accumulate contributions via `target +=`:

```julia
@logjoint begin
    # Priors
    target += normal_lpdf(mu, 0.0, 10.0)
    target += multi_normal_diag_lpdf(beta, 0.0, 2.5)

    # Likelihood
    for i in 1:N
        eta = alpha + X[i, :] * beta
        target += bernoulli_logit_lpdf(y[i], eta)
    end
end
```

### The `@for` Macro

Inside `@logjoint`, the `@for` macro converts broadcast expressions into zero-allocation scalar loops. This gives you readable vectorized notation with the performance of hand-written loops:

```julia
@for begin
    log_k = mu_k .+ (X * beta_k) .+ mu_group[group_ids]
    log_scale = log_s .- (gamma .* log_k) .* inv_shape
end
```

This expands to:
```julia
__matvec_1 = X * beta_k
log_k = Vector{Float64}(undef, N)
log_scale = Vector{Float64}(undef, N)
@inbounds @simd for i in 1:N
    log_k[i] = mu_k + __matvec_1[i] + mu_group[group_ids[i]]
    log_scale[i] = log_s - (gamma * log_k[i]) * inv_shape
end
```

The `@for` macro handles:
- Element-wise broadcast operations (`.+`, `.-`, `.*`, `.÷`)
- Matrix-vector products (`X * beta`)
- Fancy indexing (`v[ids]`)
- Broadcast function calls (`exp.(x)`)
- Multiple chained assignments in a single loop
- Column-sliced matrix-vector products

### `log_mix` for Mixture Models

Use `log_mix` for numerically stable mixture log-densities:

```julia
for i in 1:N
    target += log_mix(theta) do j
        normal_lpdf(x[i], mus[j], sigma)
    end
end
```

## Available Log-Density Functions

All functions are pure arithmetic (Enzyme-safe, zero allocations):

| Function | Distribution |
|----------|-------------|
| `normal_lpdf(x, μ, σ)` | Normal |
| `cauchy_lpdf(x, μ, γ)` | Cauchy |
| `student_t_lpdf(x, ν, μ, σ)` | Student-t |
| `exponential_lpdf(x, λ)` | Exponential |
| `gamma_lpdf(x, α, β)` | Gamma |
| `beta_lpdf(x, α, β)` | Beta |
| `lognormal_lpdf(x, μ, σ)` | Log-Normal |
| `uniform_lpdf(x, a, b)` | Uniform |
| `laplace_lpdf(x, μ, b)` | Laplace |
| `logistic_lpdf(x, μ, s)` | Logistic |
| `weibull_lpdf(x, α, σ)` | Weibull |
| `weibull_logsigma_lpdf(x, α, log_σ)` | Weibull (log-scale) |
| `weibull_logsigma_lccdf(x, α, log_σ)` | Weibull survival (log-scale) |
| `poisson_lpdf(k, λ)` | Poisson |
| `binomial_lpdf(k, n, p)` | Binomial |
| `bernoulli_logit_lpdf(y, α)` | Bernoulli (logit) |
| `binomial_logit_lpdf(y, n, α)` | Binomial (logit) |
| `neg_binomial_2_lpdf(y, μ, ϕ)` | Negative Binomial |
| `beta_binomial_lpdf(k, n, α, β)` | Beta-Binomial |
| `categorical_logit_lpdf(k, logits)` | Categorical (logit) |
| `dirichlet_lpdf(x, α)` | Dirichlet |
| `multi_normal_diag_lpdf(x, μ, σ)` | MVN (diagonal) |
| `multi_normal_cholesky_lpdf(x, μ, L)` | MVN (Cholesky) |
| `lkj_corr_cholesky_lpdf(L, η)` | LKJ Correlation |
