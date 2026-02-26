using Pkg; Pkg.activate(@__DIR__)
using Reactant   # must load before Skate.sample to activate extension
using Skate
using Random
using Test

# ── Simple Normal-Normal conjugate model ────────────────────────────────────
@spec NormalModel begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        target += normal_lpdf(y, mu, sigma)
    end
end

# Generate data with known truth
Random.seed!(42)
true_mu = 3.0
true_sigma = 1.5
N = 100
y_data = true_mu .+ true_sigma .* randn(N)

data = NormalModel_DataSet(N=N, y=y_data)
model = make_normalmodel(data)

println("Model dim: ", model.dim)
println("True mu=$true_mu, sigma=$true_sigma")

# ── Test XLA backend ────────────────────────────────────────────────────────
println("\n═══ XLA Backend Test ═══")
@time ch_xla = Skate.sample(model, 1000;
    backend=XLABackend(), warmup=500, ϵ=0.1, L=10, chains=1, ad=:forward)

mu_mean = Skate.mean(ch_xla, :mu)
sigma_mean = Skate.mean(ch_xla, :sigma)
println("XLA posterior: mu=$(round(mu_mean; digits=3)), sigma=$(round(sigma_mean; digits=3))")

# Tolerances account for single-chain MCMC noise
@test abs(mu_mean - true_mu) < 1.0
@test abs(sigma_mean - true_sigma) < 1.0
println("✓ XLA backend test passed")
