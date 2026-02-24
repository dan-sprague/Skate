include("src/bijections.jl")
include("src/utilities.jl")
include("src/logdensitygenerator.jl")
include("src/lpdfs.jl")
include("src/hmc.jl")

import Enzyme
import Base.@kwdef
using Random
include("src/lang.jl")

@spec MixtureModel begin
    @constants begin
        N::Int
        K::Int
        D::Int
        x::Matrix{Float64}
    end

    @params begin
        theta = param(Vector{Float64}, K; simplex = true)
        mu = param(Matrix{Float64}, K, D; ordered = 1)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin
        target += dirichlet_lpdf(theta, 1.0)
        target += normal_lpdf(sigma, 0.0, 5.0)
        target += multi_normal_diag_lpdf(mu, 0.0, 10.0)
        for i in 1:N
            target += log_mix(theta, j -> 
            multi_normal_diag_lpdf(x[i, :], mu[j, :], sigma)
            )
        end
    end
end


# ── Generate 2D mixture data ─────────────────────────────────────────────────
K = 2
D = 2
N = 500
true_mu = [-2.0 1.0; 2.0 -1.0]  # K×D
true_sigma = 0.5

components = [rand() < 0.5 ? 1 : 2 for _ in 1:N]
x_data = Matrix{Float64}(undef, N, D)
for i in 1:N
    for dd in 1:D
        x_data[i, dd] = randn() * true_sigma + true_mu[components[i], dd]
    end
end

d = MixtureModel_DataSet(N=N, K=K, D=D, x=x_data)
m = make_mixturemodel(d);

using Test
@time samples = sample(m, 1000; ϵ = 0.1, L = 10, warmup = 100)


# ── Turing comparison ────────────────────────────────────────────────────────
using Turing
using Bijectors: ordered
using FillArrays
using LinearAlgebra
using Distributions: MixtureModel, MvNormal

x_turing = Matrix(x_data')  # D×N matrix

@model function gmm_marginalized(x, ::Val{K}) where {K}
    D, N = size(x)
    mu1 ~ ordered(MvNormal(Zeros(K), 10.0^2 * I))
    mu2 ~ MvNormal(Zeros(K), 10.0^2 * I)
    w ~ Dirichlet(K, 1.0)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    dists = [MvNormal([mu1[k], mu2[k]], σ^2 * I) for k in 1:K]
    mix = MixtureModel(dists, w)
    for n in 1:N
        x[:, n] ~ mix
    end
end

model = gmm_marginalized(x_turing, Val(K))
@time chain = Turing.sample(model, HMC(0.1, 10), 1000)