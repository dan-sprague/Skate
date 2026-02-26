using Pkg; Pkg.activate(@__DIR__)
using Skate
using Random

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
@time ch = sample(m, 1000;
ϵ = 0.1, L = 10, warmup = 100, chains = 4,
ad = :forward)


# ── Full-covariance mixture model (Stan-style Cholesky parameterization) ─────
@spec FullCovMixture begin
    @constants begin
        N::Int
        K::Int
        D::Int
        x::Matrix{Float64}
    end

    @params begin
        theta = param(Vector{Float64}, K; simplex = true)
        mu = param(Matrix{Float64}, K, D; ordered = 1)
        L_Omega = param(CholCorr, K, D)
        log_sigma = param(Matrix{Float64}, K, D)
    end

    @logjoint begin
        target += dirichlet_lpdf(theta, 1.0)
        for k in 1:K
            target += lkj_corr_cholesky_lpdf(L_Omega[:,:,k], 2.0)
            for d in 1:D
                target += normal_lpdf(log_sigma[k,d], 0.0, 1.0)
                target += normal_lpdf(mu[k,d], 0.0, 10.0)
            end
        end
        for n in 1:N
            target += log_mix(theta, k ->
                multi_normal_cholesky_scaled_lpdf(x[n,:], mu[k,:], log_sigma[k,:], L_Omega[:,:,k]) # @view is auto applied :) 
            )
        end
    end
end

# Generate 2D mixture data with full covariance
K_fc = 2; D_fc = 2; N_fc = 200
true_mu_fc = [-2.0 1.0; 2.0 -1.0]
# Cholesky factors: L[:,:,k] for component k
true_L = zeros(D_fc, D_fc, K_fc)
true_L[:,:,1] = [1.0 0.0; 0.3 0.95]
true_L[:,:,2] = [0.5 0.0; -0.2 0.48]

x_fc = Matrix{Float64}(undef, N_fc, D_fc)
for i in 1:N_fc
    k = rand() < 0.5 ? 1 : 2
    z = randn(D_fc)
    for d in 1:D_fc
        x_fc[i, d] = true_mu_fc[k, d]
        for dd in 1:D_fc
            x_fc[i, d] += true_L[d, dd, k] * z[dd]
        end
    end
end

d_fc = FullCovMixture_DataSet(N=N_fc, K=K_fc, D=D_fc, x=x_fc)
m_fc = make_fullcovmixture(d_fc)

# Verify model dimension: (K-1) + K*D + K*D*(D-1)/2 + K*D
expected_dim = (K_fc - 1) + K_fc * D_fc + K_fc * div(D_fc * (D_fc - 1), 2) + K_fc * D_fc
@test m_fc.dim == expected_dim

# Verify log-prob evaluates without error
q_test = randn(m_fc.dim)
lp = m_fc.ℓ(q_test, m_fc.data)
@test isfinite(lp)

# Verify constrain returns valid structure
c = m_fc.constrain(q_test, m_fc.data)
@test haskey(c, :L_Omega)
@test size(c.L_Omega) == (D_fc, D_fc, K_fc)
# Each slice should be lower-triangular with positive diagonal
for k in 1:K_fc
    Lk = c.L_Omega[:,:,k]
    for i in 1:D_fc
        @test Lk[i,i] > 0
        for j in (i+1):D_fc
            @test Lk[i,j] ≈ 0.0 atol=1e-15
        end
    end
    # LL^T should have ones on diagonal (correlation matrix)
    R = Lk * Lk'
    for i in 1:D_fc
        @test R[i,i] ≈ 1.0 atol=1e-10
    end
end

println("Full-covariance mixture model: dim=$(m_fc.dim), log_prob=$(round(lp, digits=2))")
@time ch_fc = sample(m_fc, 2000; ϵ = 0.05, L = 10, warmup = 500)

# ── Verify FullCovMixture posterior recovers truth ────────────────────────────
using LinearAlgebra

# Posterior mean of mu (K×D)
mu_mean = mean(ch_fc, :mu)
println("Posterior mean mu:\n", round.(mu_mean; digits=3))
println("True mu:\n", true_mu_fc)
for k in 1:K_fc, d in 1:D_fc
    @test abs(mu_mean[k, d] - true_mu_fc[k, d]) < 0.5
end

# Posterior mean of theta
theta_mean = mean(ch_fc, :theta)
println("Posterior mean theta: ", round.(theta_mean; digits=3))
@test abs(theta_mean[1] - 0.5) < 0.15
@test abs(theta_mean[2] - 0.5) < 0.15

# Reconstruct L_Sigma = diag(exp(log_sigma[k,:])) * L_Omega[:,:,k] and compare to true_L
# (this needs raw samples since it's a nonlinear transform)
log_sigma_samp = samples(ch_fc, :log_sigma)   # (nsamples, K, D, nchains)
L_Omega_samp   = samples(ch_fc, :L_Omega)     # (nsamples, D, D, K, nchains)
ns, nc = size(ch_fc.data, 1), size(ch_fc.data, 3)
L_Sigma_mean = zeros(D_fc, D_fc, K_fc)
for c in 1:nc, i in 1:ns
    for k in 1:K_fc
        sigma_k = exp.(log_sigma_samp[i, k, :, c])
        L_Sigma_mean[:, :, k] .+= sigma_k .* L_Omega_samp[i, :, :, k, c]
    end
end
L_Sigma_mean ./= (ns * nc)

println("Posterior mean L_Sigma[:,:,1]:\n", round.(L_Sigma_mean[:,:,1]; digits=3))
println("True L[:,:,1]:\n", true_L[:,:,1])
println("Posterior mean L_Sigma[:,:,2]:\n", round.(L_Sigma_mean[:,:,2]; digits=3))
println("True L[:,:,2]:\n", true_L[:,:,2])

for k in 1:K_fc
    for i in 1:D_fc, j in 1:D_fc
        @test abs(L_Sigma_mean[i, j, k] - true_L[i, j, k]) < 0.3
    end
end

println("✓ FullCovMixture posterior matches truth")

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