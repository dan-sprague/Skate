using Test
using PhaseSkate
using LinearAlgebra
using Statistics
using Distributions
using Enzyme
using FiniteDifferences

@testset "PhaseSkate" begin

# ═══════════════════════════════════════════════════════════════════════════════
# Log PDF Functions
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Log PDF Functions" begin
    @testset "Density Evaluations" begin
        @test normal_lpdf(0.0, 0.0, 1.0) ≈ logpdf(Normal(0.0, 1.0), 0.0)
        @test cauchy_lpdf(0.0, 0.0, 1.0) ≈ logpdf(Cauchy(0.0, 1.0), 0.0)
        @test exponential_lpdf(1.0, 2.0) ≈ logpdf(Exponential(2.0), 1.0)
        @test gamma_lpdf(1.0, 2.0, 3.0) ≈ logpdf(Gamma(2.0, 1.0 / 3.0), 1.0)
        @test poisson_lpdf(3, 2.5) ≈ logpdf(Poisson(2.5), 3)
        @test binomial_lpdf(5, 10, 0.5) ≈ logpdf(Binomial(10, 0.5), 5)
        @test beta_binomial_lpdf(5, 10, 2.0, 3.0) ≈ logpdf(BetaBinomial(10, 2.0, 3.0), 5)
        @test weibull_lpdf(1.5, 2.0, 3.0) ≈ logpdf(Weibull(2.0, 3.0), 1.5)
    end

    @testset "Beta Distribution" begin
        @test beta_lpdf(-0.5, 2.0, 2.0) == -Inf
        @test multi_normal_cholesky_lpdf([0.0], [0.0], [0.0]) == -Inf
    end

    @testset "MVN Cholesky Log-PDF" begin
        Σ = [2.0 0.5;
             0.5 1.0]
        L = cholesky(Σ).L
        μ = [1.0, -0.5]
        x = [0.5, 0.2]

        @testset "Mathematical Accuracy" begin
            dist = MvNormal(μ, Σ)
            expected_logpdf = logpdf(dist, x)
            @test multi_normal_cholesky_lpdf(x, μ, L) ≈ expected_logpdf
        end

        @testset "Illegal Parameter Rejection" begin
            L_zero = LowerTriangular([0.0 0.0;
                                      0.5 1.0])
            @test multi_normal_cholesky_lpdf(x, μ, L_zero) == -Inf

            L_negative = LowerTriangular([-1.0  0.0;
                                           0.5  1.0])
            @test multi_normal_cholesky_lpdf(x, μ, L_negative) == -Inf
        end
    end

    @testset "LKJ Correlation Cholesky Log-PDF" begin
        @testset "Mathematical Accuracy (2x2 Case)" begin
            ρ = 0.5
            L_valid = LowerTriangular([1.0 0.0;
                                       ρ   sqrt(1.0 - ρ^2)])
            η = 2.0
            expected_logpdf = log(0.75)
            @test lkj_corr_cholesky_lpdf(L_valid, η) ≈ expected_logpdf
        end

        @testset "Illegal Parameter Rejection" begin
            L_valid = LowerTriangular([1.0 0.0; 0.5 sqrt(0.75)])
            @test lkj_corr_cholesky_lpdf(L_valid, 0.0) == -Inf
            @test lkj_corr_cholesky_lpdf(L_valid, -1.0) == -Inf

            L_zero_diag = LowerTriangular([0.0 0.0;
                                           0.5 1.0])
            @test lkj_corr_cholesky_lpdf(L_zero_diag, 2.0) == -Inf

            L_neg_diag = LowerTriangular([1.0  0.0;
                                          0.5 -0.1])
            @test lkj_corr_cholesky_lpdf(L_neg_diag, 2.0) == -Inf
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Welford Online Variance
# ═══════════════════════════════════════════════════════════════════════════════

using PhaseSkate: WelfordState, welford_update!, welford_variance

@testset "Welford Online Variance" begin
    @testset "Ground Truth Equivalence" begin
        D = 10
        N = 1000
        data = [randn(D) for _ in 1:N]

        W = WelfordState(D)
        for x in data
            welford_update!(W, x)
        end

        data_matrix = reduce(hcat, data)'
        expected_mean = vec(mean(data_matrix, dims=1))
        expected_var = vec(var(data_matrix, dims=1))

        @test W.mean ≈ expected_mean
        @test welford_variance(W) ≈ expected_var
    end

    @testset "Zero Variance (Identical Samples)" begin
        D = 5
        x = randn(D)
        W = WelfordState(D)
        for _ in 1:100
            welford_update!(W, x)
        end
        @test W.mean ≈ x
        @test all(welford_variance(W) .≈ 0.0)
        @test all(welford_variance(W) .>= 0.0)
    end

    @testset "Numerical Stability (Catastrophic Cancellation)" begin
        D = 3
        offset = 1e9
        data = [randn(D) .+ offset for _ in 1:1000]

        W = WelfordState(D)
        for x in data
            welford_update!(W, x)
        end

        data_matrix = reduce(hcat, data)'
        expected_var = vec(var(data_matrix, dims=1))
        @test welford_variance(W) ≈ expected_var rtol=1e-5
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Enzyme Gradient Tests (via @skate model)
# ═══════════════════════════════════════════════════════════════════════════════

using PhaseSkate: ∇logp_reverse!

@skate SimpleNormal begin
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

@testset "Enzyme Gradient via @skate Model" begin
    y_data = randn(10)
    d = SimpleNormalData(N=10, y=y_data)
    m = make(d)

    q = zeros(Float64, m.dim)
    g = zeros(Float64, m.dim)

    lp, ok = ∇logp_reverse!(g, m, q)
    @test ok
    @test isfinite(lp)

    # Compare against finite differences
    fdm = central_fdm(5, 1)
    fd_grad = FiniteDifferences.grad(fdm, q -> log_prob(m, q), q)[1]
    @test g ≈ fd_grad rtol=1e-5
end

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint Transforms
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Constraint Transforms" begin
    @testset "IdentityConstraint" begin
        c = IdentityConstraint()
        @test transform(c, 1.5) == 1.5
        @test log_abs_det_jacobian(c, 1.5) == 0.0
    end

    @testset "LowerBounded" begin
        c = LowerBounded(0.0)
        @test transform(c, 0.0) ≈ 1.0
        @test transform(c, -10.0) > 0.0
    end

    @testset "Bounded" begin
        c = Bounded(0.0, 1.0)
        @test 0.0 < transform(c, 0.0) < 1.0
        @test 0.0 < transform(c, -5.0) < 1.0
        @test 0.0 < transform(c, 5.0) < 1.0
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Integration Test: @skate → make → sample
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Integration: Define → Make → Sample" begin
    y_data = randn(20)
    d = SimpleNormalData(N=20, y=y_data)
    m = make(d)

    @test m.dim == 2
    @test isfinite(log_prob(m, zeros(m.dim)))

    ch = sample(m, 100; warmup=50, chains=2, seed=42)
    @test ch isa Chains
    @test size(ch.data, 1) == 100   # nsamples
    @test size(ch.data, 3) == 2     # nchains

    mu_post = mean(ch, :mu)
    @test isfinite(mu_post)

    ess_val = min_ess(ch)
    @test ess_val > 0
end

end # @testset "PhaseSkate"

# ── PosteriorDB validation tests (opt-in) ────────────────────────────────────
if get(ENV, "POSTERIORDB_TESTS", "false") == "true"
    include("posteriordb/runtests.jl")
end
