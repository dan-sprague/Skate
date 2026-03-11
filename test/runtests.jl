using Test
using PhaseSkate
using LinearAlgebra
using Statistics
using Distributions
using Enzyme
using FiniteDifferences

# Resolve name conflicts between Distributions.sample and PhaseSkate.sample
const sample = PhaseSkate.sample

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

# ═══════════════════════════════════════════════════════════════════════════════
# Regression Tests — Untested lpdf Functions
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Scalar lpdf Regression" begin
    @testset "lognormal_lpdf" begin
        for (x, μ, σ) in [(1.0, 0.0, 1.0), (2.5, 1.0, 0.5), (0.1, -1.0, 2.0)]
            @test lognormal_lpdf(x, μ, σ) ≈ logpdf(LogNormal(μ, σ), x)
        end
    end

    @testset "student_t_lpdf" begin
        for (x, ν, μ, σ) in [(0.0, 3.0, 0.0, 1.0), (1.5, 5.0, 1.0, 2.0), (-2.0, 1.0, 0.0, 0.5)]
            @test student_t_lpdf(x, ν, μ, σ) ≈ logpdf(LocationScale(μ, σ, TDist(ν)), x)
        end
    end

    @testset "uniform_lpdf" begin
        for (x, lo, hi) in [(0.5, 0.0, 1.0), (3.0, 2.0, 5.0), (-1.0, -2.0, 0.0)]
            @test uniform_lpdf(x, lo, hi) ≈ logpdf(Uniform(lo, hi), x)
        end
    end

    @testset "laplace_lpdf" begin
        for (x, μ, b) in [(0.0, 0.0, 1.0), (2.0, 1.0, 0.5), (-3.0, 0.0, 2.0)]
            @test laplace_lpdf(x, μ, b) ≈ logpdf(Laplace(μ, b), x)
        end
    end

    @testset "logistic_lpdf" begin
        for (x, μ, s) in [(0.0, 0.0, 1.0), (2.0, 1.0, 0.5), (-3.0, 0.0, 2.0)]
            @test logistic_lpdf(x, μ, s) ≈ logpdf(Logistic(μ, s), x)
        end
    end

    @testset "neg_binomial_2_lpdf" begin
        for (y, μ, ϕ) in [(0, 5.0, 2.0), (3, 10.0, 1.0), (10, 3.0, 5.0)]
            p = ϕ / (ϕ + μ)
            r = ϕ
            @test neg_binomial_2_lpdf(y, μ, ϕ) ≈ logpdf(NegativeBinomial(r, p), y)
        end
    end

    @testset "bernoulli_logit_lpdf" begin
        for (y, α) in [(1, 0.0), (0, 2.0), (1, -1.5)]
            p = 1.0 / (1.0 + exp(-α))
            @test bernoulli_logit_lpdf(y, α) ≈ logpdf(Bernoulli(p), y)
        end
    end

    @testset "binomial_logit_lpdf" begin
        for (y, n, α) in [(3, 10, 0.0), (0, 5, -1.0), (7, 10, 1.5)]
            p = 1.0 / (1.0 + exp(-α))
            @test PhaseSkate.binomial_logit_lpdf(y, n, α) ≈ logpdf(Binomial(n, p), y)
        end
    end

    @testset "categorical_logit_lpdf" begin
        α = [1.0, 2.0, 0.5]
        softmax_α = exp.(α) ./ sum(exp.(α))
        for y in 1:3
            @test categorical_logit_lpdf(y, α) ≈ logpdf(Categorical(softmax_α), y)
        end
    end

    @testset "weibull_logsigma_lpdf" begin
        for (x, α, log_σ) in [(1.0, 2.0, 0.0), (0.5, 1.5, -0.5), (2.0, 0.8, 1.0)]
            @test weibull_logsigma_lpdf(x, α, log_σ) ≈ weibull_lpdf(x, α, exp(log_σ))
        end
    end
end

@testset "Multivariate lpdf Regression" begin
    @testset "multi_normal_diag_lpdf — all overloads" begin
        x = [1.0, -0.5, 0.3]

        # Scalar μ, scalar σ
        @test multi_normal_diag_lpdf(x, 0.0, 1.0) ≈ logpdf(MvNormal(zeros(3), I), x)

        # Vector μ, scalar σ
        μ_vec = [0.5, -0.5, 0.0]
        @test multi_normal_diag_lpdf(x, μ_vec, 2.0) ≈ logpdf(MvNormal(μ_vec, 4.0 * I), x)

        # Vector μ, vector σ
        σ_vec = [1.0, 2.0, 0.5]
        Σ_diag = Diagonal(σ_vec .^ 2)
        @test multi_normal_diag_lpdf(x, μ_vec, σ_vec) ≈ logpdf(MvNormal(μ_vec, Σ_diag), x)
    end

    @testset "multi_normal_cholesky_scaled_lpdf" begin
        x = [1.0, -0.5]
        μ = [0.0, 0.0]
        log_sigma = [log(2.0), log(3.0)]
        L_corr = LowerTriangular([1.0 0.0; 0.3 sqrt(1 - 0.09)])
        L_full = diag_pre_multiply(exp.(log_sigma), L_corr)
        @test multi_normal_cholesky_scaled_lpdf(x, μ, log_sigma, L_corr) ≈ multi_normal_cholesky_lpdf(x, μ, L_full)
    end

    @testset "dirichlet_lpdf" begin
        x = [0.2, 0.3, 0.5]
        α_vec = [2.0, 3.0, 1.0]
        @test dirichlet_lpdf(x, α_vec) ≈ logpdf(Dirichlet(α_vec), x)

        # Scalar α overload
        @test dirichlet_lpdf(x, 2.0) ≈ logpdf(Dirichlet(3, 2.0), x)
    end

    @testset "correlated_topic_lpdf" begin
        # K=2 topics, V=3 words — hand-computed
        eta = [0.0, 0.0]  # equal topic weights → θ = [0.5, 0.5]
        log_phi = [log(0.5) log(0.3);
                   log(0.3) log(0.5);
                   log(0.2) log(0.2)]
        lse_phi = [log(0.5 + 0.3 + 0.2), log(0.3 + 0.5 + 0.2)]
        x_row = [10.0, 5.0, 3.0]

        result = correlated_topic_lpdf(x_row, eta, log_phi, lse_phi)
        @test isfinite(result)

        # Manual: with equal θ, p(w=v) = 0.5*(phi_norm[v,1] + phi_norm[v,2])
        # phi_norm[:,k] = phi[:,k] / sum(phi[:,k]) — but phi already sums to 1 per topic
        p_word = 0.5 .* [0.5, 0.3, 0.2] .+ 0.5 .* [0.3, 0.5, 0.2]
        expected = sum(x_row .* log.(p_word))
        @test result ≈ expected
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Regression Tests — Untested Bijections
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Bijection Regression" begin
    @testset "UpperBounded" begin
        c = UpperBounded(5.0)
        y = transform(c, 0.0)
        @test y < 5.0
        @test isfinite(log_abs_det_jacobian(c, 0.0))
    end

    @testset "SimplexConstraint" begin
        y = randn(2)  # K-1 = 2, K = 3
        x, log_jac = simplex_transform(y)
        @test length(x) == 3
        @test sum(x) ≈ 1.0
        @test all(0 .< x .< 1)
        @test isfinite(log_jac)

        # simplex_transform! matches simplex_transform
        x2 = similar(x)
        log_jac2 = simplex_transform!(x2, y)
        @test x2 ≈ x
        @test log_jac2 ≈ log_jac

        # Extreme values still sum to 1
        x_extreme, _ = simplex_transform([100.0, -100.0])
        @test sum(x_extreme) ≈ 1.0 atol=1e-10
    end

    @testset "OrderedConstraint" begin
        x = randn(4)
        y, log_jac = ordered_transform(x)
        @test issorted(y)
        @test isfinite(log_jac)

        # ordered_transform! matches ordered_transform
        y2 = similar(y)
        log_jac2 = PhaseSkate.ordered_transform!(y2, x)
        @test y2 ≈ y
        @test log_jac2 ≈ log_jac

        # Also test via struct interface
        oc = OrderedConstraint()
        y3 = transform(oc, x)
        @test y3 ≈ y
    end

    @testset "corr_cholesky_transform" begin
        # D=2: 1 free param
        z2 = [0.5]
        L2, jac2 = corr_cholesky_transform(z2, 2)
        R2 = L2 * L2'
        @test R2[1,1] ≈ 1.0
        @test R2[2,2] ≈ 1.0
        @test isfinite(jac2)
        @test all(eigvals(Symmetric(R2)) .> 0)

        # D=3: 3 free params
        z3 = [0.0, 0.3, -0.5]
        L3, jac3 = corr_cholesky_transform(z3, 3)
        R3 = L3 * L3'
        for i in 1:3
            @test R3[i,i] ≈ 1.0
        end
        @test isfinite(jac3)
        @test all(eigvals(Symmetric(R3)) .> 0)
    end

    @testset "ordered_simplex_matrix!" begin
        K, V = 2, 3
        q = randn(K * (V - 1))
        mat = zeros(K, V)
        log_jac = ordered_simplex_matrix!(mat, q, K, V)

        # Each row sums to 1
        for k in 1:K
            @test sum(mat[k, :]) ≈ 1.0
        end
        # First column strictly increasing
        @test mat[1, 1] < mat[2, 1]
        @test isfinite(log_jac)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Regression Tests — Utilities and Chain Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Utility Regression" begin
    @testset "log_sum_exp" begin
        @test log_sum_exp([0.0]) == 0.0
        @test log_sum_exp([0.0, 0.0]) ≈ log(2)
        # Numerical stability with extreme values
        @test log_sum_exp([-1000.0, -1000.0]) ≈ -1000.0 + log(2)
    end

    @testset "log_mix — callback overload" begin
        # Equal weights, same component → simplifies to single component
        weights = [0.5, 0.5]
        result = log_mix(weights) do j
            normal_lpdf(0.0, 0.0, 1.0)
        end
        @test result ≈ normal_lpdf(0.0, 0.0, 1.0)
    end

    @testset "log_mix — two-vector overload" begin
        log_w = [log(0.3), log(0.7)]
        log_lik = [-1.0, -2.0]
        expected = log(0.3 * exp(-1.0) + 0.7 * exp(-2.0))
        @test log_mix(log_w, log_lik) ≈ expected
    end

    @testset "log_mix — three-arg overload" begin
        a = [log(0.3), log(0.7)]
        b = [-1.0, -2.0]
        offset = 0.5
        expected = log_mix(a .+ offset, b)
        @test log_mix(a, b, offset) ≈ expected
    end
end

@testset "Chain Diagnostics Regression" begin
    # Build a minimal Chains object for testing ci and thin
    y_data = randn(20)
    d = SimpleNormalData(N=20, y=y_data)
    m = make(d)
    ch = sample(m, 200; warmup=100, chains=2, seed=123)

    @testset "ci" begin
        lo95, hi95 = ci(ch, :mu; level=0.95)
        mu_mean = mean(ch, :mu)
        @test lo95 < mu_mean < hi95

        # Narrower at level=0.5
        lo50, hi50 = ci(ch, :mu; level=0.5)
        @test (hi50 - lo50) < (hi95 - lo95)
    end

    @testset "thin" begin
        ch_thin = thin(ch, 50)
        @test size(ch_thin.data, 1) == 50

        # M > nsamples → clamped to nsamples
        ch_big = thin(ch, 10000)
        @test size(ch_big.data, 1) == size(ch.data, 1)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

using PhaseSkate: WelfordState, welford_update!

@testset "Argument Error Guards" begin
    @test_throws ArgumentError uniform_lpdf(0.5, 5.0, 2.0)   # lo > hi
    @test_throws ArgumentError uniform_lpdf(0.5, 3.0, 3.0)   # lo == hi
    @test_throws ArgumentError log_sum_exp(Float64[])

    @testset "welford_update! dimension mismatch" begin
        ws = WelfordState(3)
        @test_throws ArgumentError welford_update!(ws, [1.0, 2.0])  # wrong length
    end

    @testset "ci level out of range" begin
        y_data = randn(20)
        d = SimpleNormalData(N=20, y=y_data)
        m = make(d)
        ch = sample(m, 50; warmup=25, chains=1, seed=99)
        @test_throws ArgumentError ci(ch, :mu; level=0.0)
        @test_throws ArgumentError ci(ch, :mu; level=1.0)
        @test_throws ArgumentError ci(ch, :mu; level=-0.5)
    end

    @testset "thin M <= 0" begin
        y_data = randn(20)
        d = SimpleNormalData(N=20, y=y_data)
        m = make(d)
        ch = sample(m, 50; warmup=25, chains=1, seed=99)
        @test_throws ArgumentError thin(ch, 0)
        @test_throws ArgumentError thin(ch, -1)
    end
end

@testset "Boundary Value Tests" begin
    @testset "beta_lpdf at support boundaries" begin
        # At exact boundaries (0 and 1) — should be finite for α,β > 1
        @test isfinite(beta_lpdf(0.0, 2.0, 2.0))
        @test isfinite(beta_lpdf(1.0, 2.0, 2.0))
        # Outside support
        @test beta_lpdf(-0.1, 2.0, 2.0) == -Inf
        @test beta_lpdf(1.1, 2.0, 2.0) == -Inf
    end

    @testset "simplex_transform extreme inputs" begin
        x, _ = simplex_transform([100.0, -100.0])
        @test sum(x) ≈ 1.0 atol=1e-10
        @test all(xi -> 0 ≤ xi ≤ 1, x)
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Dense Metric Tests
# ═══════════════════════════════════════════════════════════════════════════════

using PhaseSkate: WelfordCovState, welford_update!, welford_covariance,
                  DenseMetric, kinetic_energy, sample_momentum!, check_uturn
using Random: Xoshiro

@testset "WelfordCovState" begin
    D = 5
    N = 1000
    data = [randn(D) for _ in 1:N]

    W = WelfordCovState(D)
    for x in data
        welford_update!(W, x)
    end

    data_matrix = reduce(hcat, data)'
    expected_mean = vec(mean(data_matrix, dims=1))
    expected_cov = cov(data_matrix)

    @test W.mean ≈ expected_mean
    @test welford_covariance(W) ≈ expected_cov rtol=1e-10
end

@testset "DenseMetric" begin
    @testset "Kinetic energy" begin
        Σ = [2.0 0.5; 0.5 1.0]
        dm = DenseMetric(Σ)
        p = [1.0, 2.0]
        expected = 0.5 * dot(p, Σ * p)
        @test kinetic_energy(p, dm) ≈ expected
    end

    @testset "Momentum sampling" begin
        Σ = [2.0 0.5; 0.5 1.0]
        dm = DenseMetric(Σ)
        rng = Xoshiro(42)
        N = 50000
        samples_mat = zeros(2, N)
        p = zeros(2)
        for i in 1:N
            sample_momentum!(rng, p, dm)
            samples_mat[:, i] .= p
        end
        # inv_metric = Σ means M = Σ⁻¹, so p ~ N(0, M) = N(0, Σ⁻¹)
        empirical_cov = cov(samples_mat')
        expected_cov = inv(Σ)
        @test empirical_cov ≈ expected_cov rtol=0.1
    end

    @testset "U-turn check" begin
        Σ = [1.0 0.0; 0.0 1.0]
        dm = DenseMetric(Σ)
        ρ = [1.0, 1.0]
        p_start = [1.0, 0.5]
        p_end = [0.5, 1.0]
        @test check_uturn(ρ, p_start, p_end, dm) == true

        p_bad = [-2.0, -2.0]
        @test check_uturn(ρ, p_start, p_bad, dm) == false
    end
end

@testset "Integration: Dense Metric Sampling" begin
    y_data = randn(20)
    d = SimpleNormalData(N=20, y=y_data)
    m = make(d)

    ch = sample(m, 200; warmup=100, chains=2, seed=42, metric=:dense)
    @test ch isa Chains
    @test size(ch.data, 1) == 200

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
