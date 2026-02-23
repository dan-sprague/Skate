using Test 
using Distributions
using Skater
using LinearAlgebra
using Statistics

using Skater: WelfordState, welford_update!, welford_variance


@testset "Log PDF Functions" begin
    @testset "Density Evaluations" begin 
        @test normal_lpdf(0.0, 0.0, 1.0) ≈ logpdf(Normal(0.0, 1.0), 0.0)
        @test cauchy_lpdf(0.0, 0.0, 1.0) ≈ logpdf(Cauchy(0.0, 1.0), 0.0)
        @test exponential_lpdf(1.0, 2.0) ≈ logpdf(Exponential(2.0), 1.0)
        @test gamma_lpdf(1.0, 2.0, 3.0) ≈ logpdf(Gamma(2.0,1.0 / 3.0), 1.0)
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
            
            # 2. Hand-calculated expected unnormalized log-pdf:
            # i=1: (2 - 1 + 2(2-1)) * log(1.0) = 3 * 0.0 = 0.0
            # i=2: (2 - 2 + 2(2-1)) * log(sqrt(0.75)) = 2 * log(sqrt(0.75)) = log(0.75)
            expected_logpdf = log(0.75)
            
            @test lkj_corr_cholesky_lpdf(L_valid, η) ≈ expected_logpdf
        end

        @testset "Illegal Parameter Rejection" begin
            L_valid = LowerTriangular([1.0 0.0; 0.5 sqrt(0.75)])

            @test lkj_corr_cholesky_lpdf(L_valid, 0.0) == -Inf 
            @test lkj_corr_cholesky_lpdf(L_valid, -1.0) == -Inf

            # 2. Test illegal L diagonal (zeros)
            L_zero_diag = LowerTriangular([0.0 0.0; 
                                        0.5 1.0])
            @test lkj_corr_cholesky_lpdf(L_zero_diag, 2.0) == -Inf

            # 3. Test illegal L diagonal (negatives)
            L_neg_diag = LowerTriangular([1.0  0.0; 
                                        0.5 -0.1])
            @test lkj_corr_cholesky_lpdf(L_neg_diag, 2.0) == -Inf
        end
    end
end;

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
end;