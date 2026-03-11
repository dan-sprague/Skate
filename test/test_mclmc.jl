using Test
using Random
using Statistics
using LinearAlgebra
using PhaseSkate
using PhaseSkate: MCLMCState, MCLMCScratch, _normalize!, _generate_unit_vector!,
                  _partially_refresh!, _esh_update!, _mclmc_step!, _mclachlan_step!,
                  _mclmc_warmup!, _adjusted_mclmc_step!, _adjusted_mclmc_warmup!,
                  _make_grad, ModelLogDensity

@testset "MCLMC" begin

    @testset "Unit vector operations" begin
        rng = Xoshiro(42)

        # _normalize! produces unit vector
        v = randn(rng, 10)
        _normalize!(v)
        @test norm(v) ≈ 1.0 atol=1e-14

        # _generate_unit_vector! produces unit vector
        u = zeros(10)
        _generate_unit_vector!(rng, u)
        @test norm(u) ≈ 1.0 atol=1e-14

        # Repeated calls always produce unit vectors
        max_err = 0.0
        for _ in 1:50
            _generate_unit_vector!(rng, u)
            max_err = max(max_err, abs(norm(u) - 1.0))
        end
        @test max_err < 1e-14
    end

    @testset "Partial momentum refresh preserves unit norm" begin
        rng = Xoshiro(123)
        u = zeros(20)
        _generate_unit_vector!(rng, u)

        # Many refreshes at various step_size/L ratios
        max_err = 0.0
        for (half_step, L) in [(0.01, 1.0), (0.1, 0.5), (0.5, 2.0), (1.0, 10.0)]
            for _ in 1:100
                _partially_refresh!(rng, u, half_step, L)
                max_err = max(max_err, abs(norm(u) - 1.0))
            end
        end
        @test max_err < 1e-13

        # L = Inf should leave momentum unchanged
        u_before = copy(u)
        _partially_refresh!(rng, u, 0.1, Inf)
        @test u == u_before

        # Very large step/L ratio should still produce unit vector
        _partially_refresh!(rng, u, 100.0, 0.1)
        @test norm(u) ≈ 1.0 atol=1e-13
    end

    @testset "ESH update preserves unit norm" begin
        rng = Xoshiro(99)
        dim = 15
        u = zeros(dim)
        _generate_unit_vector!(rng, u)
        g = randn(rng, dim)
        g_buf = zeros(dim)
        sqrt_inv_mass = ones(dim)

        # Identity preconditioning
        max_err = 0.0
        all_finite = true
        for _ in 1:200
            g .= randn(rng, dim)
            ke = _esh_update!(u, g, g_buf, 0.1, 0.5, sqrt_inv_mass)
            max_err = max(max_err, abs(norm(u) - 1.0))
            all_finite &= isfinite(ke)
        end
        @test max_err < 1e-13
        @test all_finite

        # Non-trivial preconditioning
        sqrt_inv_mass .= rand(rng, dim) .* 2.0 .+ 0.1
        max_err = 0.0
        all_finite = true
        for _ in 1:200
            g .= randn(rng, dim)
            ke = _esh_update!(u, g, g_buf, 0.05, 0.3, sqrt_inv_mass)
            max_err = max(max_err, abs(norm(u) - 1.0))
            all_finite &= isfinite(ke)
        end
        @test max_err < 1e-13
        @test all_finite
    end

    @testset "McLachlan integrator energy conservation" begin
        dim = 10
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)
        ∇!, _ = _make_grad(model; ad=:auto)

        rng = Xoshiro(7)
        state = MCLMCState(randn(rng, dim), zeros(dim), zeros(dim), 0.0)
        _generate_unit_vector!(rng, state.u)
        scratch = MCLMCScratch(dim)

        lp, ok = ∇!(state.g, model, state.q)
        @test ok
        state.lp = lp

        # Small step size → energy changes should be small
        energy_changes = Float64[]
        n_ok = 0
        for _ in 1:100
            ΔE, ok = _mclmc_step!(rng, state, scratch, model, ∇!, 0.05, 5.0)
            n_ok += ok
            push!(energy_changes, abs(ΔE))
        end
        @test n_ok == 100
        @test median(energy_changes) < 0.5
    end

    @testset "Standard normal sampling" begin
        dim = 5
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)

        ch = redirect_stdout(devnull) do
            sample_mclmc(model, 5000; warmup=2000, chains=2, seed=42)
        end

        # Posterior means ≈ 0
        m = mean(ch, :x)
        @test all(abs.(m) .< 0.15)

        # Posterior stds ≈ 1
        s = samples(ch, :x)
        stds = [std(vec(s[:, j, :])) for j in 1:dim]
        @test all(abs.(stds .- 1.0) .< 0.15)
    end

    @testset "Correlated 2D normal" begin
        mu1 = 3.0; mu2 = -1.0; rho = 0.8
        det_s = 1.0 - rho^2
        inv00 = 1.0 / det_s; inv01 = -rho / det_s; inv11 = 1.0 / det_s

        function ℓ_corr(q)
            d1 = q[1] - mu1
            d2 = q[2] - mu2
            return -0.5 * (inv00 * d1^2 + 2 * inv01 * d1 * d2 + inv11 * d2^2)
        end

        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(2, ℓ_corr, constrain_fn)

        ch = redirect_stdout(devnull) do
            sample_mclmc(model, 5000; warmup=2000, chains=2, seed=77)
        end

        m = mean(ch, :x)
        @test abs(m[1] - mu1) < 0.15
        @test abs(m[2] - mu2) < 0.15

        # Check correlation
        s = samples(ch, :x)
        x1 = vec(s[:, 1, :])
        x2 = vec(s[:, 2, :])
        @test abs(cor(x1, x2) - rho) < 0.1
    end

    @testset "Adaptation produces reasonable parameters" begin
        dim = 10
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)
        ∇!, _ = _make_grad(model; ad=:auto)

        rng = Xoshiro(55)
        state = MCLMCState(randn(rng, dim), zeros(dim), zeros(dim), 0.0)
        _generate_unit_vector!(rng, state.u)
        scratch = MCLMCScratch(dim)

        lp, _ = ∇!(state.g, model, state.q)
        state.lp = lp

        ε, L = _mclmc_warmup!(rng, state, scratch, model, ∇!, 500)

        @test 0.0 < ε < 10.0
        @test 0.0 < L < 50.0
        @test isfinite(ε)
        @test isfinite(L)
    end

end

@testset "Adjusted MCLMC (MAMS)" begin

    @testset "MH step accepts/rejects correctly" begin
        dim = 10
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)
        ∇!, _ = _make_grad(model; ad=:auto)

        rng = Xoshiro(42)
        state = MCLMCState(randn(rng, dim), zeros(dim), zeros(dim), 0.0)
        _generate_unit_vector!(rng, state.u)
        scratch = MCLMCScratch(dim)

        lp, ok = ∇!(state.g, model, state.q)
        @test ok
        state.lp = lp

        # Run many steps, check acceptance rate is reasonable
        n_accept = 0
        n_div = 0
        for _ in 1:200
            α, accepted, div = _adjusted_mclmc_step!(rng, state, scratch, model, ∇!,
                                                      0.3, 3, Inf)
            n_accept += accepted
            n_div += div
            @test 0.0 <= α <= 1.0
        end
        accept_rate = n_accept / 200
        @test accept_rate > 0.1
        @test n_div == 0
    end

    @testset "Adjusted MCLMC adaptation" begin
        dim = 10
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)
        ∇!, _ = _make_grad(model; ad=:auto)

        rng = Xoshiro(55)
        state = MCLMCState(randn(rng, dim), zeros(dim), zeros(dim), 0.0)
        _generate_unit_vector!(rng, state.u)
        scratch = MCLMCScratch(dim)

        lp, _ = ∇!(state.g, model, state.q)
        state.lp = lp

        ε, L = _adjusted_mclmc_warmup!(rng, state, scratch, model, ∇!, 500)

        @test 0.0 < ε < 10.0
        @test 0.0 < L < 50.0
        @test isfinite(ε)
        @test isfinite(L)
    end

    @testset "Standard normal sampling (MAMS)" begin
        dim = 5
        ℓ(q) = begin
            s = 0.0
            for i in eachindex(q)
                s -= 0.5 * q[i]^2
            end
            s
        end
        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(dim, ℓ, constrain_fn)

        ch = redirect_stdout(devnull) do
            sample_adjusted_mclmc(model, 5000; warmup=2000, chains=2, seed=42)
        end

        # Posterior means ≈ 0
        m = mean(ch, :x)
        @test all(abs.(m) .< 0.15)

        # Posterior stds ≈ 1
        s = samples(ch, :x)
        stds = [std(vec(s[:, j, :])) for j in 1:dim]
        @test all(abs.(stds .- 1.0) .< 0.15)
    end

    @testset "Correlated 2D normal (MAMS)" begin
        mu1 = 3.0; mu2 = -1.0; rho = 0.8
        det_s = 1.0 - rho^2
        inv00 = 1.0 / det_s; inv01 = -rho / det_s; inv11 = 1.0 / det_s

        function ℓ_corr_mams(q)
            d1 = q[1] - mu1
            d2 = q[2] - mu2
            return -0.5 * (inv00 * d1^2 + 2 * inv01 * d1 * d2 + inv11 * d2^2)
        end

        constrain_fn(q) = (x = copy(q),)
        model = ModelLogDensity(2, ℓ_corr_mams, constrain_fn)

        ch = redirect_stdout(devnull) do
            sample_adjusted_mclmc(model, 5000; warmup=2000, chains=2, seed=77)
        end

        m = mean(ch, :x)
        @test abs(m[1] - mu1) < 0.15
        @test abs(m[2] - mu2) < 0.15

        # Check correlation
        s = samples(ch, :x)
        x1 = vec(s[:, 1, :])
        x2 = vec(s[:, 2, :])
        @test abs(cor(x1, x2) - rho) < 0.1
    end

end
