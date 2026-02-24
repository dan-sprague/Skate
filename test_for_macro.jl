include("src/bijections.jl")
include("src/utilities.jl")
include("src/logdensitygenerator.jl")
include("src/lpdfs.jl")
include("src/hmc.jl")

import Enzyme
import Base.@kwdef
using Random, Test
include("src/lang.jl")

println("═══ @for macro tests ═══\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Scalar .+ vector
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_ScalarPlusVec begin
    @constants begin
        N::Int
    end
    @params begin
        mu::Float64
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = mu .+ x
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_ScalarPlusVec_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        mu::Float64
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(mu + x[i], 0.0, 1.0)
        end
    end
end

let N = 10
    d = ForTest_ScalarPlusVec_DataSet(N=N)
    m = make_fortest_scalarplusvec(d)
    d2 = ForTest_ScalarPlusVec_Manual_DataSet(N=N)
    m2 = make_fortest_scalarplusvec_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ scalar .+ vector")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Vector .* scalar
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_VecTimesScalar begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        @for y = x .* sigma
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_VecTimesScalar_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(x[i] * sigma, 0.0, 1.0)
        end
    end
end

let N = 8
    d = ForTest_VecTimesScalar_DataSet(N=N)
    m = make_fortest_vectimesscalar(d)
    d2 = ForTest_VecTimesScalar_Manual_DataSet(N=N)
    m2 = make_fortest_vectimesscalar_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ vector .* scalar")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Chained broadcasts: a .+ b .* c .- d
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_Chained begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = a .+ b .* c .- d
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_Chained_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(a[i] + b[i] * c[i] - d[i], 0.0, 1.0)
        end
    end
end

let N = 5
    d = ForTest_Chained_DataSet(N=N)
    m = make_fortest_chained(d)
    d2 = ForTest_Chained_Manual_DataSet(N=N)
    m2 = make_fortest_chained_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ chained broadcasts a .+ b .* c .- d")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Matrix-vector product (full)
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_MatVec begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, D)
    end
    @logjoint begin
        @for y = X * beta
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_MatVec_Manual begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, D)
    end
    @logjoint begin
        for i in 1:N
            dot_i = 0.0
            for j in 1:D
                dot_i += X[i, j] * beta[j]
            end
            target += normal_lpdf(dot_i, 0.0, 1.0)
        end
    end
end

let N = 12, D = 4
    X = randn(N, D)
    d = ForTest_MatVec_DataSet(N=N, D=D, X=X)
    m = make_fortest_matvec(d)
    d2 = ForTest_MatVec_Manual_DataSet(N=N, D=D, X=X)
    m2 = make_fortest_matvec_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ matrix-vector product (full)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Matrix-vector product (column slice): X[:, 1:k] * beta
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_MatVecSlice begin
    @constants begin
        N::Int
        D::Int
        p::Int
        X::Matrix{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, p)
    end
    @logjoint begin
        @for y = X[:, 1:p] * beta
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_MatVecSlice_Manual begin
    @constants begin
        N::Int
        D::Int
        p::Int
        X::Matrix{Float64}
    end
    @params begin
        beta = param(Vector{Float64}, p)
    end
    @logjoint begin
        for i in 1:N
            dot_i = 0.0
            for j in 1:p
                dot_i += X[i, j] * beta[j]
            end
            target += normal_lpdf(dot_i, 0.0, 1.0)
        end
    end
end

let N = 10, D = 6, p = 3
    X = randn(N, D)
    d = ForTest_MatVecSlice_DataSet(N=N, D=D, p=p, X=X)
    m = make_fortest_matvecslice(d)
    d2 = ForTest_MatVecSlice_Manual_DataSet(N=N, D=D, p=p, X=X)
    m2 = make_fortest_matvecslice_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ matrix-vector product (column slice)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Fancy indexing: v[ids]
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_FancyIndex begin
    @constants begin
        N::Int
        K::Int
        ids::Vector{Int}
    end
    @params begin
        v = param(Vector{Float64}, K)
    end
    @logjoint begin
        @for y = v[ids]
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_FancyIndex_Manual begin
    @constants begin
        N::Int
        K::Int
        ids::Vector{Int}
    end
    @params begin
        v = param(Vector{Float64}, K)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(v[ids[i]], 0.0, 1.0)
        end
    end
end

let N = 15, K = 3
    ids = rand(1:K, N)
    d = ForTest_FancyIndex_DataSet(N=N, K=K, ids=ids)
    m = make_fortest_fancyindex(d)
    d2 = ForTest_FancyIndex_Manual_DataSet(N=N, K=K, ids=ids)
    m2 = make_fortest_fancyindex_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ fancy indexing v[ids]")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Broadcast function call: exp.(x)
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_BroadcastFunc begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = exp.(x)
        for i in 1:N
            target += normal_lpdf(y[i], 1.0, 1.0)
        end
    end
end

@spec ForTest_BroadcastFunc_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(exp(x[i]), 1.0, 1.0)
        end
    end
end

let N = 7
    d = ForTest_BroadcastFunc_DataSet(N=N)
    m = make_fortest_broadcastfunc(d)
    d2 = ForTest_BroadcastFunc_Manual_DataSet(N=N)
    m2 = make_fortest_broadcastfunc_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ broadcast function call exp.(x)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Broadcast division: a ./ b
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_BroadcastDiv begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N; lower = 0.1)
    end
    @logjoint begin
        @for y = a ./ b
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_BroadcastDiv_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N; lower = 0.1)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(a[i] / b[i], 0.0, 1.0)
        end
    end
end

let N = 6
    d = ForTest_BroadcastDiv_DataSet(N=N)
    m = make_fortest_broadcastdiv(d)
    d2 = ForTest_BroadcastDiv_Manual_DataSet(N=N)
    m2 = make_fortest_broadcastdiv_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ broadcast division a ./ b")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Literal constants in broadcasts: 0.5 .* x .+ 1.0
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_Literals begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = 0.5 .* x .+ 1.0
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_Literals_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        x = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(0.5 * x[i] + 1.0, 0.0, 1.0)
        end
    end
end

let N = 5
    d = ForTest_Literals_DataSet(N=N)
    m = make_fortest_literals(d)
    d2 = ForTest_Literals_Manual_DataSet(N=N)
    m2 = make_fortest_literals_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ literal constants 0.5 .* x .+ 1.0")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Compound: scalar + matvec + fancy_index * scalar + vector * vector
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_Compound begin
    @constants begin
        N::Int
        D::Int
        K::Int
        X::Matrix{Float64}
        ids::Vector{Int}
    end
    @params begin
        mu::Float64
        beta = param(Vector{Float64}, D)
        v = param(Vector{Float64}, K)
        z = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = mu .+ (X * beta) .+ (v[ids] .* 0.1) .+ z
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_Compound_Manual begin
    @constants begin
        N::Int
        D::Int
        K::Int
        X::Matrix{Float64}
        ids::Vector{Int}
    end
    @params begin
        mu::Float64
        beta = param(Vector{Float64}, D)
        v = param(Vector{Float64}, K)
        z = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            dot_i = 0.0
            for j in 1:D
                dot_i += X[i, j] * beta[j]
            end
            target += normal_lpdf(mu + dot_i + v[ids[i]] * 0.1 + z[i], 0.0, 1.0)
        end
    end
end

let N = 20, D = 4, K = 3
    X = randn(N, D)
    ids = rand(1:K, N)
    d = ForTest_Compound_DataSet(N=N, D=D, K=K, X=X, ids=ids)
    m = make_fortest_compound(d)
    d2 = ForTest_Compound_Manual_DataSet(N=N, D=D, K=K, X=X, ids=ids)
    m2 = make_fortest_compound_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ compound: mu .+ (X * beta) .+ (v[ids] .* 0.1) .+ z")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Fused block: two assignments, same dimension
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_FusedBlock begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for begin
            y1 = a .+ b
            y2 = y1 .* c .- d
        end
        for i in 1:N
            target += normal_lpdf(y2[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_FusedBlock_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        y1 = Vector{Float64}(undef, N)
        y2 = Vector{Float64}(undef, N)
        for i in 1:N
            y1[i] = a[i] + b[i]
            y2[i] = y1[i] * c[i] - d[i]
        end
        for i in 1:N
            target += normal_lpdf(y2[i], 0.0, 1.0)
        end
    end
end

let N = 8
    d = ForTest_FusedBlock_DataSet(N=N)
    m = make_fortest_fusedblock(d)
    d2 = ForTest_FusedBlock_Manual_DataSet(N=N)
    m2 = make_fortest_fusedblock_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ fused block: two assignments")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Sum-reduction: target += sum(normal_lpdf.(x, mu, sigma))
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_SumReduction begin
    @constants begin
        N::Int
        obs::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        @for target += sum(normal_lpdf.(obs, mu, sigma))
    end
end

@spec ForTest_SumReduction_Manual begin
    @constants begin
        N::Int
        obs::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(obs[i], mu[i], sigma)
        end
    end
end

let N = 10
    obs = randn(N)
    d = ForTest_SumReduction_DataSet(N=N, obs=obs)
    m = make_fortest_sumreduction(d)
    d2 = ForTest_SumReduction_Manual_DataSet(N=N, obs=obs)
    m2 = make_fortest_sumreduction_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ sum-reduction: target += sum(normal_lpdf.(...))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 13. Nested parentheses: (a .+ (b .* (c .+ d)))
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_NestedParens begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        @for y = (a .+ (b .* (c .+ d)))
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_NestedParens_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        c = param(Vector{Float64}, N)
        d = param(Vector{Float64}, N)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(a[i] + b[i] * (c[i] + d[i]), 0.0, 1.0)
        end
    end
end

let N = 6
    d = ForTest_NestedParens_DataSet(N=N)
    m = make_fortest_nestedparens(d)
    d2 = ForTest_NestedParens_Manual_DataSet(N=N)
    m2 = make_fortest_nestedparens_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ nested parentheses (a .+ (b .* (c .+ d)))")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 14. Data vectors in broadcasts: mu .+ data_x
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_DataVec begin
    @constants begin
        N::Int
        x::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        @for y = mu .+ x
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, sigma)
        end
    end
end

@spec ForTest_DataVec_Manual begin
    @constants begin
        N::Int
        x::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(mu + x[i], 0.0, sigma)
        end
    end
end

let N = 10
    x = randn(N)
    d = ForTest_DataVec_DataSet(N=N, x=x)
    m = make_fortest_datavec(d)
    d2 = ForTest_DataVec_Manual_DataSet(N=N, x=x)
    m2 = make_fortest_datavec_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ data vector in broadcast: mu .+ x (data)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 15. Compound with column-sliced matvec + fancy index + broadcast
#     (mirrors the JointALM pattern)
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_ALMPattern begin
    @constants begin
        N::Int
        D::Int
        p::Int
        K::Int
        X::Matrix{Float64}
        country_ids::Vector{Int}
    end
    @params begin
        mu_k::Float64
        beta_k = param(Vector{Float64}, p)
        omega_k::Float64
        z_k = param(Vector{Float64}, N)
        mu_country = param(Vector{Float64}, K)
    end
    @logjoint begin
        @for begin
            country_eff = mu_country[country_ids]
            log_k = mu_k .+ (X[:, 1:p] * beta_k) .+ (country_eff .* 0.1) .+ (omega_k .* z_k)
        end
        for i in 1:N
            target += normal_lpdf(log_k[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_ALMPattern_Manual begin
    @constants begin
        N::Int
        D::Int
        p::Int
        K::Int
        X::Matrix{Float64}
        country_ids::Vector{Int}
    end
    @params begin
        mu_k::Float64
        beta_k = param(Vector{Float64}, p)
        omega_k::Float64
        z_k = param(Vector{Float64}, N)
        mu_country = param(Vector{Float64}, K)
    end
    @logjoint begin
        for i in 1:N
            dot_i = 0.0
            for j in 1:p
                dot_i += X[i, j] * beta_k[j]
            end
            target += normal_lpdf(mu_k + dot_i + mu_country[country_ids[i]] * 0.1 + omega_k * z_k[i], 0.0, 1.0)
        end
    end
end

let N = 25, D = 8, p = 4, K = 5
    X = randn(N, D)
    country_ids = rand(1:K, N)
    d = ForTest_ALMPattern_DataSet(N=N, D=D, p=p, K=K, X=X, country_ids=country_ids)
    m = make_fortest_almpattern(d)
    d2 = ForTest_ALMPattern_Manual_DataSet(N=N, D=D, p=p, K=K, X=X, country_ids=country_ids)
    m2 = make_fortest_almpattern_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ ALM pattern: fused block with matvec_slice + fancy_index + broadcast")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 16. Fused block with 3 assignments and dependency chain
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_FusedDependency begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
        K::Int
        ids::Vector{Int}
    end
    @params begin
        mu = param(Vector{Float64}, K)
        beta = param(Vector{Float64}, D)
        gamma::Float64
        scale = param(Float64; lower = 0.0)
    end
    @logjoint begin
        @for begin
            ce = mu[ids]
            lk = ce .+ (X * beta)
            ls = scale .- (lk .* gamma)
        end
        for i in 1:N
            target += normal_lpdf(ls[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_FusedDependency_Manual begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
        K::Int
        ids::Vector{Int}
    end
    @params begin
        mu = param(Vector{Float64}, K)
        beta = param(Vector{Float64}, D)
        gamma::Float64
        scale = param(Float64; lower = 0.0)
    end
    @logjoint begin
        for i in 1:N
            ce_i = mu[ids[i]]
            dot_i = 0.0
            for j in 1:D
                dot_i += X[i, j] * beta[j]
            end
            lk_i = ce_i + dot_i
            ls_i = scale - lk_i * gamma
            target += normal_lpdf(ls_i, 0.0, 1.0)
        end
    end
end

let N = 15, D = 3, K = 4
    X = randn(N, D)
    ids = rand(1:K, N)
    d = ForTest_FusedDependency_DataSet(N=N, D=D, X=X, K=K, ids=ids)
    m = make_fortest_fuseddependency(d)
    d2 = ForTest_FusedDependency_Manual_DataSet(N=N, D=D, X=X, K=K, ids=ids)
    m2 = make_fortest_fuseddependency_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ fused block: 3 assignments with dependency chain")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 17. Multiple @for in same model (sequential, not fused)
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_MultipleFor begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        mu::Float64
    end
    @logjoint begin
        @for y1 = a .+ mu
        @for y2 = b .- mu
        for i in 1:N
            target += normal_lpdf(y1[i], 0.0, 1.0)
            target += normal_lpdf(y2[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_MultipleFor_Manual begin
    @constants begin
        N::Int
    end
    @params begin
        a = param(Vector{Float64}, N)
        b = param(Vector{Float64}, N)
        mu::Float64
    end
    @logjoint begin
        for i in 1:N
            target += normal_lpdf(a[i] + mu, 0.0, 1.0)
            target += normal_lpdf(b[i] - mu, 0.0, 1.0)
        end
    end
end

let N = 7
    d = ForTest_MultipleFor_DataSet(N=N)
    m = make_fortest_multiplefor(d)
    d2 = ForTest_MultipleFor_Manual_DataSet(N=N)
    m2 = make_fortest_multiplefor_manual(d2)
    q = randn(m.dim)
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ multiple sequential @for in same model")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 18. Enzyme gradient compatibility
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_Enzyme begin
    @constants begin
        N::Int
        D::Int
        X::Matrix{Float64}
        obs::Vector{Float64}
    end
    @params begin
        mu::Float64
        beta = param(Vector{Float64}, D)
        sigma = param(Float64; lower = 0.0)
    end
    @logjoint begin
        @for yhat = mu .+ (X * beta)
        for i in 1:N
            target += normal_lpdf(obs[i], yhat[i], sigma)
        end
    end
end

let N = 20, D = 3
    X = randn(N, D)
    beta_true = randn(D)
    obs = X * beta_true .+ 0.5 .* randn(N)
    d = ForTest_Enzyme_DataSet(N=N, D=D, X=X, obs=obs)
    m = make_fortest_enzyme(d)

    q = randn(m.dim)
    lp = m.ℓ(q)
    @test isfinite(lp)

    # Test gradient computation
    g = zeros(m.dim)
    ∇logp_reverse!(g, m, q)
    @test all(isfinite, g)
    @test any(g .!= 0.0)  # non-trivial gradient

    # Finite-difference check
    ε = 1e-5
    for j in 1:min(m.dim, 5)  # check first 5 dims
        q_plus = copy(q); q_plus[j] += ε
        q_minus = copy(q); q_minus[j] -= ε
        fd = (m.ℓ(q_plus) - m.ℓ(q_minus)) / (2ε)
        @test abs(g[j] - fd) < 1e-4 * max(1.0, abs(fd))
    end
    println("  ✓ Enzyme reverse-mode gradient matches finite differences")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 19. Regression: existing models still work (no @for used)
# ═══════════════════════════════════════════════════════════════════════════════

# Re-test MixtureModel (already defined in test_macro.jl pattern)
@spec ForTest_NoFor begin
    @constants begin
        N::Int
        K::Int
        x::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, K; ordered = true)
        sigma = param(Float64; lower = 0.0)
        theta = param(Vector{Float64}, K; simplex = true)
    end
    @logjoint begin
        target += normal_lpdf(sigma, 0.0, 5.0)
        target += dirichlet_lpdf(theta, 1.0)
        for i in 1:N
            target += log_mix(theta, j -> normal_lpdf(x[i], mu[j], sigma))
        end
    end
end

let N = 50, K = 2
    x = vcat(randn(25) .- 2.0, randn(25) .+ 2.0)
    d = ForTest_NoFor_DataSet(N=N, K=K, x=x)
    m = make_fortest_nofor(d)
    q = randn(m.dim)
    lp = m.ℓ(q)
    @test isfinite(lp)
    println("  ✓ regression: model without @for still works")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 20. Param matrix * param vector
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_ParamMatVec begin
    @constants begin
        N::Int
        D::Int
    end
    @params begin
        M = param(Matrix{Float64}, N, D)
        v = param(Vector{Float64}, D)
    end
    @logjoint begin
        @for y = M * v
        for i in 1:N
            target += normal_lpdf(y[i], 0.0, 1.0)
        end
    end
end

@spec ForTest_ParamMatVec_Manual begin
    @constants begin
        N::Int
        D::Int
    end
    @params begin
        M = param(Matrix{Float64}, N, D)
        v = param(Vector{Float64}, D)
    end
    @logjoint begin
        for i in 1:N
            dot_i = 0.0
            for j in 1:D
                dot_i += M[i, j] * v[j]
            end
            target += normal_lpdf(dot_i, 0.0, 1.0)
        end
    end
end

let N = 5, D = 3
    d = ForTest_ParamMatVec_DataSet(N=N, D=D)
    m = make_fortest_parammatvec(d)
    d2 = ForTest_ParamMatVec_Manual_DataSet(N=N, D=D)
    m2 = make_fortest_parammatvec_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ param matrix * param vector")
end

# ═══════════════════════════════════════════════════════════════════════════════
# 21. Mixed @for and non-@for statements
# ═══════════════════════════════════════════════════════════════════════════════
@spec ForTest_Mixed begin
    @constants begin
        N::Int
        obs::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
        alpha::Float64
    end
    @logjoint begin
        target += normal_lpdf(alpha, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        @for y = alpha .+ mu
        for i in 1:N
            target += normal_lpdf(obs[i], y[i], sigma)
        end
    end
end

@spec ForTest_Mixed_Manual begin
    @constants begin
        N::Int
        obs::Vector{Float64}
    end
    @params begin
        mu = param(Vector{Float64}, N)
        sigma = param(Float64; lower = 0.0)
        alpha::Float64
    end
    @logjoint begin
        target += normal_lpdf(alpha, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        for i in 1:N
            target += normal_lpdf(obs[i], alpha + mu[i], sigma)
        end
    end
end

let N = 8
    obs = randn(N)
    d = ForTest_Mixed_DataSet(N=N, obs=obs)
    m = make_fortest_mixed(d)
    d2 = ForTest_Mixed_Manual_DataSet(N=N, obs=obs)
    m2 = make_fortest_mixed_manual(d2)
    q = randn(m.dim)
    @test m.dim == m2.dim
    @test m.ℓ(q) ≈ m2.ℓ(q) atol=1e-10
    println("  ✓ mixed @for and non-@for statements")
end

println("\n═══ All @for macro tests passed! ═══")
