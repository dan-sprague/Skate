using Pkg; Pkg.activate(@__DIR__)
using Skate
using Random
using Test
using Distributions
using LinearAlgebra: I

# ══════════════════════════════════════════════════════════════════════════════
# 1. Normal-Normal conjugate model (sanity check — should always pass)
# ══════════════════════════════════════════════════════════════════════════════

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
        for i in 1:N
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end

function simulate_normal()
    mu_true    = rand(Normal(0.0, 10.0))
    sigma_true = rand(Exponential(1.0))
    N = 50
    y = rand(Normal(mu_true, sigma_true), N)
    data  = NormalModel_DataSet(N=N, y=y)
    model = make_normalmodel(data)
    return (mu=mu_true, sigma=sigma_true), model
end

println("─── SBC: Normal-Normal ───")
result_normal = sbc(simulate_normal;
    N=50, M=100, num_samples=500, warmup=200, ϵ=0.1, L=10, ad=:forward)
display(result_normal)
@test calibrated(result_normal)


# Misspecified: same NormalModel, but data generated from Student-t
function simulate_studentt_misspecified()
    mu_true    = rand(Normal(0.0, 10.0))
    sigma_true = rand(Exponential(1.0))
    N = 50
    # generate data from Student-t(df=3) instead of Normal
    y = mu_true .+ sigma_true .* rand(TDist(3), N)
    data  = NormalModel_DataSet(N=N, y=y)
    model = make_normalmodel(data)
    return (mu=mu_true, sigma=sigma_true), model
end

println("\n─── SBC: Normal model, Student-t(3) data (misspecified) ───")
result_misspec = sbc(simulate_studentt_misspecified;
    N=20, M=500, num_samples=1000, warmup=200, ϵ=0.1, L=10, ad=:forward)
display(result_misspec)
println("Misspecified calibrated: ", calibrated(result_misspec))

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
println("\n─── Summary ───")
println("Normal-Normal (well-specified):     ", calibrated(result_normal)   ? "PASSED" : "FAILED")
println("Normal model, t(3) data (misspec):  ", calibrated(result_misspec)  ? "PASSED" : "FAILED")
