# benchmarks/survival/generate_data.jl
# Generate synthetic Weibull survival data with hierarchical correlated frailties.
# Writes benchmarks/survival/data/survival_data.json and true_params.json.
#
# Usage: julia benchmarks/survival/generate_data.jl

using Random
using Statistics: median
using Distributions: Weibull, Exponential

# ── Minimal JSON writer (no external deps) ──────────────────────────────────

function write_json(path, obj)
    open(path, "w") do io
        _write_json(io, obj)
        println(io)
    end
end

function _write_json(io, x::AbstractDict)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        print(io, "\n  \"$k\": ")
        _write_json(io, v)
    end
    print(io, "\n}")
end

function _write_json(io, x::Real)
    isinteger(x) && abs(x) < 1e15 ? print(io, Int(x)) : print(io, x)
end
_write_json(io, x::Integer) = print(io, x)
_write_json(io, x::AbstractString) = print(io, "\"$x\"")

function _write_json(io, x::AbstractVector)
    print(io, "[")
    for (i, v) in enumerate(x)
        i > 1 && print(io, ", ")
        _write_json(io, v)
    end
    print(io, "]")
end

function _write_json(io, x::AbstractMatrix)
    nr, nc = size(x)
    print(io, "[")
    for i in 1:nr
        i > 1 && print(io, ",\n    ")
        print(io, "[")
        for j in 1:nc
            j > 1 && print(io, ", ")
            _write_json(io, x[i, j])
        end
        print(io, "]")
    end
    print(io, "]")
end

# ── Ground truth parameters ─────────────────────────────────────────────────

const TRUE = (
    log_alpha      = 0.4,          # alpha ~ 1.49 (increasing hazard)
    beta_0         = 2.5,          # baseline log-scale
    beta           = [0.3, -0.4, 0.2, -0.15, 0.25, -0.1, 0.35, -0.2],
    log_sigma_int  = -0.5,         # sigma_int ~ 0.61
    log_sigma_slope = -0.8,        # sigma_slope ~ 0.45
    rho            = 0.4,          # frailty correlation
)

const H = 100    # hospitals
const N = 5000   # patients
const P = 8      # covariates

# ── Simulate data ───────────────────────────────────────────────────────────

Random.seed!(42)

alpha = exp(TRUE.log_alpha)
sigma_int   = exp(TRUE.log_sigma_int)
sigma_slope = exp(TRUE.log_sigma_slope)
rho = TRUE.rho
sqrt_1mrho2 = sqrt(1.0 - rho^2)

# Hospital frailties (non-centered)
z_int   = randn(H)
z_slope = randn(H)

# Derived frailties (centered)
frailty_int   = sigma_int .* z_int
frailty_slope = sigma_slope .* (rho .* z_int .+ sqrt_1mrho2 .* z_slope)

# Patient covariates
X = randn(N, P)
trt = Float64.(rand(0:1, N))            # binary treatment
hosp = rand(1:H, N)                      # hospital assignment

# Compute log-scale for each patient
log_scale = Vector{Float64}(undef, N)
for i in 1:N
    h = hosp[i]
    xb = sum(X[i, j] * TRUE.beta[j] for j in 1:P)
    log_scale[i] = TRUE.beta_0 + xb + frailty_int[h] + trt[i] * frailty_slope[h]
end

# Generate Weibull survival times
times = [rand(Weibull(alpha, exp(log_scale[i]))) for i in 1:N]

# Apply random right-censoring (~30%)
cens_times = rand(Exponential(median(times) * 1.5), N)
observed = times .<= cens_times
times .= min.(times, cens_times)
obs_idx  = findall(observed)
cens_idx = findall(.!observed)

println("Survival frailty data generated.")
println("  Hospitals:  $H")
println("  Patients:   $N")
println("  Covariates: $P")
println("  Observed:   $(length(obs_idx)) ($(round(100*length(obs_idx)/N, digits=1))%)")
println("  Censored:   $(length(cens_idx)) ($(round(100*length(cens_idx)/N, digits=1))%)")

# ── Write to JSON ───────────────────────────────────────────────────────────

data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

data_dict = Dict{String, Any}(
    "N"        => N,
    "P"        => P,
    "H"        => H,
    "X"        => X,
    "trt"      => trt,
    "times"    => times,
    "hosp"     => hosp,
    "N_obs"    => length(obs_idx),
    "N_cens"   => length(cens_idx),
    "obs_idx"  => obs_idx,
    "cens_idx" => cens_idx,
)

write_json(joinpath(data_dir, "survival_data.json"), data_dict)
println("\nWrote: benchmarks/survival/data/survival_data.json")

# True parameters (for plotting and verification)
true_dict = Dict{String, Any}(
    "log_alpha"       => TRUE.log_alpha,
    "beta_0"          => TRUE.beta_0,
    "beta"            => collect(TRUE.beta),
    "log_sigma_int"   => TRUE.log_sigma_int,
    "log_sigma_slope" => TRUE.log_sigma_slope,
    "rho"             => TRUE.rho,
    "z_int"           => z_int,
    "z_slope"         => z_slope,
    "frailty_int"     => frailty_int,
    "frailty_slope"   => frailty_slope,
)

write_json(joinpath(data_dir, "true_params.json"), true_dict)
println("Wrote: benchmarks/survival/data/true_params.json")
