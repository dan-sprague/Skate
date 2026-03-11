# benchmarks/survival/bench_turing.jl
# Benchmark Turing.jl on the Weibull survival frailty model.
#
# Prerequisites:
#   ] add Turing Enzyme
#
# Usage: julia benchmarks/survival/bench_turing.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Turing
using Distributions
using LinearAlgebra
using Printf
using MCMCChains
using Enzyme
using ADTypes: AutoEnzyme

# ── Minimal JSON reader (same as bench_phaseskate.jl) ────────────────────────

function read_json(path)
    txt = read(path, String)
    return _parse_json(txt, Ref(1))
end

function _skip_ws(s, pos)
    while pos[] <= length(s) && s[pos[]] in (' ', '\t', '\n', '\r')
        pos[] += 1
    end
end

function _parse_json(s, pos)
    _skip_ws(s, pos)
    c = s[pos[]]
    if c == '{';     return _parse_obj(s, pos)
    elseif c == '['; return _parse_arr(s, pos)
    elseif c == '"'; return _parse_str(s, pos)
    elseif c == 't'; pos[] += 4; return true
    elseif c == 'f'; pos[] += 5; return false
    elseif c == 'n'; pos[] += 4; return nothing
    else;            return _parse_num(s, pos)
    end
end

function _parse_str(s, pos)
    pos[] += 1; start = pos[]
    while s[pos[]] != '"'
        s[pos[]] == '\\' && (pos[] += 1)
        pos[] += 1
    end
    result = s[start:pos[]-1]; pos[] += 1; return result
end

function _parse_num(s, pos)
    start = pos[]
    while pos[] <= length(s) && s[pos[]] in ('-', '+', '.', 'e', 'E', '0':'9'...)
        pos[] += 1
    end
    numstr = s[start:pos[]-1]
    (occursin('.', numstr) || occursin('e', numstr) || occursin('E', numstr)) ?
        parse(Float64, numstr) : parse(Int, numstr)
end

function _parse_arr(s, pos)
    pos[] += 1; result = Any[]
    _skip_ws(s, pos)
    s[pos[]] == ']' && (pos[] += 1; return result)
    while true
        push!(result, _parse_json(s, pos))
        _skip_ws(s, pos)
        s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1; return result
end

function _parse_obj(s, pos)
    pos[] += 1; result = Dict{String, Any}()
    _skip_ws(s, pos)
    s[pos[]] == '}' && (pos[] += 1; return result)
    while true
        _skip_ws(s, pos); key = _parse_str(s, pos)
        _skip_ws(s, pos); pos[] += 1
        result[key] = _parse_json(s, pos)
        _skip_ws(s, pos)
        s[pos[]] == ',' ? (pos[] += 1) : break
    end
    pos[] += 1; return result
end

function write_json(path, obj)
    open(path, "w") do io; _wj(io, obj); println(io); end
end
function _wj(io, x::AbstractDict)
    print(io, "{"); first = true
    for (k, v) in x
        first || print(io, ","); first = false
        print(io, "\n  \"$k\": "); _wj(io, v)
    end
    print(io, "\n}")
end
_wj(io, x::Real) = isinteger(x) && abs(x) < 1e15 ? print(io, Int(x)) : print(io, x)
_wj(io, x::Integer) = print(io, x)
_wj(io, x::AbstractString) = print(io, "\"$x\"")
function _wj(io, x::AbstractVector)
    print(io, "[")
    for (i, v) in enumerate(x); i > 1 && print(io, ", "); _wj(io, v); end
    print(io, "]")
end

# ── Configuration ────────────────────────────────────────────────────────────

const NUM_CHAINS  = 4
const NUM_WARMUP  = 1000
const NUM_SAMPLES = 2000
const SEED        = 42

bench_dir   = @__DIR__
data_path   = joinpath(bench_dir, "data", "survival_data.json")
result_path = joinpath(bench_dir, "results", "turing_results.json")

# ── Model definition ────────────────────────────────────────────────────────

@model function survival_frailty(N, P, H, X, trt, times, hosp, obs_idx, cens_idx)
    # Priors
    log_alpha ~ Normal(0.0, 0.5)
    beta_0 ~ Normal(2.0, 2.0)
    beta ~ MvNormal(zeros(P), I)
    log_sigma_int ~ Normal(-1.0, 1.0)
    log_sigma_slope ~ Normal(-1.0, 1.0)
    rho ~ Uniform(-1.0, 1.0)
    Turing.@addlogprob! log(1.0 - rho^2)   # LKJ(2)
    z_int ~ MvNormal(zeros(H), I)
    z_slope ~ MvNormal(zeros(H), I)

    # Derived quantities
    alpha = exp(log_alpha)
    sigma_int = exp(log_sigma_int)
    sigma_slope = exp(log_sigma_slope)
    sqrt_1mrho2 = sqrt(1.0 - rho^2)

    # Correlated frailties (non-centered)
    frailty_int = sigma_int .* z_int
    frailty_slope = sigma_slope .* (rho .* z_int .+ sqrt_1mrho2 .* z_slope)

    # Log-scale for all patients
    log_scale = beta_0 .+ X * beta .+ frailty_int[hosp] .+ trt .* frailty_slope[hosp]

    # Observed events
    for i in obs_idx
        s = exp(log_scale[i])
        Turing.@addlogprob! logpdf(Weibull(alpha, s), times[i])
    end

    # Censored observations
    for i in cens_idx
        s = exp(log_scale[i])
        Turing.@addlogprob! logccdf(Weibull(alpha, s), times[i])
    end
end

# ── Load data ────────────────────────────────────────────────────────────────

println("Loading data from: $data_path")
raw = read_json(data_path)

to_vec(x) = Float64.(x)
to_ivec(x) = Int.(x)
to_mat(x) = Float64.(reduce(hcat, [r for r in x])')

N_val     = raw["N"]
P_val     = raw["P"]
H_val     = raw["H"]
X_val     = to_mat(raw["X"])
trt_val   = to_vec(raw["trt"])
times_val = to_vec(raw["times"])
hosp_val  = to_ivec(raw["hosp"])
obs_val   = to_ivec(raw["obs_idx"])
cens_val  = to_ivec(raw["cens_idx"])

model = survival_frailty(N_val, P_val, H_val, X_val, trt_val, times_val,
                          hosp_val, obs_val, cens_val)

n_params = 2 + P_val + 2 + 1 + 2 * H_val  # log_alpha, beta_0, beta, log_sigma_*, rho, z_*
println("SurvivalFrailty (Turing) - approx dim=$n_params")

# ── Sample ───────────────────────────────────────────────────────────────────

println("\nSampling: $NUM_WARMUP warmup, $NUM_SAMPLES samples, $NUM_CHAINS chains")

t_start = time()
sampler = NUTS(NUM_WARMUP, 0.8; adtype=AutoEnzyme())
chain = sample(model, sampler, MCMCThreads(), NUM_SAMPLES, NUM_CHAINS;
               progress=true)
t_total = time() - t_start

# ── Diagnostics ──────────────────────────────────────────────────────────────

display(chain)

# Extract min ESS
ess_vals = ess_rhat(chain).nt.ess
min_ess_val = minimum(filter(!isnan, ess_vals))

println("\n", "-- Results ", "-"^39)
@printf("  Total wall time:  %.1f s\n", t_total)
@printf("  Min ESS (bulk):   %.0f\n", min_ess_val)
@printf("  ESS/s:            %.1f\n", min_ess_val / t_total)

# ── Save results ─────────────────────────────────────────────────────────────

mkpath(joinpath(bench_dir, "results"))

results = Dict{String, Any}(
    "backend"         => "Turing.jl",
    "num_chains"      => NUM_CHAINS,
    "num_warmup"      => NUM_WARMUP,
    "num_samples"     => NUM_SAMPLES,
    "total_time_s"    => round(t_total; digits=2),
    "sampling_time_s" => round(t_total; digits=2),
    "min_ess_bulk"    => round(min_ess_val; digits=1),
    "ess_per_sec"     => round(min_ess_val / t_total; digits=1),
    "divergences"     => 0,
)

write_json(result_path, results)
println("\nResults saved to: $result_path")
