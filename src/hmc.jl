## LEAPFROG INTEGRATOR

### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(q, p, g, model, ϵ, ∇!)
    
    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]
        q[i] += ϵ * p[i] 
    end

    lp, ok = ∇!(g, model, q)

    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]  
    end

    return lp, ok
end

struct PhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
end
struct HMCState
    curr::PhaseSpacePoint
    proposal::PhaseSpacePoint
    grad::Vector{Float64}
end

## Nesterov dual averaging for step size adaptation
mutable struct DualAveraging
    μ::Float64       # log(10 * ε₀)
    log_ε̄::Float64   # smoothed log step size
    H̄::Float64       # smoothed acceptance statistic
    m::Int           # iteration count
    δ::Float64       # target acceptance rate
    γ::Float64
    t₀::Float64
    κ::Float64
end

function DualAveraging(ε₀::Float64; δ = 0.8)
    DualAveraging(log(10.0 * ε₀), 0.0, 0.0, 0, δ, 0.05, 10.0, 0.75)
end

function adapt!(da::DualAveraging, α)
    da.m += 1
    m = da.m
    w = 1.0 / (m + da.t₀)
    da.H̄ = (1.0 - w) * da.H̄ + w * (da.δ - α)
    log_ε = da.μ - sqrt(m) / da.γ * da.H̄
    mk = m^(-da.κ)
    da.log_ε̄ = mk * log_ε + (1.0 - mk) * da.log_ε̄
    return exp(log_ε)
end

adapted_ε(da::DualAveraging) = exp(da.log_ε̄)

## One HMC transition.  Returns (α, divergent).
function _hmc_step!(HMC, model, ϵ, L, ∇!)
    randn!(HMC.curr.p)
    HMC.proposal.q .= HMC.curr.q
    HMC.proposal.p .= HMC.curr.p

    lp₀, ok = ∇!(HMC.grad, model, HMC.proposal.q)
    if !ok
        @warn "Current position has non-finite log-prob — randomizing"
        randn!(HMC.curr.q)
        return 0.0, true
    end
    H_current = -lp₀ + 0.5 * sum(abs2, HMC.proposal.p)

    lp = lp₀
    for _ in 1:L
        lp, ok = leapfrog!(HMC.proposal.q, HMC.proposal.p, HMC.grad, model, ϵ, ∇!)
        if !ok; return 0.0, true; end
    end

    H_proposal = -lp + 0.5 * sum(abs2, HMC.proposal.p)
    ΔH = H_proposal - H_current
    divergent = !isfinite(ΔH) || ΔH > 1000.0

    α = min(1.0, exp(-ΔH))
    if !isfinite(α); α = 0.0; end
    if rand() < α
        HMC.curr.q .= HMC.proposal.q
        HMC.curr.p .= HMC.proposal.p
    end
    return α, divergent
end

### DIM THRESHOLD FOR ENZYME FORWARD OR REVERSE 

const FORWARD_MODE_THRESHOLD = 20

## note -- ∇logp! handles zeroing of gradient buffer

function _make_grad(model; ad = :auto)
    dim = model.dim
    use_forward = if ad == :forward
        true
    elseif ad == :reverse
        false
    else
        dim ≤ FORWARD_MODE_THRESHOLD
    end

    if use_forward
        seeds = ntuple(i -> (v = zeros(dim); v[i] = 1.0; v), dim)
        ∇! = (g, m, q) -> ∇logp_forward!(g, m, q, seeds)
        ad_label = "Forward"
    else
        ∇! = ∇logp_reverse!
        ad_label = "Reverse"
    end

    printstyled("⚙  Compiling gradient ($ad_label mode, dim=$dim)...\n"; color=:yellow, bold=true)
    q_test = randn(dim)
    lp_test = try log_prob(model, q_test) catch e
        printstyled("✗  log_prob failed at test point\n"; color=:red, bold=true)
        rethrow(e)
    end
    if !isfinite(lp_test)
        printstyled("⚠  log_prob = $lp_test at test point, retrying from zeros\n"; color=:yellow)
        q_test = zeros(dim)
    end
    _, ok = ∇!(zeros(dim), model, q_test)
    if !ok
        printstyled("✗  Gradient compilation failed (check @error messages above)\n"; color=:red, bold=true)
        error("Enzyme autodiff failed — see log above for details")
    end
    printstyled("✓  Gradient ready\n"; color=:green, bold=true)
    return ∇!
end

"""Run a single chain: warmup + sampling. Returns (raw_samples::Matrix, n_divergent)."""
function _run_chain(model, num_samples, ϵ, L, warmup, ∇!; chain_id = 1, quiet = false)
    dim = model.dim

    HMC = HMCState(
        PhaseSpacePoint(zeros(Float64, dim), zeros(Float64, dim)),
        PhaseSpacePoint(zeros(Float64, dim), zeros(Float64, dim)),
        zeros(Float64, dim)
    )

    ## ── Warmup ──
    if !quiet
        printstyled("  Chain $chain_id  "; color=:yellow, bold=true)
        printstyled("warmup $warmup iterations  ϵ₀=$ϵ\n"; color=:yellow)
    end

    da = DualAveraging(Float64(ϵ))
    ϵ_curr = Float64(ϵ)
    n_warmup_div = 0
    for i in 1:warmup
        α, div = _hmc_step!(HMC, model, ϵ_curr, L, ∇!)
        n_warmup_div += div
        ϵ_curr = adapt!(da, α)
    end
    ϵ_adapted = adapted_ε(da)

    if !quiet
        printstyled("  Chain $chain_id  "; color=:green, bold=true)
        printstyled("adapted ϵ = $(round(ϵ_adapted; sigdigits=4))\n"; color=:white, bold=true)
        if n_warmup_div > 0
            printstyled("  Chain $chain_id  ⚠ $n_warmup_div divergent transitions during warmup\n"; color=:yellow)
        end
    end

    ## ── Sampling ──
    raw = Matrix{Float64}(undef, dim, num_samples)
    n_divergent = 0
    progress_interval = max(1, num_samples ÷ 10)
    @inbounds for i in 1:num_samples
        _, div = _hmc_step!(HMC, model, ϵ_adapted, L, ∇!)
        n_divergent += div
        raw[:, i] .= HMC.curr.q
        if !quiet && i % progress_interval == 0
            pct = 100i ÷ num_samples
            printstyled("  Chain $chain_id  $(lpad(pct, 3))%  $i/$num_samples\n"; color=:light_black)
        end
    end
    if !quiet && n_divergent > 0
        printstyled("  Chain $chain_id  ⚠ $n_divergent divergent transitions\n"; color=:yellow)
    end

    return raw, n_divergent
end

function sample(model, num_samples; ϵ = 0.1, L = 10, warmup = 1000, ad = :auto, chains = 4)
    ∇! = _make_grad(model; ad)

    printstyled("~  Sampling "; color=:cyan, bold=true)
    printstyled("$chains chain(s) × $num_samples samples"; color=:white, bold=true)
    printstyled("  L=$L\n"; color=:cyan)

    tasks = [Threads.@spawn _run_chain(model, num_samples, ϵ, L, warmup, ∇!;
                                        chain_id=c, quiet=true)
             for c in 1:chains]

    raw_chains = Vector{Matrix{Float64}}(undef, chains)
    total_div = 0
    for c in 1:chains
        raw, ndiv = fetch(tasks[c])
        raw_chains[c] = raw
        total_div += ndiv
        if ndiv > 0
            printstyled("  Chain $c  ⚠ $ndiv divergent transitions\n"; color=:yellow)
        end
    end

    printstyled("✓  Done"; color=:green, bold=true)
    if total_div > 0
        printstyled("  ⚠ $total_div total divergent transitions\n"; color=:yellow)
    else
        println()
    end

    result = Chains(raw_chains, model.constrain)
    println()
    show(stdout, MIME"text/plain"(), result)
    println()
    return result
end
