## ── Dense Mass Matrix ──

struct DenseMetric
    inv_metric::Matrix{Float64}   # M⁻¹ for leapfrog / KE / u-turn
    L::LowerTriangular{Float64, Matrix{Float64}}  # cholesky(M).L for momentum sampling
    tmp::Vector{Float64}          # scratch buffer (avoids allocs in hot loop)
end

"""Construct a `DenseMetric` from a covariance matrix Σ (the inverse metric).
`L = cholesky(Σ⁻¹).L` so that `L * z` ~ N(0, Σ⁻¹) = N(0, M)."""
function DenseMetric(Σ::AbstractMatrix{Float64})
    C = cholesky(Symmetric(Σ))
    inv_metric = Matrix(Σ)
    M = inv(C)  # Σ⁻¹ via Cholesky inverse (numerically stable)
    L = LowerTriangular(Matrix(cholesky(Symmetric(M)).L))
    tmp = zeros(size(Σ, 1))
    DenseMetric(inv_metric, L, tmp)
end

function DenseMetric(dim::Int)
    DenseMetric(Matrix{Float64}(I, dim, dim))
end

"""Update `inv_metric` in-place from Welford covariance state, with Stan-style shrinkage.
Returns the (possibly new) metric — replaces the DenseMetric struct since it's immutable."""
function _update_dense_metric!(old_metric::DenseMetric, welford::WelfordCovState, dim::Int)
    n = welford.n
    if n < 2
        return old_metric
    end
    Σ = welford_covariance(welford)
    # Stan-style shrinkage: (n/(n+5))*Σ + (5/(n+5))*1e-3*I
    shrink_data = n / (n + 5.0)
    shrink_prior = 5.0 / (n + 5.0)
    @inbounds for j in 1:dim, i in 1:dim
        Σ[i, j] = shrink_data * Σ[i, j]
    end
    @inbounds for i in 1:dim
        Σ[i, i] += shrink_prior * 1e-3
    end
    # Try Cholesky; add jitter on failure; fall back to diagonal
    try
        return DenseMetric(Σ)
    catch
        @inbounds for i in 1:dim
            Σ[i, i] += 1e-6
        end
        try
            return DenseMetric(Σ)
        catch
            # Fall back to diagonal
            diag_vec = ones(dim)
            @inbounds for i in 1:dim
                diag_vec[i] = max(Σ[i, i], 1e-3)
            end
            return DenseMetric(diagm(diag_vec))
        end
    end
end

## ── Dense dispatch methods ──

function leapfrog!(q, p, g, model, ϵ, metric::DenseMetric, ∇!)
    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]
    end

    mul!(metric.tmp, metric.inv_metric, p)
    @simd for i in eachindex(q)
        q[i] += ϵ * metric.tmp[i]
    end

    lp, ok = ∇!(g, model, q)

    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]
    end

    return lp, ok
end

function sample_momentum!(rng, p, metric::DenseMetric)
    randn!(rng, metric.tmp)
    mul!(p, metric.L, metric.tmp)
end

function kinetic_energy(p, metric::DenseMetric)
    mul!(metric.tmp, metric.inv_metric, p)
    return 0.5 * dot(p, metric.tmp)
end

function check_uturn(ρ, p_start, p_end, metric::DenseMetric)
    mul!(metric.tmp, metric.inv_metric, p_start)
    s1 = dot(ρ, metric.tmp)
    mul!(metric.tmp, metric.inv_metric, p_end)
    s2 = dot(ρ, metric.tmp)
    s1 > 0 && s2 > 0
end

## LEAPFROG INTEGRATOR

### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(q, p, g, model, ϵ,inv_metric, ∇!)
    
    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]
        q[i] += ϵ * inv_metric[i] * p[i]
    end

    lp, ok = ∇!(g, model, q)

    @simd for i in eachindex(p)
        p[i] += (ϵ / 2) * g[i]  
    end

    return lp, ok
end

function sample_momentum!(rng,p,inv_metric)
    @simd for i in eachindex(p)
        p[i] = randn(rng) / sqrt(inv_metric[i])
    end
end 


mutable struct PhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
    g::Vector{Float64}
    V::Float64
end

function _copy_psp!(dst::PhaseSpacePoint, src::PhaseSpacePoint)
    dst.q .= src.q
    dst.p .= src.p
    dst.g .= src.g
    dst.V = src.V
end

function logsumexp(a::Float64, b::Float64)
    m = max(a, b)
    m == -Inf && return -Inf
    return m + log(exp(a - m) + exp(b - m))
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

function kinetic_energy(p, inv_metric)
   k = 0.0
   @simd for i in eachindex(p)
        k += 0.5 * inv_metric[i] * p[i]^2
   end 
   k 
end

function find_reasonable_epsilon(rng, z, model, inv_metric, ∇!)
    ϵ = 1.0
    p_save = copy(z.p)
    q_try = similar(z.q)
    p_try = similar(z.p)
    g_try = similar(z.g)

    for _ in 1:20
        sample_momentum!(rng, z.p, inv_metric)
        H = z.V + kinetic_energy(z.p, inv_metric)

        q_try .= z.q
        p_try .= z.p
        g_try .= z.g
        lp, ok = leapfrog!(q_try, p_try, g_try, model, ϵ, inv_metric, ∇!)

        if !ok
            ϵ /= 2.0
            continue
        end

        V_try = -lp
        H_new = V_try + kinetic_energy(p_try, inv_metric)
        acc_ratio = isfinite(H_new) ? exp(min(H - H_new, 0.0)) : 0.0

        acc_ratio > 0.25 ? ϵ *= 2.0 : ϵ /= 2.0
    end

    z.p .= p_save
    max(ϵ, 1e-4)
end


function check_uturn(ρ, p_start, p_end, inv_metric)
    s1 = 0.0
    s2 = 0.0
    @inbounds @simd for i in eachindex(ρ)
        m = inv_metric[i]
        s1 += ρ[i] * m * p_start[i]
        s2 += ρ[i] * m * p_end[i]
    end
    s1 > 0 && s2 > 0
end

struct TreeScratch
    ρ_init::Vector{Float64}
    ρ_final::Vector{Float64}
    p_init_end::Vector{Float64}
    p_final_beg::Vector{Float64}
    z_propose_final::PhaseSpacePoint
    log_sum_weight_init::Ref{Float64}
    log_sum_weight_final::Ref{Float64}
end

struct NUTSScratch
    z_propose::PhaseSpacePoint
    z_fwd::PhaseSpacePoint
    z_bwd::PhaseSpacePoint
    z_propose_new::PhaseSpacePoint
    ρ::Vector{Float64}
    ρ_new::Vector{Float64}
    p_new_beg::Vector{Float64}
    p_new_end::Vector{Float64}
    log_sum_weight::Ref{Float64}
    log_sum_weight_new::Ref{Float64}
    sum_metro_prob::Ref{Float64}
    n_leapfrog::Ref{Int}
    divergent::Ref{Bool}
    tree::Vector{TreeScratch}
end

function TreeScratch(dim::Int)
    TreeScratch(
        zeros(dim), zeros(dim), zeros(dim), zeros(dim),
        PhaseSpacePoint(zeros(dim), zeros(dim), zeros(dim), 0.0),
        Ref(-Inf), Ref(-Inf)
    )
end

function NUTSScratch(dim::Int, max_depth::Int)
    NUTSScratch(
        PhaseSpacePoint(zeros(dim), zeros(dim), zeros(dim), 0.0),
        PhaseSpacePoint(zeros(dim), zeros(dim), zeros(dim), 0.0),
        PhaseSpacePoint(zeros(dim), zeros(dim), zeros(dim), 0.0),
        PhaseSpacePoint(zeros(dim), zeros(dim), zeros(dim), 0.0),
        zeros(dim), zeros(dim),
        zeros(dim), zeros(dim),
        Ref(0.0), Ref(-Inf), Ref(0.0), Ref(0), Ref(false),
        [TreeScratch(dim) for _ in 1:max_depth]
    )
end

function build_tree!(rng, z::PhaseSpacePoint, z_propose::PhaseSpacePoint,
    ρ, p_beg, p_end,
    depth, direction, ϵ, inv_metric, model, ∇!, H0, max_deltaH,
    scratch, log_sum_weight, sum_metro_prob, n_leapfrog, divergent)

    if depth == 0
        lp, ok = leapfrog!(z.q, z.p, z.g, model, direction * ϵ, inv_metric, ∇!)
        n_leapfrog[] += 1

        z.V = ok ? -lp : Inf
        H = z.V + kinetic_energy(z.p, inv_metric)
        isnan(H) && (H = Inf)

        if (H - H0) > max_deltaH
            divergent[] = true
        end

        log_sum_weight[] = logsumexp(log_sum_weight[], H0 - H)
        sum_metro_prob[] += H0 - H > 0 ? 1.0 : exp(H0 - H)

        z_propose.q .= z.q
        z_propose.g .= z.g
        z_propose.V = z.V

        ρ .+= z.p
        p_beg .= z.p
        p_end .= z.p

        return !divergent[]
    else
        scr = scratch[depth]
        fill!(scr.ρ_init, 0.0)
        scr.log_sum_weight_init[] = -Inf

        valid_init = build_tree!(rng, z, z_propose,
            scr.ρ_init, p_beg, scr.p_init_end,
            depth - 1, direction, ϵ, inv_metric, model, ∇!, H0, max_deltaH,
            scratch, scr.log_sum_weight_init, sum_metro_prob, n_leapfrog, divergent)
        if !valid_init
            return false
        end

        fill!(scr.ρ_final, 0.0)
        scr.log_sum_weight_final[] = -Inf

        valid_final = build_tree!(rng, z, scr.z_propose_final,
            scr.ρ_final, scr.p_final_beg, p_end,
            depth - 1, direction, ϵ, inv_metric, model, ∇!, H0, max_deltaH,
            scratch, scr.log_sum_weight_final, sum_metro_prob, n_leapfrog, divergent)
        if !valid_final
            return false
        end

        log_sum_weight[] = logsumexp(scr.log_sum_weight_init[], scr.log_sum_weight_final[])
        if rand(rng) < exp(scr.log_sum_weight_final[] - log_sum_weight[])
            z_propose.q .= scr.z_propose_final.q
            z_propose.g .= scr.z_propose_final.g
            z_propose.V = scr.z_propose_final.V
        end

        # Sub-tree u-turn checks (before combining clobbers ρ_init)
        valid_init_uturn = check_uturn(scr.ρ_init, p_beg, scr.p_init_end, inv_metric)
        valid_final_uturn = check_uturn(scr.ρ_final, scr.p_final_beg, p_end, inv_metric)

        # Combine momentum sums — reuse ρ_init for the combined sum
        scr.ρ_init .+= scr.ρ_final
        ρ .+= scr.ρ_init

        return check_uturn(scr.ρ_init, p_beg, p_end, inv_metric) &&
               valid_init_uturn && valid_final_uturn
    end
end


## One NUTS transition.  Returns (α, divergent).
function _nuts_transition!(rng, z, ns::NUTSScratch, model, ϵ, max_depth, inv_metric, ∇!)
    # Sample fresh momentum
    sample_momentum!(rng, z.p, inv_metric)

    # Compute gradient at current position
    lp, ok = ∇!(z.g, model, z.q)
    if !ok
        return 0.0, true
    end
    z.V = -lp

    # Initial Hamiltonian
    H0 = z.V + kinetic_energy(z.p, inv_metric)

    # Initialize forward/backward walkers to current position (need full copy incl. p)
    _copy_psp!(ns.z_fwd, z)
    _copy_psp!(ns.z_bwd, z)

    # Initialize proposal (p not needed — only q, g, V are used from proposals)
    ns.z_propose.q .= z.q
    ns.z_propose.g .= z.g
    ns.z_propose.V = z.V

    # Initialize momentum sum
    ns.ρ .= z.p

    # Reset counters
    ns.log_sum_weight[] = 0.0   # starting point has weight exp(0) = 1
    ns.sum_metro_prob[] = 0.0
    ns.n_leapfrog[] = 0
    ns.divergent[] = false

    for depth in 0:max_depth-1
        direction = rand(rng, (-1, 1))

        # Reset new subtree buffers
        fill!(ns.ρ_new, 0.0)
        ns.log_sum_weight_new[] = -Inf

        # Build new subtree extending in chosen direction
        if direction == 1
            valid = build_tree!(rng, ns.z_fwd, ns.z_propose_new,
                ns.ρ_new, ns.p_new_beg, ns.p_new_end,
                depth, direction, ϵ, inv_metric, model, ∇!, H0, 1000.0,
                ns.tree, ns.log_sum_weight_new, ns.sum_metro_prob, ns.n_leapfrog, ns.divergent)
        else
            valid = build_tree!(rng, ns.z_bwd, ns.z_propose_new,
                ns.ρ_new, ns.p_new_beg, ns.p_new_end,
                depth, direction, ϵ, inv_metric, model, ∇!, H0, 1000.0,
                ns.tree, ns.log_sum_weight_new, ns.sum_metro_prob, ns.n_leapfrog, ns.divergent)
        end

        if !valid
            break
        end

        # Multinomial coin flip: replace proposal with new subtree's?
        log_sum_weight_total = logsumexp(ns.log_sum_weight[], ns.log_sum_weight_new[])
        if rand(rng) < exp(ns.log_sum_weight_new[] - log_sum_weight_total)
            ns.z_propose.q .= ns.z_propose_new.q
            ns.z_propose.g .= ns.z_propose_new.g
            ns.z_propose.V = ns.z_propose_new.V
        end
        ns.log_sum_weight[] = log_sum_weight_total

        # Accumulate momentum
        ns.ρ .+= ns.ρ_new

        # U-turn check — walker PSPs track boundary momenta directly
        if !check_uturn(ns.ρ, ns.z_bwd.p, ns.z_fwd.p, inv_metric)
            break
        end
    end

    # Accept unconditionally (multinomial NUTS)
    z.q .= ns.z_propose.q
    z.g .= ns.z_propose.g
    z.V = ns.z_propose.V

    # Mean acceptance probability for step size adaptation
    α = ns.n_leapfrog[] > 0 ? ns.sum_metro_prob[] / ns.n_leapfrog[] : 0.0
    α = min(1.0, α)

    return α, ns.divergent[]
end

### DIM THRESHOLD FOR ENZYME FORWARD OR REVERSE

const FORWARD_MODE_THRESHOLD = 20

"""Estimate tree depth from leapfrog count: depth ≈ log2(n_leapfrog)."""
_tree_depth(n_leapfrog::Int) = n_leapfrog > 0 ? floor(Int, log2(n_leapfrog)) : 0

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
    g_test = zeros(dim)
    _, ok = ∇!(g_test, model, q_test)
    if !ok
        printstyled("✗  Gradient compilation failed (check @error messages above)\n"; color=:red, bold=true)
        error("Enzyme autodiff failed — see log above for details")
    end
    printstyled("✓  Gradient ready\n"; color=:green, bold=true)

    # Benchmark gradient (post-JIT) — 5 calls to get steady-state timing
    t0 = time_ns()
    for _ in 1:5
        ∇!(g_test, model, q_test)
    end
    grad_μs = (time_ns() - t0) / 5 / 1e3
    printstyled("   Gradient: $(round(grad_μs; digits=1)) μs/eval\n"; color=:light_black)

    return ∇!, grad_μs
end

"""Run a single NUTS chain: warmup + sampling. Returns (raw_samples::Matrix, n_divergent)."""
function _run_chain(rng, model, num_samples, ϵ₀, max_depth, warmup, ∇!; chain_id = 1, quiet = false, δ = 0.8, print_lock = ReentrantLock(), callback = nothing, metric = :diagonal)
    dim = model.dim
    ns = NUTSScratch(dim, max_depth)
    use_dense = (metric == :dense)

    # Initialize phase space at random position
    # Start diagonal even for dense — switch to dense once we have enough warmup samples
    z = PhaseSpacePoint(randn(rng, dim), zeros(dim), zeros(dim), 0.0)
    inv_metric = ones(dim)

    # Evaluate gradient at initial position
    lp, ok = ∇!(z.g, model, z.q)
    if !ok
        z.q .= 0.0
        lp, _ = ∇!(z.g, model, z.q)
    end
    z.V = -lp

    # Find reasonable initial step size
    ϵ = find_reasonable_epsilon(rng, z, model, inv_metric, ∇!)

    if !quiet
        lock(print_lock) do
            printstyled("  Chain $chain_id  "; color=:yellow, bold=true)
            printstyled("warmup $warmup iterations  ϵ₀=$(round(ϵ; sigdigits=3))\n"; color=:yellow)
        end
    end

    ## ── Stan-style three-phase windowed warmup ──
    # Phase I  (init_buffer):  step size adaptation only
    # Phase II:                step size + metric with expanding windows
    # Phase III (term_buffer): step size adaptation only (final metric)
    #
    # Stan defaults: init_buffer=75, term_buffer=50, base_window=25
    # Windows double: 25, 50, 100, 200, ... with final window stretched to fill
    if warmup >= 150  # enough room for Stan's default schedule
        init_buffer = 75
        term_buffer = 50
        base_window = 25
    else
        # Short warmup: 15%/75%/10% split
        init_buffer = max(1, warmup * 15 ÷ 100)
        term_buffer = max(1, warmup * 10 ÷ 100)
        base_window = max(1, warmup * 10 ÷ 100)
    end
    n_window = max(0, warmup - init_buffer - term_buffer)

    # Stan window schedule: doubling with final window stretched
    window_ends = Int[]  # 1-based iteration indices (relative to Phase II start) where windows end
    if n_window > 0
        window_size = base_window
        next_end = window_size
        while next_end <= n_window
            # Check if the NEXT window after this one would fit
            next_next_end = next_end + 2 * window_size
            if next_next_end > n_window
                # Stretch this window to fill remaining space
                push!(window_ends, n_window)
                break
            else
                push!(window_ends, next_end)
                window_size *= 2
                next_end += window_size
            end
        end
    end

    welford_diag = WelfordState(dim)
    welford_cov = use_dense ? WelfordCovState(dim) : nothing
    var_buf = ones(dim)
    da = DualAveraging(ϵ; δ)
    ϵ_curr = ϵ
    n_warmup_div = 0
    window_end_idx = 1  # index into window_ends

    t_warmup_start = time_ns()
    for i in 1:warmup
        α, div = _nuts_transition!(rng, z, ns, model, ϵ_curr, max_depth, inv_metric, ∇!)
        n_warmup_div += div
        ϵ_curr = adapt!(da, α)

        if callback !== nothing
            try
                put!(callback, (chain_id=chain_id, phase=:warmup, iteration=i, total=warmup,
                                step_size=ϵ_curr, tree_depth=_tree_depth(ns.n_leapfrog[]),
                                accept_rate=α, n_divergent=n_warmup_div,
                                elapsed_ns=time_ns() - t_warmup_start))
            catch; end
        end

        # Phase II: metric adaptation with expanding windows
        if i > init_buffer && i <= init_buffer + n_window
            welford_update!(welford_diag, z.q)
            if welford_cov !== nothing
                welford_update!(welford_cov, z.q)
            end

            window_idx = i - init_buffer
            if window_end_idx <= length(window_ends) && window_idx == window_ends[window_end_idx]
                if use_dense && welford_cov !== nothing && welford_cov.n > dim
                    # Enough samples — switch to dense metric
                    inv_metric = _update_dense_metric!(
                        inv_metric isa DenseMetric ? inv_metric : DenseMetric(dim),
                        welford_cov, dim)
                    welford_cov = WelfordCovState(dim)
                else
                    # Diagonal update
                    welford_variance!(var_buf, welford_diag)
                    n = welford_diag.n
                    if inv_metric isa Vector
                        @inbounds for j in eachindex(inv_metric)
                            inv_metric[j] = max((n / (n + 5.0)) * var_buf[j] + (5.0 / (n + 5.0)) * 1e-3, 1e-3)
                        end
                    end
                    # Don't reset welford_cov — keep accumulating for dense
                end
                welford_diag = WelfordState(dim)
                window_end_idx += 1

                # Re-find step size and restart dual averaging (Stan does this at every window end)
                ϵ_curr = find_reasonable_epsilon(rng, z, model, inv_metric, ∇!)
                da = DualAveraging(ϵ_curr; δ)
            end
        end
    end

    ϵ_adapted = adapted_ε(da)

    if !quiet
        lock(print_lock) do
            printstyled("  Chain $chain_id  "; color=:green, bold=true)
            printstyled("adapted ϵ = $(round(ϵ_adapted; sigdigits=4))\n"; color=:white, bold=true)
            if n_warmup_div > 0
                printstyled("  Chain $chain_id  ⚠ $n_warmup_div divergent transitions during warmup\n"; color=:yellow)
            end
        end
    end

    ## ── Sampling ──
    raw = Matrix{Float64}(undef, dim, num_samples)
    n_divergent = 0
    total_leapfrog = 0
    progress_interval = max(1, num_samples ÷ 10)
    t_sample_start = time_ns()
    @inbounds for i in 1:num_samples
        _, div = _nuts_transition!(rng, z, ns, model, ϵ_adapted, max_depth, inv_metric, ∇!)
        n_divergent += div
        total_leapfrog += ns.n_leapfrog[]
        raw[:, i] .= z.q
        if !quiet && i % progress_interval == 0
            lock(print_lock) do
                pct = 100i ÷ num_samples
                printstyled("  Chain $chain_id  $(lpad(pct, 3))%  $i/$num_samples\n"; color=:light_black)
            end
        end
        if callback !== nothing
            try
                put!(callback, (chain_id=chain_id, phase=:sampling, iteration=i, total=num_samples,
                                step_size=ϵ_adapted, tree_depth=_tree_depth(ns.n_leapfrog[]),
                                accept_rate=min(1.0, ns.sum_metro_prob[] / max(1, ns.n_leapfrog[])),
                                n_divergent=n_divergent,
                                elapsed_ns=time_ns() - t_sample_start))
            catch; end
        end
    end
    elapsed_s = (time_ns() - t_sample_start) / 1e9

    if !quiet
        lock(print_lock) do
            avg_lf = total_leapfrog / num_samples
            μs_per_lf = elapsed_s * 1e6 / total_leapfrog
            printstyled("  Chain $chain_id  "; color=:green, bold=true)
            printstyled("$(num_samples) samples, $(total_leapfrog) leapfrog steps (avg $(round(avg_lf; digits=1))/sample), ")
            printstyled("$(round(μs_per_lf; digits=1)) μs/step, ")
            printstyled("$(round(elapsed_s; digits=2))s total\n")
            if n_divergent > 0
                printstyled("  Chain $chain_id  ⚠ $n_divergent divergent transitions\n"; color=:yellow)
            end
        end
    end

    if callback !== nothing
        try
            put!(callback, (chain_id=chain_id, phase=:done, iteration=num_samples, total=num_samples,
                            step_size=ϵ_adapted, tree_depth=0, accept_rate=0.0,
                            n_divergent=n_divergent, elapsed_ns=time_ns() - t_sample_start))
        catch; end
    end

    return raw, n_divergent
end

"""
    sample(model, num_samples; warmup=1000, chains=4, ϵ=0.1, max_depth=10, ad=:auto, seed=nothing, δ=0.8, metric=:auto) → Chains

Run NUTS (No-U-Turn Sampler) on a compiled model. Returns a `Chains` object.

# Arguments
- `model`: A `ModelLogDensity` from `make(data)`.
- `num_samples`: Number of post-warmup draws per chain.
- `warmup`: Number of warmup/adaptation steps per chain.
- `chains`: Number of parallel chains.
- `ϵ`: Initial step size (adapted during warmup).
- `max_depth`: Maximum tree depth for NUTS.
- `ad`: Autodiff mode — `:auto`, `:forward`, or `:reverse`.
- `seed`: RNG seed for reproducibility.
- `δ`: Target acceptance probability for step size adaptation.
- `metric`: Mass matrix type — `:auto` (dense if dim ≤ 500), `:dense`, or `:diagonal`.
"""
function sample(model, num_samples; ϵ = 0.1, max_depth = 10, warmup = 1000, ad = :auto, chains = 4, seed = nothing, δ = 0.8, callback = nothing, metric = :auto)
    num_samples > 0 || throw(ArgumentError("sample: num_samples must be > 0, got $num_samples"))
    warmup >= 0 || throw(ArgumentError("sample: warmup must be >= 0, got $warmup"))
    chains > 0 || throw(ArgumentError("sample: chains must be > 0, got $chains"))
    if chains > 1 && Threads.nthreads() < chains + 1
        @warn "PhaseSkate needs $(chains + 1) threads for $chains parallel chains " *
              "(have $(Threads.nthreads())). Chains will run sequentially. " *
              "Start Julia with: julia -t $(chains + 1)"
    end
    ∇!, _ = _make_grad(model; ad)

    # Resolve metric type
    resolved_metric = if metric == :auto
        model.dim ≤ 500 ? :dense : :diagonal
    else
        metric
    end

    # Generate per-chain RNGs for thread safety
    master_rng = seed === nothing ? Xoshiro() : Xoshiro(seed)
    chain_seeds = [rand(master_rng, UInt64) for _ in 1:chains]

    metric_label = resolved_metric == :dense ? "dense" : "diagonal"
    printstyled("~  Sampling "; color=:cyan, bold=true)
    printstyled("$chains chain(s) × $num_samples samples"; color=:white, bold=true)
    printstyled("  max_depth=$max_depth  metric=$metric_label\n"; color=:cyan)

    print_lock = ReentrantLock()

    tasks = [Threads.@spawn _run_chain(Xoshiro(chain_seeds[c]),
                                        model, num_samples, ϵ, max_depth, warmup, ∇!;
                                        chain_id=c, quiet=(callback !== nothing), δ, print_lock, callback, metric=resolved_metric)
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
