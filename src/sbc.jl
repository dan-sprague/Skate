## ── SBC: Simulation-Based Calibration ────────────────────────────────────────
## Reference: https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html

struct SBCResult
    ranks::Matrix{Int}            # (N_simulations × num_scalar_params)
    param_names::Vector{String}   # flattened labels: "mu", "sigma[1]", ...
    M::Int                        # thinned posterior draws per simulation
    p_values::Vector{Float64}     # chi-squared uniformity p-value per param
    N::Int                        # number of successful simulations
end

## ── Internal helpers ─────────────────────────────────────────────────────────

function _discover_params(theta::NamedTuple)
    layout = Tuple{Symbol, Tuple, Int}[]
    for name in keys(theta)
        val = theta[name]
        if val isa Float64
            push!(layout, (name, (), 1))
        else
            push!(layout, (name, size(val), length(val)))
        end
    end
    return layout
end

function _build_sbc_labels(layout)
    labels = String[]
    for (name, shape, nscalars) in layout
        for k in 1:nscalars
            push!(labels, _elem_label(name, shape, k))
        end
    end
    return labels
end

function _flatten_theta(theta::NamedTuple, layout)
    flat = Float64[]
    for (name, _, _) in layout
        val = theta[name]
        if val isa Float64
            push!(flat, val)
        else
            append!(flat, vec(val))
        end
    end
    return flat
end

"""
    _sbc_sample(model, M, num_samples, warmup, chains, L, ϵ, ad) → Chains

Sample and thin to M roughly independent draws. Doubles num_samples until
min ESS ≥ M across all parameters, then thins to M draws.
"""
function _sbc_sample(model, M, num_samples, warmup, chains, L, ϵ, ad)
    ns = num_samples
    max_doublings = 4
    for attempt in 0:max_doublings
        ch = redirect_stdout(devnull) do
            sample(model, ns; ϵ=ϵ, L=L, warmup=warmup, ad=ad, chains=chains)
        end
        ess = min_ess(ch)
        if ess ≥ M || attempt == max_doublings
            return thin(ch, M)
        end
        ns *= 2
    end
end

function _compute_ranks!(ranks, sim_idx, ch::Chains, theta_flat, layout)
    nsamples = size(ch.data, 1)
    nchains  = size(ch.data, 3)
    col_offset = 0
    for (name, _, nscalars) in layout
        info = ch.name_map[name]
        for k in 1:nscalars
            col = info.cols[k]
            true_val = theta_flat[col_offset + k]
            rank = 0
            @inbounds for c in 1:nchains, s in 1:nsamples
                if ch.data[s, col, c] < true_val
                    rank += 1
                end
            end
            ranks[sim_idx, col_offset + k] = rank
        end
        col_offset += nscalars
    end
    M_actual = nsamples * nchains
    return M_actual
end

function _chi2_pvalue(chi2::Float64, df::Int)
    chi2 <= 0.0 && return 1.0
    _, Q = gamma_inc(df / 2.0, chi2 / 2.0)
    return Q
end

function _chi_squared_uniformity(ranks::AbstractVector{Int}, M::Int, J::Int)
    N = length(ranks)
    bin_counts = zeros(Int, J)
    for r in ranks
        bin = min(J, floor(Int, r / (M + 1) * J) + 1)
        bin_counts[bin] += 1
    end
    expected = N / J
    chi2 = sum((c - expected)^2 / expected for c in bin_counts)
    return _chi2_pvalue(chi2, J - 1)
end

## ── Main entry point ─────────────────────────────────────────────────────────

"""
    sbc(simulate; N=100, M=200, num_samples=1000, ...)

Run Simulation-Based Calibration.

`simulate()` takes no arguments and returns `(theta_true::NamedTuple, model::ModelLogDensity)`.
The function should draw parameters from the prior, simulate data, build the model, and return both.

Posterior draws are thinned to M roughly independent samples using ESS-based
thinning. If the effective sample size is less than M, the sampler doubles
`num_samples` (up to 4 times) until min ESS ≥ M.

Returns an `SBCResult` with rank statistics and chi-squared uniformity p-values.
"""
function sbc(simulate::Function;
             N::Int = 100,
             M::Int = 200,
             num_samples::Int = 1000,
             warmup::Int = 500,
             chains::Int = 1,
             L::Int = 10,
             ϵ::Float64 = 0.1,
             ad::Symbol = :auto,
             bins::Int = 20)

    # first simulation: discover parameter layout
    theta_true, model = simulate()
    layout = _discover_params(theta_true)
    num_scalar = sum(p[3] for p in layout)
    labels = _build_sbc_labels(layout)

    ranks = Matrix{Int}(undef, N, num_scalar)
    valid = trues(N)

    printstyled("SBC: $N simulations, $num_scalar parameters, M=$M\n"; color=:cyan, bold=true)

    # run first simulation
    M_actual = M
    try
        ch = _sbc_sample(model, M, num_samples, warmup, chains, L, ϵ, ad)
        theta_flat = _flatten_theta(theta_true, layout)
        M_actual = _compute_ranks!(ranks, 1, ch, theta_flat, layout)
    catch e
        @warn "SBC simulation 1 failed: $e"
        valid[1] = false
    end
    print("\r  SBC progress: 1 / $N")

    # remaining simulations
    for i in 2:N
        try
            theta_true_i, model_i = simulate()
            ch = _sbc_sample(model_i, M, num_samples, warmup, chains, L, ϵ, ad)
            theta_flat = _flatten_theta(theta_true_i, layout)
            _compute_ranks!(ranks, i, ch, theta_flat, layout)
        catch e
            @warn "SBC simulation $i failed: $e"
            valid[i] = false
        end
        print("\r  SBC progress: $i / $N")
    end
    println()

    # filter to valid simulations
    n_valid = count(valid)
    if n_valid < N
        printstyled("  $n_valid / $N simulations succeeded\n"; color=:yellow)
    end
    valid_ranks = ranks[valid, :]

    # chi-squared uniformity test
    p_values = [_chi_squared_uniformity(valid_ranks[:, j], M_actual, bins)
                for j in 1:num_scalar]

    printstyled("SBC: complete\n"; color=:green, bold=true)
    return SBCResult(valid_ranks, labels, M_actual, p_values, n_valid)
end

## ── Diagnostics ──────────────────────────────────────────────────────────────

"""
    calibrated(result; alpha=0.01)

Return `true` if all parameters pass the chi-squared uniformity test at level `alpha`.
"""
calibrated(result::SBCResult; alpha::Float64 = 0.01) = all(p -> p > alpha, result.p_values)

## ── Pretty-print ─────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", r::SBCResult)
    println(io, "SBC Result: $(r.N) simulations, $(r.M) posterior draws, $(length(r.param_names)) parameters\n")

    hdr = @sprintf("  %-20s %10s %10s %10s %8s", "Parameter", "Mean Rank", "Std Rank", "p-value", "Status")
    println(io, hdr)
    println(io, "  " * "─"^62)

    for j in 1:length(r.param_names)
        col = @view r.ranks[:, j]
        rmean = Statistics.mean(col)
        rstd  = Statistics.std(col)
        pval  = r.p_values[j]
        status = pval < 0.01 ? "FAIL" : "ok"

        row = @sprintf("  %-20s %10.1f %10.1f %10.4f %8s", r.param_names[j], rmean, rstd, pval, status)
        if pval < 0.01
            printstyled(io, row, "\n"; color=:red)
        elseif pval < 0.05
            printstyled(io, row, "\n"; color=:yellow)
        else
            println(io, row)
        end
    end

    n_fail = count(p -> p < 0.01, r.p_values)
    println(io)
    if n_fail > 0
        printstyled(io, "  $n_fail parameter(s) failed calibration (p < 0.01)\n"; color=:red)
    else
        printstyled(io, "  All parameters passed calibration check.\n"; color=:green)
    end
end
