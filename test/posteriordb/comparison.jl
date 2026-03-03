# Statistical comparison utilities for PosteriorDB validation tests.
# Assumes `using Statistics` is active in the calling context.

"""
    _z_critical(α)

Two-tailed z critical value for common significance levels.
"""
function _z_critical(α::Float64)
    α ≈ 0.001 && return 3.291
    α ≈ 0.01  && return 2.576
    α ≈ 0.05  && return 1.960
    error("Unsupported α level: $α. Use 0.001, 0.01, or 0.05.")
end

"""
    _ks_critical(n, m, α)

Asymptotic critical value for the two-sample Kolmogorov-Smirnov test.
"""
function _ks_critical(n::Int, m::Int, α::Float64)
    c = if α ≈ 0.001
        1.949
    elseif α ≈ 0.01
        1.628
    elseif α ≈ 0.05
        1.358
    else
        error("Unsupported α level: $α")
    end
    return c * sqrt((n + m) / (n * m))
end

"""
    _ks_statistic(x, y)

Two-sample Kolmogorov-Smirnov statistic: max |F_x(t) - F_y(t)|.
Uses a sortperm-based merge for O((n+m) log(n+m)) performance.
"""
function _ks_statistic(x::Vector{Float64}, y::Vector{Float64})
    n1 = length(x)
    n2 = length(y)
    combined = vcat(x, y)
    idx = sortperm(combined)

    d_max = 0.0
    ecdf_x = 0.0
    ecdf_y = 0.0

    for k in eachindex(idx)
        if idx[k] <= n1
            ecdf_x += 1.0 / n1
        else
            ecdf_y += 1.0 / n2
        end
        d = abs(ecdf_x - ecdf_y)
        d_max > d || (d_max = d)
    end

    return d_max
end

"""
    _subsample(x, n)

Thin vector `x` to `n` evenly-spaced elements.
"""
function _subsample(x::Vector{Float64}, n::Int)
    m = length(x)
    n >= m && return x
    indices = round.(Int, range(1, m, length=n))
    return x[indices]
end

# ─── Public comparison functions ─────────────────────────────────────────────

"""
    z_test_means(ps_draws, ref_draws; α=0.01)

Two-sample z-test comparing posterior means for each parameter.
Returns `(pass, max_z, failing_params)`.
"""
function z_test_means(ps_draws::Dict{String, Vector{Float64}},
                      ref_draws::Dict{String, Vector{Float64}};
                      α::Float64=0.01)
    z_crit = _z_critical(α)
    max_z = 0.0
    failing = String[]

    for name in sort(collect(keys(ps_draws)))
        haskey(ref_draws, name) || continue
        ps  = ps_draws[name]
        ref = ref_draws[name]

        n_ps  = length(ps)
        n_ref = length(ref)
        (n_ps < 2 || n_ref < 2) && continue

        mean_ps  = sum(ps) / n_ps
        mean_ref = sum(ref) / n_ref

        var_ps  = sum((xi - mean_ps)^2  for xi in ps)  / (n_ps  - 1)
        var_ref = sum((xi - mean_ref)^2 for xi in ref) / (n_ref - 1)

        se = sqrt(var_ps / n_ps + var_ref / n_ref)
        z  = se > 0 ? abs(mean_ps - mean_ref) / se : 0.0

        max_z = max(max_z, z)
        if z > z_crit
            push!(failing, name)
        end
    end

    return (pass=isempty(failing), max_z=max_z, failing_params=failing)
end

"""
    ks_test(ps_draws, ref_draws; α=0.01)

Two-sample KS test per parameter. Both samples are thinned to equal size.
Returns `(pass, max_stat, failing_params)`.
"""
function ks_test(ps_draws::Dict{String, Vector{Float64}},
                 ref_draws::Dict{String, Vector{Float64}};
                 α::Float64=0.01)
    max_stat = 0.0
    failing = String[]

    for name in sort(collect(keys(ps_draws)))
        haskey(ref_draws, name) || continue
        ps  = ps_draws[name]
        ref = ref_draws[name]

        n = min(length(ps), length(ref))
        n < 10 && continue
        ps_sub  = _subsample(ps,  n)
        ref_sub = _subsample(ref, n)

        D     = _ks_statistic(ps_sub, ref_sub)
        D_crit = _ks_critical(n, n, α)

        max_stat = max(max_stat, D)
        if D > D_crit
            push!(failing, name)
        end
    end

    return (pass=isempty(failing), max_stat=max_stat, failing_params=failing)
end

"""
    quantile_check(ps_draws, ref_draws; quantiles=[0.025, 0.5, 0.975], tol=0.05)

Compare quantiles of posterior samples. Differences are normalized by the
reference IQR. Pass if all normalized differences < `tol`.
Returns `(pass, max_diff, failing_params)`.
"""
function quantile_check(ps_draws::Dict{String, Vector{Float64}},
                        ref_draws::Dict{String, Vector{Float64}};
                        quantiles::Vector{Float64}=[0.025, 0.5, 0.975],
                        tol::Float64=0.05)
    max_diff = 0.0
    failing = String[]

    for name in sort(collect(keys(ps_draws)))
        haskey(ref_draws, name) || continue
        ps  = ps_draws[name]
        ref = ref_draws[name]

        ref_iqr = quantile(ref, 0.75) - quantile(ref, 0.25)
        ref_iqr ≈ 0 && continue   # skip degenerate parameters

        for q in quantiles
            ps_q  = quantile(ps, q)
            ref_q = quantile(ref, q)

            d = abs(ps_q - ref_q) / ref_iqr
            max_diff = max(max_diff, d)
            if d > tol && !(name in failing)
                push!(failing, name)
            end
        end
    end

    return (pass=isempty(failing), max_diff=max_diff, failing_params=failing)
end
