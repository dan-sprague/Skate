module SkateReactantExt

using Skate: Skate, ModelLogDensity, XLABackend, Chains,
             _setup_ad, _compile_gradient!, _warmup, _constrain_fn
using Reactant: Reactant, @compile, ConcreteRArray, TracedRArray
using Enzyme: Enzyme

## ── Scalar extraction override for TracedRArray ──────────────────────────────
## q[i:i] emits a static stablehlo.slice (no assertscalar, XLA optimizes
## known bounds). Reshape to 0-d + [begin] extracts TracedRNumber without
## assertscalar (0-d getindex is unguarded).
@inline Skate._qscalar(q::TracedRArray, i) = reshape(q[i:i], ())[]

## ── XLA HMC kernel ─────────────────────────────────────────────────────────
##
## Functional (no mutation) HMC transition compiled as a single XLA program.
##
## Inside @compile, broadcasting (.+, .*) on TracedRArrays does NOT allocate:
## XLA fuses element-wise ops into a single kernel. The broadcasting syntax
## is just how Julia expresses element-wise operations — XLA's optimizer
## eliminates all intermediates.
##
## Arguments are split by tracing role:
##   Traced (vary per call):  q, data
##   Constant (baked in):     ℓ, ε, L
##
## The for loop over 1:L is unrolled at trace time (L is a concrete Int).

"""
    _xla_hmc_step(ℓ, q, data, ε, L)

Single HMC transition. Returns `(q_next, α)`.

The gradient is computed once before the loop and reused across leapfrog
steps (L+1 gradient evaluations total, matching the CPU kernel).
"""
function _xla_hmc_step(ℓ, q, data, ε, L)
    T = eltype(q)
    dim = length(q)

    # ── Sample momentum ──
    p = randn(T, dim)

    # ── Current Hamiltonian ──
    lp₀ = ℓ(q, data)
    H_current = -lp₀ + T(0.5) * sum(abs2, p)

    # ── Leapfrog integration ──
    # Initial gradient (reused as first half-kick)
    q_prop = q
    p_prop = p
    (g,) = Enzyme.gradient(Enzyme.Reverse, x -> ℓ(x, data), q_prop)

    for _ in 1:L
        p_prop = p_prop .+ (ε / 2) .* g        # half-kick
        q_prop = q_prop .+ ε .* p_prop           # full position step
        (g,) = Enzyme.gradient(Enzyme.Reverse, x -> ℓ(x, data), q_prop)
        p_prop = p_prop .+ (ε / 2) .* g        # half-kick
    end

    # ── Proposal Hamiltonian ──
    lp = ℓ(q_prop, data)
    H_proposal = -lp + T(0.5) * sum(abs2, p_prop)

    # ── Metropolis-Hastings (branchless) ──
    # If ΔH is NaN/Inf: exp(-ΔH) → 0 or NaN, min(1,·) ≤ 1, rand() < · → false → reject.
    ΔH = H_proposal - H_current
    α = min(one(T), exp(-ΔH))
    accept = rand(T) < α
    q_next = ifelse.(accept, q_prop, q)

    return q_next, α
end

## ── XLA sampling dispatch ──────────────────────────────────────────────────

function Skate._sample_impl(::XLABackend, model, num_samples; ϵ, L, warmup, ad, chains)
    dim = model.dim

    model.data === nothing &&
        error("XLABackend requires a model created with @spec (data must not be nothing)")

    # ── Phase 1: CPU warmup for step-size adaptation ──
    printstyled("⚙  XLA: CPU warmup ($warmup iterations)...\n"; color=:yellow, bold=true)
    ∇!, ad_label = Skate._setup_ad(model, dim, ad)
    Skate._compile_gradient!(model, dim, ∇!, ad_label)

    ε_adapted, q_init = Skate._warmup(model, ∇!, dim; ϵ, L, warmup)
    printstyled("✓  Adapted ε = $(round(ε_adapted; sigdigits=4))\n"; color=:green, bold=true)

    # ── Phase 2: Convert data to XLA arrays ──
    data_xla = Reactant.to_rarray(model.data)
    q_xla = ConcreteRArray(q_init)

    # ── Phase 3: Compile XLA kernel ──
    printstyled("⚙  Compiling XLA HMC kernel (L=$L)...\n"; color=:yellow, bold=true)
    compiled_step = @compile _xla_hmc_step(model.ℓ, q_xla, data_xla, ε_adapted, L)
    printstyled("✓  XLA kernel compiled\n"; color=:green, bold=true)

    # ── Phase 4: Sample ──
    if chains > 1
        printstyled("⚠  XLA: multi-chain runs sequentially (no multi-device yet)\n";
                    color=:yellow)
    end
    printstyled("~  Sampling $chains chain(s) × $num_samples draws on XLA\n";
                color=:cyan, bold=true)

    all_raws = Vector{Matrix{Float64}}(undef, chains)
    for c in 1:chains
        raw = Matrix{Float64}(undef, dim, num_samples)
        # Each chain starts from warmup position (perturbed for diversity)
        q = c == 1 ? q_xla : ConcreteRArray(q_init .+ 0.1 .* randn(dim))

        for i in 1:num_samples
            q, _ = compiled_step(model.ℓ, q, data_xla, ε_adapted, L)
            raw[:, i] .= Array(q)
        end

        printstyled("[chain $c] Done\n"; color=:green)
        all_raws[c] = raw
    end

    printstyled("✓  XLA sampling complete\n"; color=:green, bold=true)
    constrain_fn = Skate._constrain_fn(model)
    return Chains(all_raws, constrain_fn)
end

end # module SkateReactantExt
