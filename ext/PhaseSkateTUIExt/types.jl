## ── Types for the PhaseSkate IDE ─────────────────────────────────────────────

## ── Enums ────────────────────────────────────────────────────────────────────

@enum OutputTab SamplingTab=1 DiagnosticsTab=2 TraceTab=3 DensityTab=4 SBCTab=5

const OUTPUT_TAB_LABELS = ["Sampling", "Diagnostics", "Traces", "Density", "SBC"]
const N_OUTPUT_TABS = 5

@enum FocusTarget FocusEditor FocusOutput FocusRepl

## ── ChainProgress ────────────────────────────────────────────────────────────

struct ChainProgress
    chain_id::Int
    phase::Symbol          # :warmup, :sampling, :done
    iteration::Int
    total::Int
    step_size::Float64
    tree_depth::Int
    accept_rate::Float64
    n_divergent::Int
    elapsed_ns::UInt64
end

## ── IDEModel ─────────────────────────────────────────────────────────────────

mutable struct IDEModel <: Tachikoma.Model
    # Focus & layout
    focus::FocusTarget
    active_tab::OutputTab

    # Editor
    editor::CodeEditor
    compile_error::Union{Nothing, String}

    # Model state
    model::Union{Nothing, PhaseSkate.ModelLogDensity}

    # Sampling state
    chain_progress::Vector{ChainProgress}
    n_chains::Int
    num_samples::Int
    warmup::Int
    max_depth::Int
    ad_mode::Symbol
    sampling_done::Bool
    sampling_channel::Channel{Any}

    # Benchmark
    grad_μs::Float64
    leapfrog_μs::Float64
    estimated_seconds::Float64
    benchmark_done::Bool

    # Results
    chains::Union{Nothing, PhaseSkate.Chains}
    sbc_result::Union{Nothing, PhaseSkate.SBCResult}
    selected_param_idx::Int
    param_labels::Vector{String}
    diagnostics_rows::Vector{PhaseSkate._RowSummary}

    # System
    tq::Tachikoma.TaskQueue
    repl_widget::Union{Nothing, REPLWidget}
    show_help::Bool
    quit::Bool

    # Binary compilation
    compiling_binary::Bool
    binary_output::Vector{String}       # juliac stdout/stderr lines
    binary_path::Union{Nothing, String} # path to compiled binary on success

    # Sampling kwargs for pass-through
    sampling_kwargs::NamedTuple
end

## ── Constructors ─────────────────────────────────────────────────────────────

const _DEFAULT_MODEL_SOURCE = """@skate MyModel begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        for i in 1:N
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end
"""

function IDEModel(;
    model=nothing,
    chains=nothing,
    sbc_result=nothing,
    n_chains=4,
    num_samples=1000,
    warmup=1000,
    max_depth=10,
    ad=:auto,
    model_source=_DEFAULT_MODEL_SOURCE,
    sampling_kwargs=(;),
)
    channel = Channel{Any}(1024)

    param_labels = String[]
    diagnostics_rows = PhaseSkate._RowSummary[]
    if chains !== nothing
        param_labels = _build_param_labels(chains)
        diagnostics_rows = _compute_diagnostics(chains)
    end

    editor = CodeEditor(;
        text=model_source,
        show_line_numbers=true,
        tab_width=4,
        focused=true,
        mode=:insert,
    )

    initial_focus = chains !== nothing ? FocusOutput : FocusEditor
    initial_tab = chains !== nothing ? DiagnosticsTab : SamplingTab

    IDEModel(
        initial_focus,
        initial_tab,
        editor,
        nothing,            # compile_error
        model,
        ChainProgress[],    # chain_progress
        n_chains,
        num_samples,
        warmup,
        max_depth,
        ad,
        chains !== nothing, # sampling_done
        channel,
        0.0, 0.0, 0.0, false,
        chains,
        sbc_result,
        1,
        param_labels,
        diagnostics_rows,
        Tachikoma.TaskQueue(),
        nothing,            # repl_widget
        false,
        false,
        false,              # compiling_binary
        String[],           # binary_output
        nothing,            # binary_path
        sampling_kwargs,
    )
end

## ── Also keep DashboardModel as an alias for backward compat ─────────────────

function DashboardModel(;
    model=nothing,
    chains=nothing,
    sbc_result=nothing,
    n_chains=4,
    num_samples=1000,
    warmup=1000,
    sampling_kwargs=(;),
)
    IDEModel(;
        model, chains, sbc_result, n_chains, num_samples, warmup,
        sampling_kwargs,
    )
end

## ── Helpers ──────────────────────────────────────────────────────────────────

function _build_param_labels(chains::PhaseSkate.Chains)
    labels = String[]
    for p in chains.params
        for k in 1:length(p.cols)
            push!(labels, PhaseSkate._elem_label(p.name, p.shape, k))
        end
    end
    labels
end

function _compute_diagnostics(chains::PhaseSkate.Chains)
    ns = size(chains.data, 1)
    nc = size(chains.data, 3)
    N = ns * nc
    rows = PhaseSkate._RowSummary[]
    for p in chains.params
        for (k, col) in enumerate(p.cols)
            x = @view chains.data[:, col, :]
            s = 0.0; s2 = 0.0
            @inbounds for j in 1:nc
                @simd for i in 1:ns
                    v = x[i, j]; s += v; s2 += v * v
                end
            end
            μ = s / N
            σ = sqrt(max(s2 / N - μ * μ, 0.0))
            pooled = vec(x)
            lo = quantile(pooled, 0.025)
            hi = quantile(pooled, 0.975)
            rhat, essb, esst = PhaseSkate._col_diagnostics(x)
            warn = rhat > PhaseSkate._RHAT_WARN || essb < PhaseSkate._ESS_WARN || esst < PhaseSkate._ESS_WARN
            label = PhaseSkate._elem_label(p.name, p.shape, k)
            push!(rows, (label=label, mean=μ, std=σ, q025=lo, q975=hi,
                         rhat=rhat, essb=essb, esst=esst, warn=warn))
        end
    end
    rows
end
