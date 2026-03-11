module PhaseSkateTUIExt

using PhaseSkate
using Tachikoma
using Tachikoma: @tachikoma_app, Layout, Vertical, Horizontal, Fixed, Fill,
                 split_layout, Block, Paragraph, Span, Gauge, BarChart, BarEntry,
                 Chart, DataSeries, DataTable, DataColumn, TabBar, StatusBar,
                 SelectableList, ScrollPane, REPLWidget, CodeEditor, Rect, Buffer,
                 render, tstyle, word_wrap, TaskQueue,
                 handle_key!, drain!, close!
using Statistics: quantile
using Random: randn

@tachikoma_app

include("theme.jl")
include("types.jl")
include("callbacks.jl")
include("tabs/sampling_tab.jl")
include("tabs/diagnostics_tab.jl")
include("tabs/trace_tab.jl")
include("tabs/density_tab.jl")
include("tabs/sbc_tab.jl")
include("tabs/repl_tab.jl")
include("app.jl")

## ── PhaseSkate.app() — main IDE entry point ─────────────────────────────────

"""
    PhaseSkate.app(; kwargs...)

Launch the PhaseSkate IDE — a full terminal-based environment for Bayesian modeling.

Opens with a model editor, live sampling dashboard, diagnostics, trace plots,
density histograms, SBC visualization, and an embedded Julia REPL.

# Keyword arguments
- `model_source::String`: Initial `@skate` model code to load in the editor.
- `model::ModelLogDensity`: Pre-compiled model to start sampling immediately.
- `chains::Chains`: Pre-existing chains to inspect.
- `sbc::SBCResult`: SBC result to visualize.
- `num_samples::Int=1000`: Post-warmup samples per chain.
- `warmup::Int=1000`: Warmup iterations per chain.
- `chains_count::Int=4`: Number of parallel chains.
- `max_depth::Int=10`: Maximum NUTS tree depth.
- `ad::Symbol=:auto`: Autodiff mode (`:auto`, `:forward`, `:reverse`).
"""
function PhaseSkate.app(;
    model_source::String=_DEFAULT_MODEL_SOURCE,
    model::Union{Nothing, PhaseSkate.ModelLogDensity}=nothing,
    chains::Union{Nothing, PhaseSkate.Chains}=nothing,
    sbc::Union{Nothing, PhaseSkate.SBCResult}=nothing,
    num_samples::Int=1000,
    warmup::Int=1000,
    chains_count::Int=4,
    max_depth::Int=10,
    ad::Symbol=:auto,
)
    m = IDEModel(;
        model=model,
        chains=chains,
        sbc_result=sbc,
        n_chains=chains_count,
        num_samples=num_samples,
        warmup=warmup,
        max_depth=max_depth,
        ad=ad,
        model_source=model_source,
    )

    Tachikoma.app(m; default_bindings=false)

    m.chains
end

## ── dashboard() — backward-compatible entry points ──────────────────────────

"""
    dashboard(model::ModelLogDensity, num_samples; warmup=1000, chains=4, ...) → Chains

Launch the IDE with a pre-compiled model. Starts sampling immediately.
"""
function PhaseSkate.dashboard(model::PhaseSkate.ModelLogDensity, num_samples::Int;
                              warmup::Int=1000,
                              chains::Int=4,
                              kwargs...)
    PhaseSkate.app(;
        model=model,
        num_samples=num_samples,
        warmup=warmup,
        chains_count=chains,
        kwargs...,
    )
end

"""
    dashboard(chains::Chains; sbc=nothing)

Launch the IDE in view-only mode to explore existing results.
"""
function PhaseSkate.dashboard(chains::PhaseSkate.Chains; sbc=nothing)
    PhaseSkate.app(; chains=chains, sbc=sbc)
    nothing
end

end # module PhaseSkateTUIExt
