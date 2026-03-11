## ── Sampling callback: bridges sampler threads → TUI event loop ─────────────
##
## The sampler puts NamedTuples into a Channel{Any}. The TUI drains them
## directly in its update loop — no converter task needed.

"""Convert a NamedTuple from the sampler callback into a ChainProgress struct."""
function _to_chain_progress(nt)::ChainProgress
    ChainProgress(
        nt.chain_id,
        nt.phase,
        nt.iteration,
        nt.total,
        nt.step_size,
        nt.tree_depth,
        nt.accept_rate,
        nt.n_divergent,
        UInt64(nt.elapsed_ns),
    )
end
