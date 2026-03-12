## ── Diagnostics Tab: DataTable with R-hat/ESS/CI ────────────────────────────

function _render_diagnostics_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        msg = if !isempty(m.chain_progress)
            "Warming up..."
        else
            "Waiting for sampling"
        end
        render(Paragraph(msg; block=Block(; title="Diagnostics")), area, buf)
        return
    end

    rows = m.diagnostics_rows
    isempty(rows) && return

    # Sort: flagged (warn=true) first, then by label
    perm = sortperm(rows; by=r -> (!r.warn, r.label))
    sorted = rows[perm]

    ns = size(m.chains.data, 1)
    n_warn = count(r -> r.warn, sorted)
    n_info = m.sampling_done ? "" : " (n=$(ns))"
    warn_info = n_warn > 0 ? " ⚠$(n_warn)" : ""
    inner_w = area.width - 2

    # Build label column with warning flag prefix
    labels = [_diag_label(r) for r in sorted]
    rhats  = [_fmt(r.rhat, 3) for r in sorted]
    essbs  = [string(round(Int, r.essb)) for r in sorted]

    if inner_w >= 75
        means  = [_fmt(r.mean, 3) for r in sorted]
        stds   = [_fmt(r.std, 3) for r in sorted]
        q025s  = [_fmt(r.q025, 2) for r in sorted]
        q975s  = [_fmt(r.q975, 2) for r in sorted]
        essts  = [string(round(Int, r.esst)) for r in sorted]
        dt = DataTable(
            ["Param", "Mean", "Std", "2.5%", "97.5%", "Rhat", "ESSb", "ESSt"],
            Vector{AbstractVector}([labels, means, stds, q025s, q975s, rhats, essbs, essts]);
            block=Block(; title="Diagnostics$(n_info)$(warn_info)"),
            selected=max(1, m.diagnostics_scroll),
        )
    elseif inner_w >= 55
        means  = [_fmt(r.mean, 2) for r in sorted]
        stds   = [_fmt(r.std, 2) for r in sorted]
        dt = DataTable(
            ["Param", "Mean", "Std", "Rhat", "ESS"],
            Vector{AbstractVector}([labels, means, stds, rhats, essbs]);
            block=Block(; title="Diagnostics$(n_info)$(warn_info)"),
            selected=max(1, m.diagnostics_scroll),
        )
    else
        dt = DataTable(
            ["Param", "Rhat", "ESS"],
            Vector{AbstractVector}([labels, rhats, essbs]);
            block=Block(; title="Diagnostics$(n_info)$(warn_info)"),
            selected=max(1, m.diagnostics_scroll),
        )
    end

    render(dt, area, buf)
end

"""Build diagnostic label with warning flag."""
function _diag_label(r)
    flag = r.warn ? "⚠ " : "  "
    flag * _short_label(r.label)
end

"""Shorten parameter labels for tight columns."""
function _short_label(s::String)
    length(s) <= 10 && return s
    return first(s, 9) * "…"
end

"""Format a float compactly."""
function _fmt(x::Float64, digits::Int)
    if abs(x) >= 1000
        string(round(Int, x))
    elseif abs(x) >= 100
        string(round(x; digits=max(0, digits - 2)))
    elseif abs(x) >= 10
        string(round(x; digits=max(0, digits - 1)))
    else
        string(round(x; digits=digits))
    end
end
