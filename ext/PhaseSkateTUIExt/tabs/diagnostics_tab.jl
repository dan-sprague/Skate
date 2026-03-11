## ── Diagnostics Tab: DataTable with R̂/ESS/CI ─────────────────────────────────

function _render_diagnostics_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        render(Paragraph("No chains available yet — run sampling first";
               block=Block(; title="Diagnostics")), area, buf)
        return
    end

    rows = m.diagnostics_rows
    isempty(rows) && return

    # Build columns
    labels = [r.label for r in rows]
    means  = [round(r.mean; digits=4) for r in rows]
    stds   = [round(r.std; digits=4) for r in rows]
    q025s  = [round(r.q025; digits=4) for r in rows]
    q975s  = [round(r.q975; digits=4) for r in rows]
    rhats  = [round(r.rhat; digits=4) for r in rows]
    essbs  = [round(r.essb; digits=1) for r in rows]
    essts  = [round(r.esst; digits=1) for r in rows]

    dt = DataTable(
        ["Parameter", "Mean", "Std", "2.5%", "97.5%", "R̂", "ESS_bulk", "ESS_tail"],
        Vector{AbstractVector}([labels, means, stds, q025s, q975s, rhats, essbs, essts]);
        block=Block(; title="Diagnostics"),
    )

    render(dt, area, buf)
end
