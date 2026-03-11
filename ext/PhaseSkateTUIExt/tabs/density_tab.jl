## ── Density Tab: posterior histograms as BarChart ────────────────────────────

function _render_density_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        render(Paragraph("No chains available yet"; block=Block(; title="Density")), area, buf)
        return
    end

    layout = Layout(Horizontal, [Fixed(min(25, area.width ÷ 4)), Fill()])
    areas = split_layout(layout, area)

    # Parameter selector
    list = SelectableList(m.param_labels; selected=m.selected_param_idx,
                          block=Block(; title="Parameters"))
    render(list, areas[1], buf)

    # Histogram
    idx = clamp(m.selected_param_idx, 1, length(m.param_labels))
    col = _param_col(m.chains, idx)
    pooled = vec(m.chains.data[:, col, :])

    nbins = 40
    lo, hi = extrema(pooled)
    if lo == hi
        lo -= 1.0; hi += 1.0
    end
    bin_width = (hi - lo) / nbins
    counts = zeros(Int, nbins)
    for v in pooled
        b = clamp(floor(Int, (v - lo) / bin_width) + 1, 1, nbins)
        counts[b] += 1
    end
    max_count = maximum(counts)

    bars = BarEntry[]
    for i in 1:nbins
        center = lo + (i - 0.5) * bin_width
        label = string(round(center; digits=2))
        push!(bars, BarEntry(label, Float64(counts[i]); style=tstyle(:primary)))
    end

    bc = BarChart(bars; max_val=Float64(max_count), show_values=false,
                  block=Block(; title="Density: $(m.param_labels[idx])"))
    render(bc, areas[2], buf)
end
