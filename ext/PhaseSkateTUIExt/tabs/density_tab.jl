## ── Density Tab: posterior histograms as BarChart ────────────────────────────

function _render_density_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        msg = if !isempty(m.chain_progress)
            "Warming up..."
        else
            "Waiting for sampling"
        end
        render(Paragraph(msg; block=Block(; title="Density")), area, buf)
        return
    end

    idx = clamp(m.selected_param_idx, 1, length(m.param_labels))
    col = _param_col(m.chains, idx)
    pooled = vec(m.chains.data[:, col, :])

    nbins = min(30, max(8, area.width ÷ 3))
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

    ns = size(m.chains.data, 1)
    n_info = m.sampling_done ? "" : " (n=$(ns))"
    bc = BarChart(bars; max_val=Float64(max_count), show_values=false,
                  block=Block(; title="Density: $(m.param_labels[idx])$(n_info)"))
    render(bc, area, buf)
end
