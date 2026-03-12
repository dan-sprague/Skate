## ── Trace Tab: line chart per parameter, one series per chain ────────────────

function _render_trace_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        msg = if !isempty(m.chain_progress)
            "Warming up..."
        else
            "Waiting for sampling"
        end
        render(Paragraph(msg; block=Block(; title="Traces")), area, buf)
        return
    end

    idx = clamp(m.selected_param_idx, 1, length(m.param_labels))
    col = _param_col(m.chains, idx)
    ns = size(m.chains.data, 1)
    nc = size(m.chains.data, 3)

    # Julia brand colors: blue, purple, green, red
    chain_colors = [tstyle(:primary), tstyle(:secondary), tstyle(:accent), tstyle(:error),
                    tstyle(:warning), tstyle(:text_bright)]

    series = DataSeries[]
    for c in 1:nc
        ys = Float64.(m.chains.data[:, col, c])
        style = chain_colors[mod1(c, length(chain_colors))]
        push!(series, DataSeries(ys; label="Ch$c", style=style))
    end

    label = m.param_labels[idx]
    n_info = m.sampling_done ? "" : " (n=$(ns))"
    chart = Chart(series; block=Block(; title="Trace: $(label)$(n_info)"),
                  x_label="Iteration", y_label="Value", show_legend=nc <= 4)
    render(chart, area, buf)
end

"""Map a flat parameter index to its column in chains.data."""
function _param_col(chains::PhaseSkate.Chains, flat_idx::Int)
    i = 0
    for p in chains.params
        for col in p.cols
            i += 1
            i == flat_idx && return col
        end
    end
    return 1
end
