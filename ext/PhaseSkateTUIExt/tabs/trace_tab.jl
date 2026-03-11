## ── Trace Tab: line chart per parameter, one series per chain ────────────────

function _render_trace_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.chains === nothing
        render(Paragraph("No chains available yet"; block=Block(; title="Traces")), area, buf)
        return
    end

    layout = Layout(Horizontal, [Fixed(min(25, area.width ÷ 4)), Fill()])
    areas = split_layout(layout, area)

    # Parameter selector
    list = SelectableList(m.param_labels; selected=m.selected_param_idx,
                          block=Block(; title="Parameters"))
    render(list, areas[1], buf)

    # Trace plot
    idx = clamp(m.selected_param_idx, 1, length(m.param_labels))
    col = _param_col(m.chains, idx)
    nc = size(m.chains.data, 3)

    chain_colors = [tstyle(:primary), tstyle(:secondary), tstyle(:accent), tstyle(:success),
                    tstyle(:warning), tstyle(:error)]

    series = DataSeries[]
    for c in 1:nc
        ys = Float64.(m.chains.data[:, col, c])
        style = chain_colors[mod1(c, length(chain_colors))]
        push!(series, DataSeries(ys; label="Chain $c", style=style))
    end

    chart = Chart(series; block=Block(; title="Trace: $(m.param_labels[idx])"),
                  x_label="Iteration", y_label="Value", show_legend=nc <= 6)
    render(chart, areas[2], buf)
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
