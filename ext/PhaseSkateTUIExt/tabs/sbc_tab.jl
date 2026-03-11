## ── SBC Tab: rank histograms with pass/fail ──────────────────────────────────

function _render_sbc_tab(m::IDEModel, area::Rect, buf::Buffer)
    if m.sbc_result === nothing
        render(Paragraph("No SBC result available — pass sbc=result to dashboard()";
               block=Block(; title="SBC")), area, buf)
        return
    end

    sbc = m.sbc_result

    layout = Layout(Horizontal, [Fixed(min(30, area.width ÷ 3)), Fill()])
    areas = split_layout(layout, area)

    # Parameter list with pass/fail indicators
    items = String[]
    for (j, name) in enumerate(sbc.param_names)
        status = sbc.p_values[j] < 0.01 ? "✗" : "✓"
        pval = round(sbc.p_values[j]; digits=4)
        push!(items, "$status $name (p=$pval)")
    end

    sbc_idx = clamp(m.selected_param_idx, 1, length(sbc.param_names))
    list = SelectableList(items; selected=sbc_idx,
                          block=Block(; title="Parameters"))
    render(list, areas[1], buf)

    # Rank histogram for selected parameter
    ranks = @view sbc.ranks[:, sbc_idx]
    M = sbc.M
    nbins = 20
    counts = zeros(Int, nbins)
    for r in ranks
        b = clamp(floor(Int, r / (M + 1) * nbins) + 1, 1, nbins)
        counts[b] += 1
    end
    max_count = maximum(counts; init=1)

    bars = BarEntry[]
    for i in 1:nbins
        push!(bars, BarEntry(string(i), Float64(counts[i]); style=tstyle(:primary)))
    end

    pval = sbc.p_values[sbc_idx]
    status = pval < 0.01 ? "FAIL" : "ok"
    bc = BarChart(bars; max_val=Float64(max_count), show_values=false,
                  block=Block(; title="SBC Ranks: $(sbc.param_names[sbc_idx]) [$status, p=$(round(pval; digits=4))]"))
    render(bc, areas[2], buf)
end
