## ── Sampling Tab: live progress gauges + stats ──────────────────────────────

function _render_sampling_tab(m::IDEModel, area::Rect, buf::Buffer)
    if isempty(m.chain_progress) && !m.sampling_done
        # Pre-sampling state
        msg = if m.compiling
            "Compiling model + gradient (Enzyme JIT)..."
        elseif m.model === nothing
            "Define data in the REPL, then press F5 to compile and run"
        elseif m.benchmark_done
            est = _format_time(m.estimated_seconds)
            "~$est  (grad: $(round(m.grad_μs; digits=1)) μs)"
        else
            "Benchmarking gradient..."
        end
        render(Paragraph(msg; wrap=word_wrap,
               block=Block(; title="Sampling")), area, buf)
        return
    end

    # Always show gauges + stats (even after done)
    title = m.sampling_done ? "Sampling ✓" : "Sampling"

    n = m.n_chains
    gauge_height = min(n + 1, max(area.height ÷ 2, n + 1))
    layout = Layout(Vertical, [Fixed(gauge_height), Fill()])
    areas = split_layout(layout, area)

    # ── Gauges (1 row each) ──
    gauge_layout = Layout(Vertical, [Fixed(1) for _ in 1:n])
    gauge_areas = split_layout(gauge_layout, areas[1])

    for i in 1:min(n, length(gauge_areas))
        cp = _latest_progress(m, i)
        if cp === nothing
            label = "Ch$i: waiting..."
            ratio = 0.0
        else
            if cp.phase == :done || m.sampling_done
                label = "Ch$i ✓ $(cp.total)"
                ratio = 1.0
            else
                ph = cp.phase == :warmup ? "w" : "s"
                pct = cp.total > 0 ? round(Int, 100 * cp.iteration / cp.total) : 0
                label = "Ch$i [$ph] $(cp.iteration)/$(cp.total)"
                ratio = cp.total > 0 ? clamp(cp.iteration / cp.total, 0.0, 1.0) : 0.0
                if cp.phase == :warmup
                    ratio *= 0.5
                else
                    ratio = 0.5 + ratio * 0.5
                end
            end
        end
        style = m.sampling_done ? tstyle(:success) : tstyle(:primary)
        gauge = Gauge(ratio; label=label, block=nothing, filled_style=style)
        render(gauge, gauge_areas[i], buf)
    end

    # ── Compact stats table with CUMULATIVE averages ──
    if !isempty(m.chain_progress)
        headers = ["Ch", "ε", "Avg Dp", "Avg Acc", "Div"]
        cols = [String[] for _ in 1:5]
        for i in 1:n
            cp = _latest_progress(m, i)
            cp === nothing && continue
            push!(cols[1], string(cp.chain_id))
            push!(cols[2], string(round(cp.step_size; sigdigits=3)))
            # Cumulative averages
            if i <= length(m.chain_cum_n) && m.chain_cum_n[i] > 0
                avg_depth = m.chain_cum_depth[i] / m.chain_cum_n[i]
                avg_accept = m.chain_cum_accept[i] / m.chain_cum_n[i]
                push!(cols[3], string(round(avg_depth; digits=1)))
                push!(cols[4], string(round(avg_accept; digits=2)))
            else
                push!(cols[3], string(cp.tree_depth))
                push!(cols[4], string(round(cp.accept_rate; digits=2)))
            end
            push!(cols[5], string(cp.n_divergent))
        end
        if !isempty(cols[1])
            dt = DataTable(headers, Vector{AbstractVector}(cols);
                           block=Block(; title=title))
            render(dt, areas[2], buf)
        end
    end
end

function _latest_progress(m::IDEModel, chain_id::Int)
    for i in length(m.chain_progress):-1:1
        m.chain_progress[i].chain_id == chain_id && return m.chain_progress[i]
    end
    nothing
end

function _format_time(seconds::Float64)
    seconds < 0.001 && return "0s"
    if seconds < 60
        return "$(round(seconds; digits=1))s"
    elseif seconds < 3600
        m = floor(Int, seconds / 60)
        s = round(Int, seconds - m * 60)
        return "$(m)m $(s)s"
    else
        h = floor(Int, seconds / 3600)
        m = round(Int, (seconds - h * 3600) / 60)
        return "$(h)h $(m)m"
    end
end
