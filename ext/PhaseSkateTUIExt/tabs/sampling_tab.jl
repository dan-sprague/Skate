## ── Sampling Tab: live progress gauges + stats ──────────────────────────────

function _render_sampling_tab(m::IDEModel, area::Rect, buf::Buffer)
    if isempty(m.chain_progress) && !m.sampling_done
        # Pre-sampling state
        msg = if m.benchmark_done
            est = _format_time(m.estimated_seconds)
            "Estimated sampling time: ~$est  (gradient: $(round(m.grad_μs; digits=1)) μs/eval)"
        else
            "Compiling gradient and benchmarking..."
        end
        render(Paragraph(msg; wrap=word_wrap,
               block=Block(; title="Sampling")), area, buf)
        return
    end

    if m.sampling_done && m.chains !== nothing
        render(Paragraph("Sampling complete — switch to Diagnostics (2) or Traces (3)";
               wrap=word_wrap, block=Block(; title="Sampling ✓")), area, buf)
        return
    end

    # Layout: one gauge row per chain + stats table below
    n = m.n_chains
    gauge_height = min(n * 2 + 1, area.height ÷ 2)
    layout = Layout(Vertical, [Fixed(gauge_height), Fill()])
    areas = split_layout(layout, area)

    # ── Gauges ──
    gauge_layout = Layout(Vertical, [Fixed(2) for _ in 1:n])
    gauge_areas = split_layout(gauge_layout, areas[1])

    for i in 1:min(n, length(gauge_areas))
        cp = _latest_progress(m, i)
        if cp === nothing
            label = "Chain $i: waiting..."
            ratio = 0.0
        else
            phase_label = cp.phase == :warmup ? "warmup" : "sampling"
            pct = cp.total > 0 ? round(Int, 100 * cp.iteration / cp.total) : 0
            label = "Chain $i [$phase_label] $(cp.iteration)/$(cp.total) ($pct%)"
            ratio = cp.total > 0 ? clamp(cp.iteration / cp.total, 0.0, 1.0) : 0.0
            if cp.phase == :warmup
                ratio *= 0.5  # warmup is first half
            else
                ratio = 0.5 + ratio * 0.5  # sampling is second half
            end
        end
        gauge = Gauge(ratio; label=label,
                      block=nothing,
                      filled_style=tstyle(:primary))
        render(gauge, gauge_areas[i], buf)
    end

    # ── Stats table ──
    if !isempty(m.chain_progress)
        headers = ["Chain", "Phase", "ε", "Depth", "Accept", "Divergent", "Elapsed"]
        cols = [String[] for _ in 1:7]
        for i in 1:n
            cp = _latest_progress(m, i)
            cp === nothing && continue
            push!(cols[1], string(cp.chain_id))
            push!(cols[2], string(cp.phase))
            push!(cols[3], string(round(cp.step_size; sigdigits=4)))
            push!(cols[4], string(cp.tree_depth))
            push!(cols[5], string(round(cp.accept_rate; digits=3)))
            push!(cols[6], string(cp.n_divergent))
            push!(cols[7], _format_time(cp.elapsed_ns / 1e9))
        end
        if !isempty(cols[1])
            dt = DataTable(headers, Vector{AbstractVector}(cols);
                           block=Block(; title="Chain Statistics"))
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
