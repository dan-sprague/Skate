## ── App lifecycle: view, update!, should_quit, task_queue, init! ─────────────

Tachikoma.should_quit(m::IDEModel) = m.quit
Tachikoma.task_queue(m::IDEModel) = m.tq

function Tachikoma.init!(m::IDEModel, terminal)
    _apply_theme!()

    # If we have a model and no chains, start sampling
    if m.model !== nothing && m.chains === nothing
        _start_sampling!(m)
    end
end

function Tachikoma.cleanup!(m::IDEModel)
    _cleanup_repl!(m)
end

## ── View ─────────────────────────────────────────────────────────────────────

function Tachikoma.view(m::IDEModel, f::Tachikoma.Frame)
    area = f.area
    buf = f.buffer

    # Help overlay
    if m.show_help
        _render_help(area, buf)
        return
    end

    # Main layout: content + status bar (1 row)
    outer = Layout(Vertical, [Fill(), Fixed(1)])
    outer_areas = split_layout(outer, area)

    # Content: left (editor) | right (output + repl)
    editor_w = max(30, area.width * 2 ÷ 5)
    content = Layout(Horizontal, [Fixed(editor_w), Fill()])
    content_areas = split_layout(content, outer_areas[1])

    # ── Left: Editor ──
    _render_editor_panel(m, content_areas[1], buf)

    # ── Right: output tabs (top) + REPL (bottom) ──
    repl_h = max(6, content_areas[2].height ÷ 4)
    right = Layout(Vertical, [Fill(), Fixed(repl_h)])
    right_areas = split_layout(right, content_areas[2])

    # Output tab bar + content
    _render_output_panel(m, right_areas[1], buf)

    # REPL
    _render_repl_panel(m, right_areas[2], buf)

    # ── Status bar ──
    _render_status_bar(m, outer_areas[2], buf)
end

## ── Panel renderers ──────────────────────────────────────────────────────────

function _render_editor_panel(m::IDEModel, area::Rect, buf::Buffer)
    is_focused = m.focus == FocusEditor
    title = is_focused ? "▸ Model Editor" : "  Model Editor"
    border_style = is_focused ? tstyle(:border_focus) : tstyle(:border)

    # Editor area with border
    block = Block(; title=title, border_style=border_style)
    inner = Tachikoma.inner_area(block, area)
    Tachikoma.render(block, area, buf)

    # Compile error indicator
    if m.compile_error !== nothing
        err_h = min(3, inner.height ÷ 3)
        editor_area = Rect(inner.x, inner.y, inner.width, inner.height - err_h)
        err_area = Rect(inner.x, inner.y + inner.height - err_h, inner.width, err_h)
        render(m.editor, editor_area, buf)
        err_text = "✗ " * first(m.compile_error, 200)
        render(Paragraph([Span(err_text, tstyle(:error))]; wrap=word_wrap), err_area, buf)
    else
        render(m.editor, inner, buf)
    end
end

function _render_output_panel(m::IDEModel, area::Rect, buf::Buffer)
    is_focused = m.focus == FocusOutput

    # Tab bar (1 row) + content
    layout = Layout(Vertical, [Fixed(1), Fill()])
    areas = split_layout(layout, area)

    tab_bar = TabBar(OUTPUT_TAB_LABELS; active=Int(m.active_tab))
    render(tab_bar, areas[1], buf)

    # Tab content
    tab = m.active_tab
    if tab == SamplingTab
        _render_sampling_tab(m, areas[2], buf)
    elseif tab == DiagnosticsTab
        _render_diagnostics_tab(m, areas[2], buf)
    elseif tab == TraceTab
        _render_trace_tab(m, areas[2], buf)
    elseif tab == DensityTab
        _render_density_tab(m, areas[2], buf)
    elseif tab == SBCTab
        _render_sbc_tab(m, areas[2], buf)
    end
end

function _render_repl_panel(m::IDEModel, area::Rect, buf::Buffer)
    is_focused = m.focus == FocusRepl
    title = is_focused ? "▸ REPL" : "  REPL"
    border_style = is_focused ? tstyle(:border_focus) : tstyle(:border)

    block = Block(; title=title, border_style=border_style)
    inner = Tachikoma.inner_area(block, area)
    Tachikoma.render(block, area, buf)

    rw = _ensure_repl_widget!(m)
    drain!(rw)
    render(rw, inner, buf)
end

function _render_status_bar(m::IDEModel, area::Rect, buf::Buffer)
    left_spans = Span[]

    # Focus indicator
    focus_label = if m.focus == FocusEditor
        "EDITOR"
    elseif m.focus == FocusOutput
        OUTPUT_TAB_LABELS[Int(m.active_tab)]
    else
        "REPL"
    end
    push!(left_spans, Span(" $focus_label ", tstyle(:primary, bold=true)))

    # Sampling status
    if m.sampling_done
        push!(left_spans, Span("  ✓ done ", tstyle(:success)))
    elseif !isempty(m.chain_progress)
        # Show aggregate progress
        total_pct = _aggregate_progress(m)
        push!(left_spans, Span("  ⟳ $(total_pct)% ", tstyle(:warning)))
    end

    if m.compiling_binary
        push!(left_spans, Span("  ⟳ compiling binary... ", tstyle(:secondary)))
    elseif m.binary_path !== nothing
        push!(left_spans, Span("  ✓ binary ready ", tstyle(:success)))
    end

    if m.compile_error !== nothing
        push!(left_spans, Span("  ✗ error ", tstyle(:error)))
    end

    right_spans = [Span(" Ctrl+E:editor  Ctrl+D:repl  1-5:tabs  F5:run  F7:build  ?:help  Ctrl+Q:quit ", tstyle(:text_dim))]
    sb = StatusBar(; left=left_spans, right=right_spans)
    render(sb, area, buf)
end

## ── Update: key events ───────────────────────────────────────────────────────

function Tachikoma.update!(m::IDEModel, evt::Tachikoma.KeyEvent)
    evt.action == Tachikoma.key_release && return

    # Help overlay dismissal
    if m.show_help
        m.show_help = false
        return
    end

    # ── Global keybindings (always active) ──

    # Ctrl+Q: quit
    if evt.key == :ctrl && evt.char == 'q'
        m.quit = true
        return
    end

    # Ctrl+E: focus editor
    if evt.key == :ctrl && evt.char == 'e'
        m.focus = FocusEditor
        return
    end

    # Ctrl+D: focus REPL
    if evt.key == :ctrl && evt.char == 'd'
        m.focus = FocusRepl
        return
    end

    # F5: compile & run
    if evt.key == :f5
        _compile_and_run!(m)
        return
    end

    # F7: compile to binary
    if evt.key == :f7
        _compile_binary!(m)
        return
    end

    # ? for help (only when not in editor/repl)
    if m.focus == FocusOutput && evt.key == :char && evt.char == '?'
        m.show_help = true
        return
    end

    # ── Panel-specific dispatch ──

    if m.focus == FocusEditor
        _editor_handle_key!(m, evt)
    elseif m.focus == FocusRepl
        _repl_focused_handle_key!(m, evt)
    else  # FocusOutput
        _output_handle_key!(m, evt)
    end
end

function _editor_handle_key!(m::IDEModel, evt)
    # Escape in editor: switch focus to output
    if evt.key == :escape
        m.focus = FocusOutput
        return
    end

    # Let the CodeEditor handle everything else
    handle_key!(m.editor, evt)

    # Check for :w command → compile
    cmd = Tachikoma.pending_command!(m.editor)
    if !isempty(cmd)
        if cmd == "w" || cmd == "wq"
            _compile_and_run!(m)
        end
        if cmd == "q" || cmd == "wq"
            m.quit = true
        end
    end
end

function _repl_focused_handle_key!(m::IDEModel, evt)
    # Escape: switch focus to output
    if evt.key == :escape
        m.focus = FocusOutput
        return
    end

    rw = _ensure_repl_widget!(m)
    handle_key!(rw, evt)
end

function _output_handle_key!(m::IDEModel, evt)
    if evt.key == :char
        c = evt.char
        if c in '1':'5'
            m.active_tab = OutputTab(Int(c - '0'))
        elseif c == 'j' || c == 'J'
            _next_param!(m)
        elseif c == 'k' || c == 'K'
            _prev_param!(m)
        elseif c == 'q' || c == 'Q'
            m.quit = true
        end
    elseif evt.key == :escape
        m.quit = true
    elseif evt.key == :tab
        m.active_tab = OutputTab(mod1(Int(m.active_tab) + 1, N_OUTPUT_TABS))
    elseif evt.key == :down
        _next_param!(m)
    elseif evt.key == :up
        _prev_param!(m)
    end
end

## ── Update: task events ──────────────────────────────────────────────────────

function Tachikoma.update!(m::IDEModel, evt::Tachikoma.TaskEvent)
    if evt.id == :sampling_done
        if evt.value isa Exception
            m.compile_error = "Sampling failed: $(sprint(showerror, evt.value))"
        else
            m.chains = evt.value
            m.sampling_done = true
            m.param_labels = _build_param_labels(m.chains)
            m.diagnostics_rows = _compute_diagnostics(m.chains)
            m.selected_param_idx = 1
            m.active_tab = DiagnosticsTab
            m.focus = FocusOutput
        end
    elseif evt.id == :benchmark
        if !(evt.value isa Exception)
            m.grad_μs, m.leapfrog_μs, m.estimated_seconds = evt.value
            m.benchmark_done = true
        end
    elseif evt.id == :compile
        if evt.value isa Exception
            m.compile_error = sprint(showerror, evt.value)
        else
            m.model = evt.value
            m.compile_error = nothing
            # Auto-start sampling
            _start_sampling!(m)
            m.active_tab = SamplingTab
            m.focus = FocusOutput
        end
    elseif evt.id == :compile_binary
        m.compiling_binary = false
        if evt.value isa Exception
            m.compile_error = "Binary compilation failed: $(sprint(showerror, evt.value))"
            push!(m.binary_output, "✗ Compilation failed")
        else
            m.binary_path = evt.value
            m.compile_error = nothing
            push!(m.binary_output, "✓ Binary written to: $(evt.value)")
        end
    elseif evt.id == :compile_binary_line
        if !(evt.value isa Exception)
            push!(m.binary_output, string(evt.value))
        end
    end
end

# Default: drain progress events
function Tachikoma.update!(m::IDEModel, evt)
    _drain_progress!(m)
end

## ── Helpers ──────────────────────────────────────────────────────────────────

function _drain_progress!(m::IDEModel)
    while isready(m.sampling_channel)
        nt = take!(m.sampling_channel)
        cp = _to_chain_progress(nt)
        idx = findfirst(p -> p.chain_id == cp.chain_id, m.chain_progress)
        if idx === nothing
            push!(m.chain_progress, cp)
        else
            m.chain_progress[idx] = cp
        end
    end
end

function _next_param!(m::IDEModel)
    n = length(m.param_labels)
    n > 0 && (m.selected_param_idx = mod1(m.selected_param_idx + 1, n))
end

function _prev_param!(m::IDEModel)
    n = length(m.param_labels)
    n > 0 && (m.selected_param_idx = mod1(m.selected_param_idx - 1, n))
end

function _aggregate_progress(m::IDEModel)
    isempty(m.chain_progress) && return 0
    total = 0.0
    for cp in m.chain_progress
        ratio = cp.total > 0 ? cp.iteration / cp.total : 0.0
        if cp.phase == :warmup
            total += ratio * 0.5
        elseif cp.phase == :done
            total += 1.0
        else
            total += 0.5 + ratio * 0.5
        end
    end
    round(Int, 100 * total / m.n_chains)
end

function _render_help(area::Rect, buf::Buffer)
    help_text = """
    PhaseSkate IDE — Keyboard Shortcuts

    Navigation
      Ctrl+E      Focus model editor
      Ctrl+D      Focus REPL
      1-5         Switch output tabs
      Tab         Next output tab
      Escape      Switch focus / quit from output

    Editor
      F5          Compile model & run sampling
      F7          Build standalone binary (juliac)
      :w          Compile model (vim command)
      :q          Quit
      Ctrl+F      Search
      i / Esc     Insert / Normal mode (vim)

    Output Tabs
      j / ↓       Next parameter
      k / ↑       Previous parameter
      ?           This help

    System
      Ctrl+Q      Quit
    """
    render(Paragraph(help_text; wrap=word_wrap,
           block=Block(; title="Help")), area, buf)
end

## ── Compile & Run ────────────────────────────────────────────────────────────

function _compile_and_run!(m::IDEModel)
    source = Tachikoma.value(m.editor)
    m.compile_error = nothing

    # Reset sampling state
    m.chain_progress = ChainProgress[]
    m.sampling_done = false
    m.benchmark_done = false
    m.chains = nothing
    m.sampling_channel = Channel{ChainProgress}(1024)

    # Compile in background
    Tachikoma.spawn_task!(m.tq, :compile) do
        _compile_model(source)
    end
end

function _compile_model(source::String)
    # Eval the @skate block + make() in a temporary module
    mod = Module(:PhaseSkateCompile)

    # Import PhaseSkate into the compile module
    Core.eval(mod, :(using PhaseSkate))

    # Parse and eval the source
    exprs = Meta.parseall(source)
    Core.eval(mod, exprs)

    # Look for the generated Data type and make() pattern
    # The @skate macro generates: <Name>Data struct + extends log_prob
    # User needs to provide data, so we return the model constructor info
    # For now, just eval and check if there's a make-able data type
    #
    # Try to find the Data type and construct with dummy data if possible
    # This is a placeholder — real IDE will have a data browser
    nothing  # model will be set by the user via REPL
end

## ── Sampling orchestration ───────────────────────────────────────────────────

function _start_sampling!(m::IDEModel)
    model = m.model
    model === nothing && return

    # TUI needs its own thread on top of the chain threads
    required = m.n_chains + 1
    if Threads.nthreads() < required
        m.compile_error = "Need $required threads for $(m.n_chains) chains + TUI. " *
                          "Start Julia with: julia -t $required"
        return
    end

    channel = m.sampling_channel
    num_samples = m.num_samples
    warmup = m.warmup
    n_chains = m.n_chains
    kwargs = m.sampling_kwargs

    # Benchmark in background
    Tachikoma.spawn_task!(m.tq, :benchmark) do
        _benchmark_model(model; ad=get(kwargs, :ad, m.ad_mode))
    end

    # Sampling in background — pass the channel directly as callback.
    # The sampler puts NamedTuples, the TUI drains and converts in _drain_progress!.
    # Thread requirement: n+1 (n chains via Threads.@spawn + main thread for TUI).
    Tachikoma.spawn_task!(m.tq, :sampling_done) do
        result = PhaseSkate.sample(model, num_samples;
            warmup=warmup,
            chains=n_chains,
            max_depth=m.max_depth,
            ad=m.ad_mode,
            callback=channel,
            kwargs...)
        result
    end
end

## ── Compile to Binary ────────────────────────────────────────────────────────

function _compile_binary!(m::IDEModel)
    if m.compiling_binary
        m.compile_error = "Already compiling — wait for it to finish"
        return
    end

    source = Tachikoma.value(m.editor)
    m.compile_error = nothing
    m.compiling_binary = true
    m.binary_output = String["⟳ Generating compilation script..."]
    m.binary_path = nothing

    n_chains = m.n_chains
    num_samples = m.num_samples
    warmup_iters = m.warmup
    max_depth = m.max_depth
    ad_mode = m.ad_mode
    n_threads = n_chains + 1

    Tachikoma.spawn_task!(m.tq, :compile_binary) do
        _do_compile_binary(source, n_chains, num_samples, warmup_iters,
                           max_depth, ad_mode, n_threads)
    end
end

function _do_compile_binary(source::String, n_chains::Int, num_samples::Int,
                            warmup::Int, max_depth::Int, ad::Symbol, n_threads::Int)
    # Create a temp directory for the build
    build_dir = mktempdir(; cleanup=false)
    script_path = joinpath(build_dir, "compile_model.jl")
    exe_name = "phaseskate_model"
    output_path = joinpath(build_dir, exe_name)

    # Extract a model name from the source (look for @skate <Name>)
    model_name_match = match(r"@skate\s+(\w+)", source)
    model_name = model_name_match !== nothing ? model_name_match[1] : "Model"
    data_type = "$(model_name)Data"

    # Generate the compilation script
    script = """
    # Auto-generated by PhaseSkate IDE
    # Build with: juliac --output-exe $exe_name -t $n_threads $script_path

    using PhaseSkate
    using Tachikoma

    # ── Model definition ──
    $source

    # ── Enzyme warmup: trigger gradient JIT during compilation ──
    # Build a minimal instance to force-compile the gradient pipeline.
    # The user's load_data() will provide real data at runtime.
    let
        # Create the smallest valid model instance for compilation
        T = $data_type
        fields = fieldnames(T)
        ftypes = fieldtypes(T)
        dummy_args = []
        for (f, ft) in zip(fields, ftypes)
            if ft == Int
                push!(dummy_args, 2)
            elseif ft == Float64
                push!(dummy_args, 0.0)
            elseif ft <: AbstractVector{Float64}
                push!(dummy_args, [0.0, 0.0])
            elseif ft <: AbstractVector{Int}
                push!(dummy_args, [1, 1])
            elseif ft <: AbstractMatrix{Float64}
                push!(dummy_args, [0.0 0.0; 0.0 0.0])
            else
                push!(dummy_args, zero(ft))
            end
        end
        d = T(dummy_args...)
        m = make(d)
        # Run 1 sample to JIT-compile the full Enzyme + NUTS pipeline
        redirect_stdout(devnull) do
            sample(m, 1; warmup=1, chains=1, ad=$(QuoteNode(ad)))
        end
    end

    # ── Entry point ──
    function main()
        # Users should edit this function to load their data.
        # The model definition above is baked into the binary.
        error("Edit main() in $script_path to load your data, then recompile.\\n" *
              "Example:\\n" *
              "  d = $(data_type)(N=length(y), y=y)\\n" *
              "  m = make(d)\\n" *
              "  app(; model=m, num_samples=$num_samples, warmup=$warmup, " *
              "chains_count=$n_chains, max_depth=$max_depth)")
    end
    """

    write(script_path, script)

    # Try to run juliac
    juliac_cmd = `juliac --output-exe $output_path -t $n_threads $script_path`

    # Check if juliac is available
    juliac_path = Sys.which("juliac")
    if juliac_path === nothing
        # juliac is typically at the same location as julia
        julia_path = joinpath(Sys.BINDIR, "juliac")
        if !isfile(julia_path)
            error("juliac not found. It ships with Julia 1.12+. " *
                  "Compilation script written to: $script_path")
        end
        juliac_cmd = `$julia_path --output-exe $output_path -t $n_threads $script_path`
    end

    # Run juliac, capturing output
    proc = open(juliac_cmd, "r")
    output_text = read(proc, String)
    success = proc.exitcode == 0

    if !success
        error("juliac failed (exit $(proc.exitcode)).\n$output_text\n" *
              "Script at: $script_path")
    end

    output_path
end

## ── Benchmark ────────────────────────────────────────────────────────────────

function _benchmark_model(model; ad=:auto)
    ∇!, grad_μs = PhaseSkate._make_grad(model; ad)

    dim = model.dim
    q = randn(dim)
    p = randn(dim)
    g = zeros(dim)
    inv_metric = ones(dim)
    ∇!(g, model, q)

    t0 = time_ns()
    n_lf = 10
    for _ in 1:n_lf
        PhaseSkate.leapfrog!(q, p, g, model, 0.01, inv_metric, ∇!)
    end
    leapfrog_μs = (time_ns() - t0) / n_lf / 1e3

    avg_lf_per_sample = 64.0
    estimated_seconds = avg_lf_per_sample * leapfrog_μs / 1e6

    (grad_μs, leapfrog_μs, estimated_seconds)
end
