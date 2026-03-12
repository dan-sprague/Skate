## ── Debug logging ────────────────────────────────────────────────────────────

const _IDE_LOG_PATH = "/tmp/phaseskate_ide.log"

function _ide_log(msg::String)
    open(_IDE_LOG_PATH, "a") do io
        println(io, round(time(); digits=3), " ", msg)
    end
end

"""Check if a symbol is defined in Main (works from precompiled extensions)."""
function _main_has(sym::Symbol)
    try
        Core.eval(Main, sym)
        true
    catch
        false
    end
end

"""Get a value from Main by name."""
function _main_get(sym::Symbol)
    Core.eval(Main, sym)
end

## ── App lifecycle: view, update!, should_quit, task_queue, init! ─────────────

Tachikoma.should_quit(m::IDEModel) = m.quit
Tachikoma.task_queue(m::IDEModel) = m.tq

function Tachikoma.init!(m::IDEModel, terminal::Terminal)
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

function Tachikoma.has_pending_output(m::IDEModel)
    # Wake the event loop when the sampling channel has progress data to drain
    isready(m.sampling_channel)
end

function Tachikoma.view(m::IDEModel, f::Tachikoma.Frame)
    # Drain progress every frame (not just on events) for real-time updates
    _drain_progress!(m)

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

    # ── Left: Editor (top) + Data Environment (bottom) ──
    data_h = max(4, content_areas[1].height ÷ 5)
    left = Layout(Vertical, [Fill(), Fixed(data_h)])
    left_areas = split_layout(left, content_areas[1])
    _render_editor_panel(m, left_areas[1], buf)
    _render_data_environment(m, left_areas[2], buf)

    # ── Right: output tabs (top) + REPL (bottom) ──
    repl_h = max(6, content_areas[2].height ÷ 4)
    right = Layout(Vertical, [Fill(), Fixed(repl_h)])
    right_areas = split_layout(right, content_areas[2])

    # Output tab bar + content
    _render_output_panel(m, right_areas[1], buf)

    # REPL
    _render_repl_panel(m, right_areas[2], buf)

    # Cache layout areas for mouse hit-testing
    m._editor_area = content_areas[1]
    m._output_area = right_areas[1]
    m._repl_area = right_areas[2]

    # ── Status bar ──
    _render_status_bar(m, outer_areas[2], buf)
end

## ── Panel renderers ──────────────────────────────────────────────────────────

function _render_editor_panel(m::IDEModel, area::Rect, buf::Buffer)
    is_focused = m.focus == FocusEditor
    title = is_focused ? "▸ Model Editor" : "  Model Editor"
    border_style = is_focused ? tstyle(:border_focus) : tstyle(:border)

    block = Block(; title=title, border_style=border_style)
    inner = Tachikoma.inner_area(block, area)
    Tachikoma.render(block, area, buf)

    # Reserve 1 row for compile error if present
    has_error = m.compile_error !== nothing
    editor_h = has_error ? inner.height - 1 : inner.height

    # Sync focused state so cursor only shows when editor has focus
    m.editor.focused = is_focused

    editor_area = Rect(inner.x, inner.y, inner.width, editor_h)
    render(m.editor, editor_area, buf)
    _overlay_selection!(m, editor_area, buf)
    _overlay_constants!(m, editor_area, buf)

    if has_error
        err_y = inner.y + editor_h
        err_area = Rect(inner.x, err_y, inner.width, 1)
        err_text = "✗ " * first(m.compile_error, inner.width - 3)
        render(Paragraph([Span(err_text, tstyle(:error))]), err_area, buf)
    end
end

"""Render the data environment table (like RStudio's Environment pane)."""
function _render_data_environment(m::IDEModel, area::Rect, buf::Buffer)
    fields = _parse_constants_block(m.editor)

    if isempty(fields)
        render(Paragraph("No @constants declared";
               block=Block(; title="Data")), area, buf)
        return
    end

    names_col = String[]
    status_col = String[]
    type_col = String[]
    size_col = String[]

    for (name, type_str) in fields
        push!(names_col, name)
        status = _check_constant(name, type_str)
        sym = Symbol(name)
        if status == :ok
            val = _main_get(sym)
            push!(status_col, "●")
            push!(type_col, _short_type(val))
            push!(size_col, _short_size(val))
        elseif status == :wrong_type
            val = _main_get(sym)
            push!(status_col, "✗")
            push!(type_col, _short_type(val) * " ≠ " * type_str)
            push!(size_col, _short_size(val))
        else
            push!(status_col, "○")
            push!(type_col, isempty(type_str) ? "?" : _shorten_type_str(type_str))
            push!(size_col, "—")
        end
    end

    n_ok = count(==("●"), status_col)
    n_total = length(fields)
    title = n_ok == n_total ? "Data ✓ $n_ok/$n_total" : "Data $n_ok/$n_total"

    dt = DataTable(
        ["", "Name", "Type", "Value"],
        Vector{AbstractVector}([status_col, names_col, type_col, size_col]);
        block=Block(; title=title),
    )
    render(dt, area, buf)
end

"""Short type name for display."""
function _short_type(val)
    if val isa Float64
        "Float64"
    elseif val isa Int || val isa Int64
        "Int"
    elseif val isa AbstractVector
        T = eltype(val)
        "Vec{$(T == Float64 ? "F64" : T == Int64 ? "Int" : string(T))}"
    elseif val isa AbstractMatrix
        T = eltype(val)
        "Mat{$(T == Float64 ? "F64" : string(T))}"
    else
        string(typeof(val))
    end
end

"""Short value summary: scalar value, or array dimensions."""
function _short_size(val)
    if val isa Bool
        string(val)
    elseif val isa Integer
        string(val)
    elseif val isa AbstractFloat
        string(round(val; sigdigits=4))
    elseif val isa AbstractVector
        "$(length(val))-elem"
    elseif val isa AbstractArray
        join(size(val), "×")
    elseif val isa AbstractString
        "\"$(first(val, 12))$(length(val) > 12 ? "…" : "")\""
    else
        summary(val)
    end
end

"""Shorten a declared type string like 'Vector{Float64}' → 'Vec{F64}'."""
function _shorten_type_str(s::AbstractString)
    s = replace(s, "Vector{Float64}" => "Vec{F64}")
    s = replace(s, "Vector{Int64}" => "Vec{Int}")
    s = replace(s, "Vector{Int}" => "Vec{Int}")
    s = replace(s, "Matrix{Float64}" => "Mat{F64}")
    s = replace(s, "Float64" => "F64")
    s
end

"""Parse @constants block into [(name, type_str), ...] pairs."""
function _parse_constants_block(ce::CodeEditor)
    line_count = length(ce.lines)
    line_count == 0 && return Tuple{String,String}[]

    in_block = false
    fields = Tuple{String,String}[]
    for i in 1:line_count
        stripped = strip(String(ce.lines[i]))
        if !in_block && occursin(r"@constants\s+begin", stripped)
            in_block = true
        elseif in_block && stripped == "end"
            break
        elseif in_block
            cm = match(r"^(\w+)(?:::(.+))?$", stripped)
            if cm !== nothing
                name = cm.captures[1]
                type_str = cm.captures[2] !== nothing ? cm.captures[2] : ""
                push!(fields, (name, type_str))
            end
        end
    end
    fields
end

"""Check a constant: returns :ok, :missing, :wrong_type."""
function _check_constant(name::AbstractString, type_str::AbstractString)
    sym = Symbol(name)
    _main_has(sym) || return :missing
    isempty(type_str) && return :ok
    val = _main_get(sym)
    # Resolve the declared type and check
    expected = try
        Core.eval(Main, Meta.parse(type_str))
    catch
        return :ok  # can't resolve type, assume ok
    end
    val isa expected ? :ok : :wrong_type
end

function _render_output_panel(m::IDEModel, area::Rect, buf::Buffer)
    is_focused = m.focus == FocusOutput

    # SBC tab gets its own full-area view
    if m.active_tab == SBCTab
        layout = Layout(Vertical, [Fixed(1), Fill()])
        areas = split_layout(layout, area)
        tab_bar = TabBar(OUTPUT_TAB_LABELS; active=Int(m.active_tab))
        render(tab_bar, areas[1], buf)
        m._tab_bar_area = areas[1]
        _render_sbc_tab(m, areas[2], buf)
        return
    end

    # Always show 2×2 results grid (Sampling | Diagnostics / Traces | Density)
    _render_results_grid(m, area, buf)
end

function _render_results_grid(m::IDEModel, area::Rect, buf::Buffer)
    # Top row:    Sampling | Diagnostics
    # Bottom row: Params | Trace | Density  (one shared param selector)
    rows = Layout(Vertical, [Fill(), Fill()])
    row_areas = split_layout(rows, area)

    top_cols = Layout(Horizontal, [Fill(), Fill()])
    top_areas = split_layout(top_cols, row_areas[1])

    _render_sampling_tab(m, top_areas[1], buf)
    _render_diagnostics_tab(m, top_areas[2], buf)

    # Bottom row: shared param list + trace + density
    if m.chains !== nothing && !isempty(m.param_labels)
        list_w = min(15, max(10, row_areas[2].width ÷ 8))
        bot = Layout(Horizontal, [Fixed(list_w), Fill(), Fill()])
        bot_areas = split_layout(bot, row_areas[2])

        list = SelectableList(m.param_labels; selected=m.selected_param_idx,
                              block=Block(; title="Params"))
        render(list, bot_areas[1], buf)

        _render_trace_tab(m, bot_areas[2], buf)
        _render_density_tab(m, bot_areas[3], buf)
    else
        bot_cols = Layout(Horizontal, [Fill(), Fill()])
        bot_areas = split_layout(bot_cols, row_areas[2])
        _render_trace_tab(m, bot_areas[1], buf)
        _render_density_tab(m, bot_areas[2], buf)
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

    # Ctrl+E: focus editor (when not already in editor); end-of-line when in editor
    if evt.key == :ctrl && evt.char == 'e'
        if m.focus == FocusEditor
            ce = m.editor
            ce.cursor_col = length(ce.lines[ce.cursor_row])
        else
            m.focus = FocusEditor
        end
        return
    end

    # Ctrl+A: beginning of line (when in editor)
    if evt.key == :ctrl && evt.char == 'a'
        if m.focus == FocusEditor
            m.editor.cursor_col = 0
        end
        return
    end

    # Ctrl+C: copy selection to clipboard (when in editor with selection)
    if evt.key == :ctrl && evt.char == 'c'
        if m.focus == FocusEditor && m.sel_anchor !== nothing
            text = _get_selected_text(m)
            if !isempty(text)
                try
                    open(pipeline(`pbcopy`), "w") do io
                        write(io, text)
                    end
                catch; end
            end
            m.sel_anchor = nothing
            return
        end
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
    # Clear selection on any keypress
    m.sel_anchor = nothing

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
            _ide_log("sampling error: $(sprint(showerror, evt.value))")
            m.compile_error = "Sampling failed: $(sprint(showerror, evt.value))"
        else
            m.chains = evt.value
            m.sampling_done = true
            m.param_labels = _build_param_labels(m.chains)
            m.diagnostics_rows = _compute_diagnostics(m.chains)
            if m.selected_param_idx < 1 || m.selected_param_idx > length(m.param_labels)
                m.selected_param_idx = 1
            end
        end
    elseif evt.id == :benchmark
        if evt.value isa Exception
            _ide_log("benchmark error: $(sprint(showerror, evt.value))")
        else
            m.grad_μs, m.leapfrog_μs, m.estimated_seconds = evt.value
            m.benchmark_done = true
        end
    elseif evt.id == :compile
        if evt.value isa Exception
            _ide_log("compile error: $(sprint(showerror, evt.value))")
            m.compile_error = sprint(showerror, evt.value)
        else
            m.model = evt.value
            m.compile_error = nothing
            m.live_constrain = evt.value.constrain
            # Mark all @constants as defined (model compiled = data provided)
            _update_defined_constants!(m)
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

# Mouse events: click tabs, switch focus, editor selection
function Tachikoma.update!(m::IDEModel, evt::MouseEvent)
    evt.button == mouse_left || return

    if evt.action == mouse_press
        # Tab bar click
        if _tab_bar_hit(m, evt)
            return
        end

        # Focus panels based on click location
        if m._editor_area.width > 0 && Base.contains(m._editor_area, evt.x, evt.y)
            m.focus = FocusEditor
            _editor_click!(m, evt)
            # Start selection
            ce = m.editor
            m.sel_anchor = (ce.cursor_row, ce.cursor_col)
            m.sel_active = true
        elseif m._output_area.width > 0 && Base.contains(m._output_area, evt.x, evt.y)
            m.focus = FocusOutput
            m.sel_anchor = nothing
            m.sel_active = false
        elseif m._repl_area.width > 0 && Base.contains(m._repl_area, evt.x, evt.y)
            m.focus = FocusRepl
            m.sel_anchor = nothing
            m.sel_active = false
        end

    elseif evt.action == Tachikoma.mouse_drag && m.sel_active
        # Extend selection while dragging in editor
        if m._editor_area.width > 0 && Base.contains(m._editor_area, evt.x, evt.y)
            _editor_click!(m, evt)
        end

    elseif evt.action == Tachikoma.mouse_release
        # Keep selection visible but stop extending
        m.sel_active = false
    end
end

"""Position editor cursor from a mouse click."""
function _editor_click!(m::IDEModel, evt::MouseEvent)
    ce = m.editor
    area = m._editor_area

    # Inner area: border is 1 char on each side
    inner_x = area.x + 1
    inner_y = area.y + 1

    # Line number gutter width
    line_count = length(ce.lines)
    gw = ce.show_line_numbers ? ndigits(max(line_count, 1)) + 1 : 0
    code_x = inner_x + gw

    # Translate click to row/col
    click_row = (evt.y - inner_y) + ce.scroll_offset + 1
    click_col = (evt.x - code_x) + ce.h_scroll

    # Clamp to valid range
    click_row = clamp(click_row, 1, line_count)
    click_col = clamp(click_col, 0, length(ce.lines[click_row]))

    ce.cursor_row = click_row
    ce.cursor_col = click_col
end

"""Render selection highlight overlay on the editor buffer."""
function _overlay_selection!(m::IDEModel, editor_area::Rect, buf::Buffer)
    m.sel_anchor === nothing && return
    ce = m.editor

    anchor_row, anchor_col = m.sel_anchor
    cursor_row, cursor_col = ce.cursor_row, ce.cursor_col

    # Nothing selected if anchor == cursor
    (anchor_row == cursor_row && anchor_col == cursor_col) && return

    # Normalize: start before end
    if (anchor_row, anchor_col) > (cursor_row, cursor_col)
        r1, c1 = cursor_row, cursor_col
        r2, c2 = anchor_row, anchor_col
    else
        r1, c1 = anchor_row, anchor_col
        r2, c2 = cursor_row, cursor_col
    end

    line_count = length(ce.lines)
    gw = ce.show_line_numbers ? ndigits(max(line_count, 1)) + 1 : 0
    code_x = editor_area.x + gw

    sel_style = Tachikoma.Style(; bg=Tachikoma.Color256(24))  # dark blue highlight

    for row in r1:r2
        vi = row - ce.scroll_offset
        (vi < 1 || vi > editor_area.height) && continue

        line_len = length(ce.lines[row])
        col_start = row == r1 ? c1 : 0
        col_end = row == r2 ? c2 : line_len

        buf_y = editor_area.y + vi - 1
        for col in col_start:(col_end - 1)
            ci = col - ce.h_scroll
            ci < 0 && continue
            x = code_x + ci
            x >= editor_area.x + editor_area.width && break
            ch = col < line_len ? ce.lines[row][col + 1] : ' '
            set_char!(buf, x, buf_y, ch, sel_style)
        end
    end
end

"""Get the selected text from the editor, or empty string if no selection."""
function _get_selected_text(m::IDEModel)
    m.sel_anchor === nothing && return ""
    ce = m.editor

    anchor_row, anchor_col = m.sel_anchor
    cursor_row, cursor_col = ce.cursor_row, ce.cursor_col
    (anchor_row == cursor_row && anchor_col == cursor_col) && return ""

    if (anchor_row, anchor_col) > (cursor_row, cursor_col)
        r1, c1 = cursor_row, cursor_col
        r2, c2 = anchor_row, anchor_col
    else
        r1, c1 = anchor_row, anchor_col
        r2, c2 = cursor_row, cursor_col
    end

    lines = String[]
    for row in r1:r2
        line = String(ce.lines[row])
        cs = row == r1 ? c1 + 1 : 1
        ce_end = row == r2 ? c2 : length(line)
        push!(lines, line[cs:ce_end])
    end
    join(lines, "\n")
end

# Default: catch-all for unhandled events
function Tachikoma.update!(m::IDEModel, evt::Event)
end

## ── Helpers ──────────────────────────────────────────────────────────────────

function _drain_progress!(m::IDEModel)
    new_samples = 0
    while isready(m.sampling_channel)
        nt = take!(m.sampling_channel)
        cp = _to_chain_progress(nt)
        idx = findfirst(p -> p.chain_id == cp.chain_id, m.chain_progress)
        if idx === nothing
            push!(m.chain_progress, cp)
        else
            m.chain_progress[idx] = cp
        end

        # Update cumulative averages
        cid = cp.chain_id
        while length(m.chain_cum_accept) < cid
            push!(m.chain_cum_accept, 0.0)
            push!(m.chain_cum_depth, 0.0)
            push!(m.chain_cum_n, 0)
        end
        m.chain_cum_accept[cid] += cp.accept_rate
        m.chain_cum_depth[cid] += cp.tree_depth
        m.chain_cum_n[cid] += 1

        # Accumulate live samples (sampling phase only, has :q field)
        if m.live_view && cp.phase == :sampling && hasproperty(nt, :q)
            cid = cp.chain_id
            # Ensure we have enough chain slots
            while length(m.live_raw) < cid
                push!(m.live_raw, Vector{Float64}[])
            end
            push!(m.live_raw[cid], nt.q)
            new_samples += 1
        end
    end

    # Rebuild live Chains periodically (every ~20 new samples)
    if new_samples > 0
        m.live_update_counter += new_samples
        if m.live_update_counter >= 20 && m.live_constrain !== nothing
            m.live_update_counter = 0
            _rebuild_live_chains!(m)
        end
    end
end

"""Rebuild m.chains from accumulated live_raw samples for real-time views."""
function _rebuild_live_chains!(m::IDEModel)
    # Find minimum sample count across chains that have data
    active_chains = [i for i in 1:length(m.live_raw) if !isempty(m.live_raw[i])]
    isempty(active_chains) && return

    min_n = minimum(length(m.live_raw[i]) for i in active_chains)
    min_n < 2 && return  # need at least 2 samples

    # Build raw_chains matrices (dim × nsamples) per chain
    dim = length(m.live_raw[active_chains[1]][1])
    raw_chains = Matrix{Float64}[]
    for c in active_chains
        mat = Matrix{Float64}(undef, dim, min_n)
        for i in 1:min_n
            mat[:, i] .= m.live_raw[c][i]
        end
        push!(raw_chains, mat)
    end

    try
        # Wrap constrain with invokelatest for world-age safety —
        # the constrain function was compiled dynamically via Core.eval
        constrain_fn = m.live_constrain
        safe_constrain = q -> Base.invokelatest(constrain_fn, q)
        chains = PhaseSkate.Chains(raw_chains, safe_constrain)
        m.chains = chains
        m.param_labels = _build_param_labels(chains)
        m.diagnostics_rows = _compute_diagnostics(chains)
        if m.selected_param_idx < 1 || m.selected_param_idx > length(m.param_labels)
            m.selected_param_idx = 1
        end
    catch e
        _ide_log("live chains rebuild error: $(sprint(showerror, e))")
    end
end

function _next_param!(m::IDEModel)
    n = length(m.param_labels)
    if n > 0
        m.selected_param_idx = mod1(m.selected_param_idx + 1, n)
        m.diagnostics_scroll = m.selected_param_idx
    end
end

function _prev_param!(m::IDEModel)
    n = length(m.param_labels)
    if n > 0
        m.selected_param_idx = mod1(m.selected_param_idx - 1, n)
        m.diagnostics_scroll = m.selected_param_idx
    end
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

## ── Tab bar hit-testing ──────────────────────────────────────────────────────

function _tab_bar_hit(m::IDEModel, evt::MouseEvent)
    area = m._tab_bar_area
    area.width == 0 && return false
    # Must be in the tab bar row
    (evt.y < area.y || evt.y > area.y + area.height - 1) && return false

    # Walk tab labels to find which was clicked.
    # TabBar renders: each tab as ` Label ` (len+2), separated by ` │ ` (3 chars).
    x_cursor = area.x
    for (i, label) in enumerate(OUTPUT_TAB_LABELS)
        tab_width = length(label) + 2  # ` Label `
        if evt.x >= x_cursor && evt.x < x_cursor + tab_width
            m.active_tab = OutputTab(i)
            m.focus = FocusOutput
            return true
        end
        x_cursor += tab_width + 3  # ` │ ` separator
    end
    return false
end

function _overlay_constants!(m::IDEModel, inner::Rect, buf::Buffer)
    ce = m.editor
    line_count = length(ce.lines)
    line_count == 0 && return

    # Find @constants block
    block_start = 0
    block_end = 0
    in_block = false
    for i in 1:line_count
        stripped = strip(String(ce.lines[i]))
        if !in_block && occursin(r"@constants\s+begin", stripped)
            in_block = true
            block_start = i
        elseif in_block && stripped == "end"
            block_end = i
            break
        end
    end
    (block_start == 0 || block_end == 0) && return

    gw = ce.show_line_numbers ? ndigits(max(line_count, 1)) + 1 : 0
    code_x = inner.x + gw
    scroll = ce.scroll_offset
    h_scroll = ce.h_scroll

    for i in (block_start + 1):(block_end - 1)
        vi = i - scroll
        (vi < 1 || vi > inner.height) && continue

        line = ce.lines[i]
        line_str = String(line)
        stripped = strip(line_str)
        isempty(stripped) && continue

        cm = match(r"^(\w+)(?:::(.+))?$", stripped)
        cm === nothing && continue
        name = cm.captures[1]
        type_str = cm.captures[2] !== nothing ? cm.captures[2] : ""

        pos = findfirst(name, line_str)
        pos === nothing && continue
        char_start = first(pos)

        status = _check_constant(name, type_str)
        style = if status == :ok
            Tachikoma.Style(; fg=Tachikoma.Color256(114), bold=true)   # green
        elseif status == :wrong_type
            Tachikoma.Style(; fg=Tachikoma.Color256(203), bold=true)   # red
        else
            Tachikoma.Style(; fg=Tachikoma.Color256(245))              # dim
        end

        buf_y = inner.y + vi - 1
        for j in 0:(length(name) - 1)
            char_idx = char_start + j
            ci = char_idx - h_scroll
            ci < 1 && continue
            x = code_x + ci - 1
            x > inner.x + inner.width - 1 && break
            ch = char_idx <= length(line) ? line[char_idx] : ' '
            set_char!(buf, x, buf_y, ch, style)
        end
    end
end

function _update_defined_constants!(m::IDEModel)
    empty!(m.defined_constants)
    source = Tachikoma.value(m.editor)
    in_block = false
    for line in split(source, '\n')
        stripped = strip(line)
        if !in_block && occursin(r"@constants\s+begin", stripped)
            in_block = true
        elseif in_block && stripped == "end"
            break
        elseif in_block
            m_match = match(r"^(\w+)", stripped)
            m_match !== nothing && push!(m.defined_constants, m_match.captures[1])
        end
    end
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
    m.sampling_channel = Channel{Any}(1024)
    m.live_raw = Vector{Vector{Float64}}[]
    m.live_constrain = nothing
    m.live_update_counter = 0
    m.chain_cum_accept = Float64[]
    m.chain_cum_depth = Float64[]
    m.chain_cum_n = Int[]
    m.diagnostics_scroll = 0
    empty!(m.defined_constants)

    # Compile in background
    Tachikoma.spawn_task!(m.tq, :compile) do
        _compile_model(source)
    end
end

function _compile_model(source::String)
    # Parse the source to extract model name and constant field names
    model_name, const_names = _parse_skate_header(source)

    # Check that all constants are defined in Main with correct types
    for (name, type_str) in const_names
        sym = Symbol(name)
        _main_has(sym) || error("Constant '$name' is not defined. " *
            "Define it in the REPL first (e.g. $name = ...)")
        if !isempty(type_str)
            val = _main_get(sym)
            expected = try Core.eval(Main, Meta.parse(type_str)) catch; nothing end
            if expected !== nothing && !(val isa expected)
                error("Constant '$name' has type $(typeof(val)), expected $type_str")
            end
        end
    end

    # Eval the @skate block in a temporary module
    mod = Module(:PhaseSkateCompile)
    Core.eval(mod, :(using PhaseSkate))
    exprs = Meta.parseall(source)
    Core.eval(mod, exprs)

    # Build the model entirely inside Core.eval so closures live in the
    # current world age — Enzyme can't differentiate through invokelatest.
    data_struct_name = Symbol(string(model_name) * "Data")
    kw_args = [Expr(:kw, Symbol(n), _main_get(Symbol(n))) for (n, _) in const_names]
    build_expr = Expr(:block,
        Expr(:(=), :_data, Expr(:call, data_struct_name, kw_args...)),
        Expr(:call, :make, :_data),
    )
    model = Core.eval(mod, build_expr)
    model::PhaseSkate.ModelLogDensity
end

"""Parse `@skate ModelName begin` and `@constants begin ... end` from source.
Returns (model_name, [(name, type_str), ...])."""
function _parse_skate_header(source::String)
    # Extract model name
    m = match(r"@skate\s+(\w+)\s+begin", source)
    m === nothing && error("Could not find '@skate ModelName begin' in source")
    model_name = m.captures[1]

    # Extract @constants fields with types
    const_fields = Tuple{String,String}[]
    in_block = false
    for line in split(source, '\n')
        stripped = strip(line)
        if !in_block && occursin(r"@constants\s+begin", stripped)
            in_block = true
        elseif in_block && stripped == "end"
            break
        elseif in_block
            cm = match(r"^(\w+)(?:::(.+))?$", stripped)
            if cm !== nothing
                name = cm.captures[1]
                type_str = cm.captures[2] !== nothing ? cm.captures[2] : ""
                push!(const_fields, (name, type_str))
            end
        end
    end
    isempty(const_fields) && error("No fields found in @constants block")

    (model_name, const_fields)
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

    # Benchmark in background — invokelatest so dynamically-compiled model
    # closures are callable (world age safety)
    Tachikoma.spawn_task!(m.tq, :benchmark) do
        Base.invokelatest(_benchmark_model, model; ad=get(kwargs, :ad, m.ad_mode))
    end

    # Sampling in background — invokelatest for same reason.
    # The sampler puts NamedTuples, the TUI drains and converts in _drain_progress!.
    # Thread requirement: n+1 (n chains via Threads.@spawn + main thread for TUI).
    Tachikoma.spawn_task!(m.tq, :sampling_done) do
        result = Base.invokelatest(PhaseSkate.sample, model, num_samples;
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
    PhaseSkate.compile(source;
        chains=n_chains, num_samples=num_samples,
        warmup=warmup, max_depth=max_depth, ad=ad)
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
