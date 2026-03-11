## ── REPL panel: embedded Julia REPL via Tachikoma's REPLWidget ───────────────

function _ensure_repl_widget!(m::IDEModel)
    if m.repl_widget === nothing
        m.repl_widget = REPLWidget(; focused=true)
    end
    m.repl_widget
end

function _cleanup_repl!(m::IDEModel)
    if m.repl_widget !== nothing
        close!(m.repl_widget)
        m.repl_widget = nothing
    end
end
