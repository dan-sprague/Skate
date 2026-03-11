## ── PhaseSkate GitHub Dark Theme ─────────────────────────────────────────────
## Inspired by GitHub's dark default color scheme.
##
## Color256 codes chosen to approximate:
##   bg:           #0d1117 → 233 (very dark blue-gray)
##   border:       #30363d → 237 (dark border gray)
##   border_focus: #58a6ff → 75  (GitHub blue)
##   text:         #e6edf3 → 253 (primary text)
##   text_dim:     #8b949e → 245 (comments, secondary)
##   text_bright:  #ffffff → 255 (bright highlights)
##   primary:      #58a6ff → 75  (links, keywords blue)
##   secondary:    #d2a8ff → 183 (purple — types, decorators)
##   accent:       #7ee787 → 114 (green — strings, success)
##   success:      #7ee787 → 114 (green)
##   warning:      #d29922 → 178 (gold/amber)
##   error:        #f85149 → 203 (red)
##   title:        #58a6ff → 75  (blue titles)

const PHASESKATE_THEME = Tachikoma.Theme(
    "phaseskate",
    Tachikoma.Color256(233),   # bg
    Tachikoma.Color256(237),   # border
    Tachikoma.Color256(75),    # border_focus
    Tachikoma.Color256(253),   # text
    Tachikoma.Color256(245),   # text_dim
    Tachikoma.Color256(255),   # text_bright
    Tachikoma.Color256(75),    # primary (blue)
    Tachikoma.Color256(183),   # secondary (purple)
    Tachikoma.Color256(114),   # accent (green)
    Tachikoma.Color256(114),   # success (green)
    Tachikoma.Color256(178),   # warning (gold)
    Tachikoma.Color256(203),   # error (red)
    Tachikoma.Color256(75),    # title (blue)
)

function _apply_theme!()
    Tachikoma.set_theme!(PHASESKATE_THEME)
end
