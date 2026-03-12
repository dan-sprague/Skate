using Documenter
using DocumenterVitepress
using PhaseSkate

makedocs(
    sitename = "PhaseSkate",
    modules = [PhaseSkate],
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/dan-sprague/PhaseSkate",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Model DSL" => "dsl.md",
        "Samplers" => "samplers.md",
        "API Reference" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/dan-sprague/PhaseSkate",
    devbranch = "main",
    push_preview = true,
)
