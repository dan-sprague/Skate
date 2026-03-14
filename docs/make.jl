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

# Copy static assets that VitePress doesn't bundle (e.g. .cast files)
let build_assets = joinpath(@__DIR__, "build", "1", "assets")
    if isdir(build_assets)
        for f in readdir(joinpath(@__DIR__, "src", "assets"))
            src = joinpath(@__DIR__, "src", "assets", f)
            dst = joinpath(build_assets, f)
            isfile(src) && !isfile(dst) && cp(src, dst)
        end
    end
end

DocumenterVitepress.deploydocs(;
    repo = "github.com/dan-sprague/PhaseSkate",
    devbranch = "main",
    push_preview = true,
)
