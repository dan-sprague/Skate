## ── TUI Extension Tests ─────────────────────────────────────────────────────
## Run with: julia --project -e 'include("test/test_tui.jl")'

using Test
using PhaseSkate
using PhaseSkate: ModelLogDensity, _RowSummary,
                  _col_diagnostics, _elem_label, _RHAT_WARN, _ESS_WARN
using Tachikoma
using Tachikoma: TestBackend, find_text, row_text, Rect, render

# Load the extension module
const TUI = Base.get_extension(PhaseSkate, :PhaseSkateTUIExt)

# ── Helper: simple 2D normal model ──────────────────────────────────────────
function _make_test_model()
    dim = 2
    ℓ(q) = -0.5 * sum(q .^ 2)
    constrain(q) = (mu = q[1], sigma = q[2])
    ModelLogDensity(dim, ℓ, constrain)
end

@testset "PhaseSkateTUIExt" begin

    @testset "IDEModel defaults" begin
        m = TUI.IDEModel()
        @test m.focus == TUI.FocusEditor
        @test m.active_tab == TUI.SamplingTab
        @test m.quit == false
        @test m.chains === nothing
        @test m.model === nothing
        @test m.repl_widget === nothing
        @test m.compile_error === nothing
        # Editor should have default model source
        @test !isempty(Tachikoma.value(m.editor))
    end

    @testset "DashboardModel backward compat" begin
        m = TUI.DashboardModel()
        @test m isa TUI.IDEModel
    end

    @testset "ChainProgress struct" begin
        cp = TUI.ChainProgress(1, :warmup, 50, 100, 0.1, 5, 0.8, 0, UInt64(1000000))
        @test cp.chain_id == 1
        @test cp.phase == :warmup
        @test cp.iteration == 50
    end

    @testset "Callback integration" begin
        model = _make_test_model()
        ch = Channel{Any}(1024)
        chains = redirect_stdout(devnull) do
            sample(model, 50; warmup=25, chains=1, callback=ch)
        end
        close(ch)
        events = collect(ch)
        @test length(events) > 0
        @test events[1].phase == :warmup
        @test events[end].phase == :done
        @test chains !== nothing
    end

    @testset "Diagnostics computation" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 200; warmup=100, chains=2)
        end
        rows = TUI._compute_diagnostics(chains)
        @test length(rows) == 2
        @test rows[1].label == "mu"
        @test rows[2].label == "sigma"
        @test all(r -> isfinite(r.rhat), rows)
    end

    @testset "Param labels" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 100; warmup=50, chains=1)
        end
        labels = TUI._build_param_labels(chains)
        @test labels == ["mu", "sigma"]
    end

    @testset "Sampling tab renders" begin
        tb = TestBackend(80, 24)
        m = TUI.IDEModel()
        area = Rect(1, 1, 80, 20)
        TUI._render_sampling_tab(m, area, tb.buf)
    end

    @testset "Diagnostics tab renders with chains" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 200; warmup=100, chains=2)
        end
        m = TUI.IDEModel(; chains=chains)
        tb = TestBackend(100, 30)
        area = Rect(1, 1, 100, 28)
        TUI._render_diagnostics_tab(m, area, tb.buf)
        found = false
        for y in 1:30
            txt = row_text(tb, y)
            if occursin("mu", txt) || occursin("sigma", txt)
                found = true
                break
            end
        end
        @test found
    end

    @testset "Trace tab renders with chains" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 200; warmup=100, chains=2)
        end
        m = TUI.IDEModel(; chains=chains)
        tb = TestBackend(100, 30)
        area = Rect(1, 1, 100, 28)
        TUI._render_trace_tab(m, area, tb.buf)
    end

    @testset "Density tab renders with chains" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 200; warmup=100, chains=2)
        end
        m = TUI.IDEModel(; chains=chains)
        tb = TestBackend(100, 30)
        area = Rect(1, 1, 100, 28)
        TUI._render_density_tab(m, area, tb.buf)
    end

    @testset "View-only model state" begin
        model = _make_test_model()
        chains = redirect_stdout(devnull) do
            sample(model, 200; warmup=100, chains=2)
        end
        m = TUI.IDEModel(; chains=chains)
        @test m.sampling_done == true
        @test m.focus == TUI.FocusOutput
        @test m.active_tab == TUI.DiagnosticsTab
        @test length(m.param_labels) == 2
        @test length(m.diagnostics_rows) == 2
    end

    @testset "Format time" begin
        @test TUI._format_time(0.5) == "0.5s"
        @test TUI._format_time(65.0) == "1m 5s"
        @test TUI._format_time(3700.0) == "1h 2m"
    end

    @testset "Theme" begin
        @test TUI.PHASESKATE_THEME.name == "phaseskate"
        @test TUI.PHASESKATE_THEME.bg == Tachikoma.Color256(233)
    end

    @testset "Entry point methods exist" begin
        @test length(methods(PhaseSkate.app)) >= 1
        @test length(methods(PhaseSkate.dashboard)) >= 2
    end

    @testset "Aggregate progress" begin
        m = TUI.IDEModel()
        @test TUI._aggregate_progress(m) == 0

        push!(m.chain_progress, TUI.ChainProgress(1, :sampling, 50, 100, 0.1, 5, 0.8, 0, UInt64(0)))
        m.n_chains = 1
        @test TUI._aggregate_progress(m) == 75  # 50% warmup done (assumed) + 25% sampling
    end

    @testset "Binary compilation fields" begin
        m = TUI.IDEModel()
        @test m.compiling_binary == false
        @test m.binary_output == String[]
        @test m.binary_path === nothing
    end

    @testset "Compile binary script generation" begin
        # _do_compile_binary will fail since juliac isn't available,
        # but we can test _compile_binary! sets state correctly
        m = TUI.IDEModel()
        @test m.compiling_binary == false

        # Test double-compile guard
        m.compiling_binary = true
        TUI._compile_binary!(m)
        @test m.compile_error == "Already compiling — wait for it to finish"
    end
end
