## ── Native Binary Compilation Pipeline ──────────────────────────────────────
#
# Compiles a @skate model into a standalone native binary via juliac (Julia 1.12+).
# The binary accepts a JSON data file and runs the full NUTS sampler.
#
# Three fixes are applied automatically:
#   1. JLL library paths baked via LocalPreferences.toml
#   2. GPU symbol stubs linked in
#   3. Base.invokelatest wraps all Enzyme calls

"""
    compile(source::String; output="phaseskate_model", chains=4,
            num_samples=1000, warmup=1000, max_depth=10, ad=:auto) → String

Compile a `@skate` model definition into a standalone native binary.

The resulting binary accepts a JSON data file and runs the full NUTS sampler:

    ./phaseskate_model data.json [--chains=4] [--samples=1000] [--warmup=1000]
                                 [--output=FILE] [--seed=INT]

Returns the path to the compiled binary.

Requires Julia 1.12+ (for juliac) and patched Enzyme at `~/.julia/dev/Enzyme`.
"""
function compile(source::String;
                 output::String = "phaseskate_model",
                 chains::Int = 4,
                 num_samples::Int = 1000,
                 warmup::Int = 1000,
                 max_depth::Int = 10,
                 ad::Symbol = :auto)

    # Validate
    model_name = _parse_model_name(source)
    libs = _find_jll_libs()
    threads = max(chains + 1, Threads.nthreads())

    # Find patched Enzyme with Preferences-based JLL library path fix
    enzyme_dev = ""
    for candidate in [joinpath(homedir(), ".julia", "dev", "Enzyme"),
                      joinpath(homedir(), "Documents", "Enzyme.jl")]
        if isdir(candidate) && isfile(joinpath(candidate, "src", "api.jl"))
            enzyme_dev = candidate
            break
        end
    end
    if isempty(enzyme_dev)
        error("Patched Enzyme not found.\n" *
              "Looked in: ~/.julia/dev/Enzyme, ~/Documents/Enzyme.jl\n" *
              "The Preferences-based JLL library path fix is required for compilation.")
    end

    # Create build directory
    build_dir = mktempdir(; cleanup=false)
    printstyled("◆  Build directory: "; color=:cyan, bold=true)
    println(build_dir)

    # Generate all build files
    _write_project(build_dir, libs, enzyme_dev)
    gpu_stubs_path = _write_gpu_stubs(build_dir)
    cc_wrapper = _write_cc_wrapper(build_dir, gpu_stubs_path)
    _write_entry_point(build_dir, source, model_name,
                       chains=chains, num_samples=num_samples,
                       warmup=warmup, max_depth=max_depth, ad=ad)

    # Run juliac
    juliac_jl = _find_juliac()
    exe_path = _run_juliac(build_dir, juliac_jl, cc_wrapper, output, threads)

    printstyled("✓  Binary compiled: "; color=:green, bold=true)
    println(exe_path)
    printstyled("   Size: "; color=:cyan)
    sz = filesize(exe_path)
    println(sz > 1_000_000 ? "$(round(sz / 1_000_000; digits=1)) MB" : "$sz bytes")

    return exe_path
end


## ── Helper: parse model name ────────────────────────────────────────────────

function _parse_model_name(source::String)
    m = match(r"@skate\s+(\w+)", source)
    m === nothing && error("Could not find @skate ModelName in source")
    return String(m[1])
end


## ── Helper: parse @constants field names and types ──────────────────────────

"""
    _parse_constants(source) → Vector{Tuple{String,String}}

Extract (name, type_string) pairs from the `@constants` block in a `@skate` source.
"""
function _parse_constants(source::String)
    m = match(r"@constants\s+begin\s*(.*?)\s*end"s, source)
    m === nothing && error("Could not find @constants block in source")
    block = m[1]
    fields = Tuple{String,String}[]
    for line in split(block, '\n')
        line = strip(line)
        isempty(line) && continue
        fm = match(r"^(\w+)\s*::\s*(.+)$", line)
        fm === nothing && continue
        push!(fields, (String(fm[1]), strip(String(fm[2]))))
    end
    return fields
end


## ── Helper: generate statically-typed _load_data function ───────────────────

"""
    _gen_load_data(source, model_name) → String

Generate a `_load_data` function body with statically-typed field parsing
and a direct constructor call (no splatting).
"""
function _gen_load_data(source::String, model_name::String)
    data_type = "$(model_name)Data"
    const_fields = _parse_constants(source)

    parse_lines = String[]
    arg_names = String[]
    for (fname, ftype) in const_fields
        vname = "_f_$fname"
        push!(arg_names, vname)
        if ftype == "Int"
            push!(parse_lines, """
        haskey(fields, "$fname") || error("Missing field \\\"$fname\\\" in JSON")
        $vname = parse(Int, fields["$fname"]::String)""")
        elseif ftype == "Float64"
            push!(parse_lines, """
        haskey(fields, "$fname") || error("Missing field \\\"$fname\\\" in JSON")
        $vname = parse(Float64, fields["$fname"]::String)""")
        elseif ftype == "Vector{Float64}"
            push!(parse_lines, """
        haskey(fields, "$fname") || error("Missing field \\\"$fname\\\" in JSON")
        $vname = Float64[parse(Float64, x) for x in fields["$fname"]::Vector{String}]""")
        elseif ftype == "Vector{Int}"
            push!(parse_lines, """
        haskey(fields, "$fname") || error("Missing field \\\"$fname\\\" in JSON")
        $vname = Int[parse(Int, x) for x in fields["$fname"]::Vector{String}]""")
        elseif ftype == "Matrix{Float64}"
            push!(parse_lines, """
        haskey(fields, "$fname") || error("Missing field \\\"$fname\\\" in JSON")
        $vname = begin
            rows_raw = fields["$fname"]::Vector{String}
            nrows = length(rows_raw)
            first_row_vals, _ = _parse_array(rows_raw[1] * "]", 1)
            ncols = length(first_row_vals)
            mat = Matrix{Float64}(undef, nrows, ncols)
            for r in 1:nrows
                row_str = lstrip(rows_raw[r], ['[', ' '])
                row_str = rstrip(row_str, [']', ' '])
                parts = split(row_str, ',')
                for c in 1:ncols
                    mat[r, c] = parse(Float64, strip(parts[c]))
                end
            end
            mat
        end""")
        else
            error("Unsupported @constants field type: $ftype for field $fname")
        end
    end

    constructor_args = join(arg_names, ", ")
    parsed = join(parse_lines, "\n")

    return """function _load_data(path::String)
        raw = read(path, String)
        fields = _parse_json_fields(raw)
$parsed
        return $data_type($constructor_args)
    end"""
end


## ── Helper: find JLL libraries ──────────────────────────────────────────────

function _find_jll_libs()
    artifacts_dir = joinpath(homedir(), ".julia", "artifacts")
    isdir(artifacts_dir) || error("Julia artifacts directory not found at $artifacts_dir")

    targets = Dict(
        "libEnzyme-18"     => "",
        "libEnzymeBCLoad-18" => "",
        "libLLVMExtra-18"  => "",
    )

    # Walk artifacts looking for dylibs
    for hash_dir in readdir(artifacts_dir; join=true)
        isdir(hash_dir) || continue
        for (root, dirs, files) in walkdir(hash_dir)
            for f in files
                for (name, _) in targets
                    if startswith(f, name) && (endswith(f, ".dylib") || endswith(f, ".so"))
                        targets[name] = joinpath(root, f)
                    end
                end
            end
        end
    end

    for (name, path) in targets
        isempty(path) && error("Could not find $name in $artifacts_dir")
    end

    return targets
end


## ── Helper: write GPU stubs ─────────────────────────────────────────────────

function _write_gpu_stubs(build_dir::String)
    stubs_c = joinpath(build_dir, "gpu_stubs.c")
    stubs_o = joinpath(build_dir, "gpu_stubs.o")

    write(stubs_c, """
    #include <stddef.h>
    #include <stdio.h>
    #include <stdlib.h>

    void gpu_signal_exception(void) {
        fprintf(stderr, "GPU signal exception (stub)\\n");
        abort();
    }

    void gpu_report_oom(size_t sz) {
        fprintf(stderr, "GPU OOM: %zu bytes (stub)\\n", sz);
        abort();
    }

    void *gpu_malloc(size_t sz) {
        fprintf(stderr, "GPU malloc: %zu bytes (stub)\\n", sz);
        return NULL;
    }

    void deferred_codegen(void *val) {
        fprintf(stderr, "deferred_codegen (stub)\\n");
    }
    """)

    # Compile to object file
    run(`cc -c -o $stubs_o $stubs_c`)
    return stubs_o
end


## ── Helper: find juliac.jl ───────────────────────────────────────────────────

function _find_juliac()
    for candidate in [joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"),
                      joinpath(Sys.BINDIR, "..", "share", "julia", "juliac.jl"),
                      joinpath(Sys.BINDIR, "juliac.jl")]
        isfile(candidate) && return candidate
    end
    error("Cannot find juliac.jl — is this Julia 1.12+?")
end


## ── Helper: write CC wrapper to inject GPU stubs ────────────────────────────

function _write_cc_wrapper(build_dir::String, gpu_stubs_path::String)
    wrapper_path = joinpath(build_dir, "cc_wrapper.sh")
    write(wrapper_path, """
    #!/bin/sh
    exec cc "\$@" $gpu_stubs_path
    """)
    run(`chmod +x $wrapper_path`)
    return wrapper_path
end


## ── Helper: write Project.toml + LocalPreferences.toml ──────────────────────

function _write_project(build_dir::String, libs::Dict{String,String}, enzyme_path::String)
    project_toml = joinpath(build_dir, "Project.toml")
    phaseskate_path = dirname(dirname(@__FILE__))  # src/../ = project root

    write(project_toml, """
    [deps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"
    LLVMExtra_jll = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
    Preferences = "21216c6a-2e73-6563-6e65-726566657250"
    PhaseSkate = "b2669007-1cce-4dba-a404-52a49cb6e0db"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

    [sources.PhaseSkate]
    path = $(repr(phaseskate_path))

    [sources.Enzyme]
    path = $(repr(enzyme_path))
    """)

    # LocalPreferences.toml — bake JLL library paths
    prefs_toml = joinpath(build_dir, "LocalPreferences.toml")

    enzyme_lib = libs["libEnzyme-18"]
    bcload_lib = libs["libEnzymeBCLoad-18"]
    llvmextra_lib = libs["libLLVMExtra-18"]

    write(prefs_toml, """
    [Enzyme_jll]
    libEnzyme_path = $(repr(enzyme_lib))
    libEnzymeBCLoad_path = $(repr(bcload_lib))

    [LLVMExtra_jll]
    libLLVMExtra_path = $(repr(llvmextra_lib))
    """)
end


## ── Helper: write entry point ───────────────────────────────────────────────

function _write_entry_point(build_dir::String, source::String, model_name::String;
                            chains::Int, num_samples::Int, warmup::Int,
                            max_depth::Int, ad::Symbol)
    data_type = "$(model_name)Data"
    entry_path = joinpath(build_dir, "model_main.jl")

    write(entry_path, """
    # Auto-generated by PhaseSkate.compile()
    # Do not edit — regenerate by calling compile() again.

    using PhaseSkate
    using PhaseSkate: ParamInfo
    using Enzyme

    # ── Model definition (baked in) ──
    $source

    # ── Minimal JSON parser ──
    # Hand-rolled to avoid JSON.jl dependency. Supports the types used by @skate @constants.

    function _skip_ws(s::String, i::Int)
        while i <= ncodeunits(s) && (s[i] == ' ' || s[i] == '\\t' || s[i] == '\\n' || s[i] == '\\r')
            i += 1
        end
        return i
    end

    function _parse_string(s::String, i::Int)
        # i points at opening "
        i += 1  # skip "
        j = i
        while j <= ncodeunits(s) && s[j] != '"'
            j += 1
        end
        return String(s[i:j-1]), j + 1
    end

    function _parse_number(s::String, i::Int)
        j = i
        while j <= ncodeunits(s) && (s[j] == '-' || s[j] == '+' || s[j] == '.' ||
              s[j] == 'e' || s[j] == 'E' || (s[j] >= '0' && s[j] <= '9'))
            j += 1
        end
        return String(s[i:j-1]), j
    end

    function _parse_array(s::String, i::Int)
        # i points at [
        i += 1
        vals = String[]
        i = _skip_ws(s, i)
        if i <= ncodeunits(s) && s[i] == ']'
            return vals, i + 1
        end
        while true
            i = _skip_ws(s, i)
            if s[i] == '['
                # nested array — skip it as a raw string for matrix parsing
                depth = 1
                j = i + 1
                while j <= ncodeunits(s) && depth > 0
                    s[j] == '[' && (depth += 1)
                    s[j] == ']' && (depth -= 1)
                    j += 1
                end
                push!(vals, String(s[i:j-1]))
                i = j
            else
                val, i = _parse_number(s, i)
                push!(vals, val)
            end
            i = _skip_ws(s, i)
            if i > ncodeunits(s) || s[i] == ']'
                return vals, i + 1
            end
            i += 1  # skip comma
        end
    end

    function _parse_json_fields(s::String)
        fields = Dict{String, Any}()
        i = _skip_ws(s, 1)
        i <= ncodeunits(s) && s[i] == '{' || error("Expected {")
        i += 1
        while true
            i = _skip_ws(s, i)
            i > ncodeunits(s) && break
            s[i] == '}' && break
            s[i] == ',' && (i += 1; continue)

            # key
            i = _skip_ws(s, i)
            key, i = _parse_string(s, i)
            i = _skip_ws(s, i)
            s[i] == ':' || error("Expected :")
            i += 1
            i = _skip_ws(s, i)

            # value
            if s[i] == '['
                arr, i = _parse_array(s, i)
                fields[key] = arr
            else
                val, i = _parse_number(s, i)
                fields[key] = val
            end
        end
        return fields
    end

    $(_gen_load_data(source, model_name))

    # ── Argument parsing ──

    function _parse_args(args::Vector{String})
        data_file = ""
        n_chains = $chains
        n_samples = $num_samples
        n_warmup = $warmup
        depth = $max_depth
        output_file = ""
        seed = nothing

        for arg in args
            if startswith(arg, "--chains=")
                n_chains = parse(Int, arg[10:end])
            elseif startswith(arg, "--samples=")
                n_samples = parse(Int, arg[11:end])
            elseif startswith(arg, "--warmup=")
                n_warmup = parse(Int, arg[10:end])
            elseif startswith(arg, "--max-depth=")
                depth = parse(Int, arg[13:end])
            elseif startswith(arg, "--output=")
                output_file = arg[10:end]
            elseif startswith(arg, "--seed=")
                seed = parse(Int, arg[8:end])
            elseif startswith(arg, "-")
                Core.println("Unknown option: " * arg)
                return nothing
            else
                data_file = arg
            end
        end

        if isempty(data_file)
            return nothing
        end

        return (data_file=data_file, chains=n_chains, samples=n_samples,
                warmup=n_warmup, max_depth=depth, output=output_file, seed=seed)
    end

    # ── Write CSV output ──

    function _write_csv(io::IO, chains_result)
        diag = PhaseSkate.diagnostics(chains_result)
        ns = size(chains_result.data, 1)
        nc = size(chains_result.data, 3)

        # Header
        print(io, "chain,draw")
        for name in diag.names
            print(io, ",", name)
        end
        println(io)

        # Data rows
        ncols = size(chains_result.data, 2)
        for c in 1:nc
            for i in 1:ns
                print(io, c, ",", i)
                for j in 1:ncols
                    print(io, ",", chains_result.data[i, j, c])
                end
                println(io)
            end
        end
    end

    # ── Main entry point ──

    function (@main)(ARGS::Vector{String})::Cint
        opts = _parse_args(ARGS)
        if opts === nothing
            Core.println("Usage: " * Base.PROGRAM_FILE * " data.json [options]")
            Core.println()
            Core.println("Options:")
            Core.println("  --chains=N      Number of chains (default: $chains)")
            Core.println("  --samples=N     Samples per chain (default: $num_samples)")
            Core.println("  --warmup=N      Warmup iterations (default: $warmup)")
            Core.println("  --max-depth=N   Max tree depth (default: $max_depth)")
            Core.println("  --output=FILE   Write CSV samples to FILE")
            Core.println("  --seed=INT      Random seed")
            return Cint(1)
        end

        # Load data
        isfile(opts.data_file) || (Core.println("File not found: " * opts.data_file); return Cint(1))
        data = _load_data(opts.data_file)
        model = make(data)

        Core.println("Model: $model_name  dim=" * string(model.dim))
        Core.println("Chains: " * string(opts.chains) * "  Samples: " * string(opts.samples) *
                     "  Warmup: " * string(opts.warmup))

        # Sample — invokelatest defers Enzyme gradient compilation to runtime
        chains_result = Base.invokelatest(
            PhaseSkate.sample, model, opts.samples;
            warmup=opts.warmup, chains=opts.chains,
            max_depth=opts.max_depth, ad=$(QuoteNode(ad)),
            seed=opts.seed
        )

        # Write CSV if output file specified
        if !isempty(opts.output)
            open(opts.output, "w") do io
                _write_csv(io, chains_result)
            end
            Core.println("Samples written to: " * opts.output)
        end

        return Cint(0)
    end
    """)

    return entry_path
end


## ── Helper: run juliac ──────────────────────────────────────────────────────

function _run_juliac(build_dir::String, juliac_jl::String, cc_wrapper::String,
                     output::String, threads::Int)
    entry_point = joinpath(build_dir, "model_main.jl")
    exe_path = joinpath(build_dir, output)

    julia_exe = joinpath(Sys.BINDIR, "julia")

    printstyled("◆  Running juliac "; color=:cyan, bold=true)
    printstyled("(this takes several minutes)\n"; color=:light_black)

    cmd = `$julia_exe -t $threads --project=$build_dir $juliac_jl --output-exe $exe_path $entry_point`
    cmd = addenv(cmd, "JULIA_CC" => cc_wrapper)
    printstyled("   "; color=:light_black)
    printstyled(string(cmd); color=:light_black)
    println()

    proc = run(pipeline(cmd; stdout=stdout, stderr=stderr); wait=true)

    isfile(exe_path) || error("juliac completed but binary not found at $exe_path")
    return exe_path
end
