# PosteriorDB validation tests for PhaseSkate.
# Run with: POSTERIORDB_TESTS=true julia --project -e 'using Pkg; Pkg.test()'

using Test
using PhaseSkate
using PosteriorDB
using Statistics

include("comparison.jl")
include("models.jl")

# ─── Reference draw loading ─────────────────────────────────────────────────

"""
    flatten_ref_draws(ref_data)

Flatten reference posterior draws into `Dict{String, Vector{Float64}}`.
Handles both per-chain `Vector{Dict}` and pre-flattened `Dict` formats.
"""
function flatten_ref_draws(ref_data)
    draws = Dict{String, Vector{Float64}}()

    if ref_data isa AbstractVector && !isempty(ref_data) && first(ref_data) isa AbstractDict
        # Per-chain format: Vector of Dicts
        for key in keys(first(ref_data))
            chain_draws = [Float64.(chain[key]) for chain in ref_data]
            draws[string(key)] = reduce(vcat, chain_draws)
        end
    elseif ref_data isa AbstractDict
        # Already flattened
        for (key, val) in ref_data
            if val isa AbstractVector{<:AbstractVector}
                # Nested: parameter → Vector of chain vectors
                draws[string(key)] = reduce(vcat, [Float64.(v) for v in val])
            else
                draws[string(key)] = Float64.(val)
            end
        end
    else
        error("Unexpected reference draw format: $(typeof(ref_data))")
    end

    return draws
end

# ─── Test runner ─────────────────────────────────────────────────────────────

const NUM_SAMPLES = 2000
const NUM_WARMUP  = 1000
const NUM_CHAINS  = 4
const SEED        = 12345

@testset "PosteriorDB Validation" begin
    pdb = database()

    for tm in POSTERIORDB_MODELS
        @testset "$(tm.posterior_name)" begin
            # Load data and reference draws from posteriordb
            post = posterior(pdb, tm.posterior_name)

            data_dict = load(dataset(post))
            ref_data  = load(reference_posterior(post))
            ref_draws = flatten_ref_draws(ref_data)

            # Build model and sample
            model = tm.build(data_dict)
            ch = sample(model, NUM_SAMPLES;
                        warmup=NUM_WARMUP, chains=NUM_CHAINS, seed=SEED)

            # Extract PhaseSkate draws
            ps_draws = tm.extract_draws(ch)

            # Only compare parameters present in both
            common_keys = intersect(keys(ps_draws), keys(ref_draws))
            ps_common  = Dict(k => ps_draws[k]  for k in common_keys)
            ref_common = Dict(k => ref_draws[k] for k in common_keys)

            @test !isempty(common_keys)

            # Z-test on means
            z_result = z_test_means(ps_common, ref_common; α=0.01)
            @testset "z-test (max_z=$(round(z_result.max_z, digits=2)))" begin
                if !z_result.pass
                    @warn "z-test failures for $(tm.posterior_name)" z_result.failing_params z_result.max_z
                end
                @test z_result.pass
            end

            # KS test
            ks_result = ks_test(ps_common, ref_common; α=0.01)
            @testset "KS test (max_D=$(round(ks_result.max_stat, digits=4)))" begin
                if !ks_result.pass
                    @warn "KS test failures for $(tm.posterior_name)" ks_result.failing_params ks_result.max_stat
                end
                @test ks_result.pass
            end

            # Quantile check
            q_result = quantile_check(ps_common, ref_common; tol=0.05)
            @testset "quantiles (max_diff=$(round(q_result.max_diff, digits=4)))" begin
                if !q_result.pass
                    @warn "Quantile failures for $(tm.posterior_name)" q_result.failing_params q_result.max_diff
                end
                @test q_result.pass
            end
        end
    end
end
