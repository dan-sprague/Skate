mutable struct WelfordState
    n::Int
    mean::Vector{Float64}
    M2n::Vector{Float64}
    
    function WelfordState(D::Int)
        new(0, zeros(D), zeros(D))
    end
end

function welford_update!(state::WelfordState, x_k::AbstractVector{Float64})
    length(x_k) == length(state.mean) || throw(ArgumentError("welford_update!: x_k length ($(length(x_k))) must match state dimension ($(length(state.mean)))"))
    state.n += 1
    
    @inbounds @simd for i in eachindex(x_k)
        Δ_old = x_k[i] - state.mean[i]
        
        state.mean[i] += Δ_old / state.n
        
        Δ_new = x_k[i] - state.mean[i]
        
        state.M2n[i] += Δ_old * Δ_new
    end
end

function welford_variance(state::WelfordState)
    if state.n < 2
        return ones(length(state.mean))
    end
    return state.M2n ./ (state.n - 1)
end

function welford_variance!(out::Vector{Float64}, state::WelfordState)
    if state.n < 2
        fill!(out, 1.0)
    else
        denom = state.n - 1
        @inbounds @simd for i in eachindex(out)
            out[i] = state.M2n[i] / denom
        end
    end
    out
end

## ── Full-covariance Welford (for dense metric) ──

mutable struct WelfordCovState
    n::Int
    mean::Vector{Float64}
    M2n::Matrix{Float64}
    delta::Vector{Float64}   # scratch buffer for rank-1 update

    function WelfordCovState(D::Int)
        new(0, zeros(D), zeros(D, D), zeros(D))
    end
end

function welford_update!(state::WelfordCovState, x_k::AbstractVector{Float64})
    length(x_k) == length(state.mean) || throw(ArgumentError("welford_update!: x_k length ($(length(x_k))) must match state dimension ($(length(state.mean)))"))
    state.n += 1

    # Δ_old = x_k - mean_old
    @inbounds @simd for i in eachindex(x_k)
        state.delta[i] = x_k[i] - state.mean[i]
    end

    # Update mean
    inv_n = 1.0 / state.n
    @inbounds @simd for i in eachindex(x_k)
        state.mean[i] += state.delta[i] * inv_n
    end

    # Δ_new = x_k - mean_new;  M2n += Δ_old * Δ_new'  via rank-1 BLAS update
    # delta currently holds Δ_old; compute Δ_new in-place after the ger!
    # We need both Δ_old and Δ_new, so compute Δ_new into a temp view
    D = length(x_k)
    @inbounds for j in 1:D
        delta_new_j = x_k[j] - state.mean[j]
        @simd for i in 1:D
            state.M2n[i, j] += state.delta[i] * delta_new_j
        end
    end
end

function welford_covariance(state::WelfordCovState)
    if state.n < 2
        D = length(state.mean)
        return Matrix{Float64}(I, D, D)
    end
    return state.M2n ./ (state.n - 1)
end