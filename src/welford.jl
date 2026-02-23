mutable struct WelfordState
    n::Int
    mean::Vector{Float64}
    M2n::Vector{Float64}
    
    function WelfordState(D::Int)
        new(0, zeros(D), zeros(D))
    end
end

function welford_update!(state::WelfordState, x_k::AbstractVector{Float64})
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