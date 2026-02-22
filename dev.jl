### WELFORD UPDATE
### Updates parameter variance information in an online fashion, without storing all samples. In place implementation.

D = 1000
xbar_prev = randn(D)
x_n = randn(D)


Δ = x_n - xbar_prev

running_mean = xbar_prev + Δ ./ D 

function welford_update(xbar_prev, M2n_prev, x_k,k)
    Δ_old = x_k - xbar_prev
    running_mean = xbar_prev + Δ_old ./ k

    Δ_new = x_k - running_mean

    M2n = M2n_prev + Δ_old .* Δ_new
    return running_mean, M2n
end

D = 1000

xbar_prev = randn(D)
M2n_prev = zeros(D)
x_n = randn(D)
for sample in 1:1000
    x_k = randn(D)
    xbar_prev, M2n_prev = welford_update(xbar_prev, M2n_prev, x_k, sample)
end


struct WelfordState
    mean::Vector{Float64}
    M2n::Vector{Float64}
    D::Int
    
    function WelfordState(mean::Vector{Float64}, M2n::Vector{Float64})
        new(mean, M2n, length(mean))
    end

end

function welford_update!(state::WelfordState, x_k::Vector{Float64})
    Δ_old = x_k - state.mean
    state.mean .+= Δ_old ./ state.D

    Δ_new = x_k - state.mean

    state.M2n .+= Δ_old .* Δ_new
end

W = WelfordState(randn(D), zeros(D))

for sample in 1:1000
    x_k = randn(D)
    welford_update!(W, x_k)
end