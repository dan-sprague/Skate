using PhaseSkate
using Random
using Distributions
using LinearAlgebra
using CairoMakie
using Distances, Clustering

# Stan naming: K=topics, V=vocab, M=docs, N=total word instances
function simulate_lda(; K=4, V=100, M=10, N_per_doc=500, beta_val=0.15, Σ=nothing)
    # topic-word distributions: each topic has sparse high words
    phi = hcat([rand(Dirichlet(fill(beta_val, V))) for _ in 1:K]...)  # V × K

    # non-diagonal covariance for correlated topic proportions
    if Σ === nothing
        A = randn(K, K)
        Σ = A'A / K + 0.5 * I
    end
    ln_dist = MvNormal(zeros(K), Σ)

    theta = zeros(M, K)
    counts = zeros(Int, M, V)
    for m in 1:M
        η = rand(ln_dist)
        theta[m, :] = exp.(η) ./ sum(exp.(η))
        for n in 1:N_per_doc
            z = rand(Distributions.Categorical(theta[m, :]))
            w = rand(Distributions.Categorical(phi[:, z]))
            counts[m, w] += 1
        end
    end
    return (; phi, theta, counts, Σ)
end

@skate CTM begin
    @constants begin
        K::Int                    # num topics
        V::Int                    # num words
        M::Int                    # num docs
        x::Matrix{Float64}       # M × V count matrix
        log_phi0::Matrix{Float64} # V × K  log of prior mean for word dists
        tau::Float64              # precision of normal prior on log_phi (higher = tighter)
        L::Matrix{Float64}       # K × K  Cholesky factor of Σ (lower triangular)
    end

    @params begin
        eta = param(Matrix{Float64}, M, K)       # unconstrained topic log-ratios
        log_phi = param(Matrix{Float64}, V, K)   # unconstrained log word weights (column = topic)
    end

    @logjoint begin
        # prior on log_phi: Normal(log_phi0, 1/sqrt(tau)) per element
        for k in 1:K
            for v in 1:V
                target += normal_lpdf(log_phi[v, k], log_phi0[v, k], 1.0 / sqrt(tau))
            end
        end
        # precompute phi normalizers once
        @let lse_phi[k in 1:K] = log_sum_exp(log_phi[:, k])
        for m in 1:M
            target += multi_normal_cholesky_lpdf(eta[m, :], 0.0, L)
            target += correlated_topic_lpdf(x[m, :], eta[m, :], log_phi, lse_phi)
        end
    end
end

sim = simulate_lda(K=4, V=100, M=10)
log_phi0 = log.(sim.phi)  # sim.phi is V×K, log directly
L = Matrix(cholesky(sim.Σ).L)
data = CTMData(
    K=4, M=size(sim.counts, 1), V=size(sim.counts, 2),
    x=Float64.(sim.counts), log_phi0=log_phi0, tau=5.0, L=L
);
model = make(data)
chain = PhaseSkate.sample(model, 2000; warmup=1000, max_depth=8)

function theta_ci(chain; ci=0.95)
    eta = samples(chain, :eta)  # (n_samples, M, K, n_chains)
    S, M, K, C = size(eta)
    # pool samples and chains → (S*C, M, K)
    eta_pool = reshape(permutedims(eta, (1, 4, 2, 3)), S * C, M, K)
    # softmax along K dim
    theta_pool = similar(eta_pool)
    for i in axes(eta_pool, 1), m in 1:M
        mx = maximum(@view eta_pool[i, m, :])
        s = 0.0
        for k in 1:K
            theta_pool[i, m, k] = exp(eta_pool[i, m, k] - mx)
            s += theta_pool[i, m, k]
        end
        theta_pool[i, m, :] ./= s
    end
    # quantiles
    α = (1 - ci) / 2
    lo = zeros(M, K)
    hi = zeros(M, K)
    med = zeros(M, K)
    for m in 1:M, k in 1:K
        col = sort!(@view(theta_pool[:, m, k]))
        lo[m, k] = quantile(col, α; sorted=true)
        med[m, k] = quantile(col, 0.5; sorted=true)
        hi[m, k] = quantile(col, 1 - α; sorted=true)
    end
    return (; lo, med, hi)
end


function plot_ctm(chain, sim; ci=0.95)
    res = theta_ci(chain; ci)
    M, K = size(res.med)
    colors = Makie.wong_colors()

    # posterior mean phi
    lp = samples(chain, :log_phi)  # (S, V, K, C)
    S, V, K_phi, C = size(lp)
    lp_mean = dropdims(mean(reshape(permutedims(lp, (1, 4, 2, 3)), S * C, V, K_phi); dims=1); dims=1)
    phi_fit = exp.(lp_mean) ./ sum(exp.(lp_mean); dims=1)

    # cluster words
    D_words = pairwise(CosineDist(), phi_fit; dims=1)
    order = hclust(D_words; linkage=:ward).order

    fig = Figure(size=(1200, 800))

    # row 1: heatmaps
    ax1 = Axis(fig[1, 1]; title="Fitted ϕ (posterior mean)",
               xlabel="Cell Type", ylabel="Genes (clustered)", yticks=([], []))
    heatmap!(ax1, 1:K, 1:V, phi_fit[order, :]'; colormap=:viridis)

    ax2 = Axis(fig[1, 2]; title="True ϕ (simulation)",
               xlabel="Cell Type", ylabel="Genes (clustered)", yticks=([], []))
    heatmap!(ax2, 1:K, 1:V, sim.phi[order, :]'; colormap=:viridis)
    Colorbar(fig[1, 3]; colormap=:viridis, label="P(word | topic)")

    # row 2: topic trajectories
    ax3 = Axis(fig[2, 1:2]; xlabel="Time Point", ylabel="Cell Type proportion",
               title="Cell Type proportions (95% CI)")
    for k in 1:K
        band!(ax3, 1:M, res.lo[:, k], res.hi[:, k]; color=(colors[k], 0.2))
        lines!(ax3, 1:M, res.med[:, k]; color=colors[k], linewidth=2,
               label="Cell Type $k (fit)")
        scatter!(ax3, 1:M, sim.theta[:, k]; color=colors[k], marker=:x,
                 markersize=12, label="Cell Type $k (true)")
    end

    # row 3: horizontal legend
    Legend(fig[3, 1:2], ax3; orientation=:horizontal, nbanks=1)
    resize_to_layout!(fig)
    fig
end

plot_ctm(chain, sim)