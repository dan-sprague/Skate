```@raw html
---
layout: home

hero:
  name: PhaseSkate
  text: High Performance Bayesian Inference In Julia
  tagline: Fast sampling built for complex and high dimension models on CPUs.
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: API Reference
      link: /api

features:
  - icon: 💻
    title: Long Live Laptop Bayesian Inference
    details: Most inference is not done on GPU clusters, but locally. PhaseSkate is built (only) for Enzyme autodiff to quickly sample complex and high dimensional models with minimal allocations.
  - icon: 🔍
    title: Clarity
    details: The @skate specification macro encourages readability and centrality of information important to understand a model at a glance.
  - icon: 𝑓(𝑥)
    title: Functional Design
    details: No DAG. PPL constructs an optimized target density function to pass to Enzyme/NUTS. Focus on your Julia!
---
```

````@raw html
<div class="vp-doc" style="width:80%; margin:auto">

<h2> Quick Example </h2>

```julia
using PhaseSkate

@skate NormalModel begin
    @constants begin
        N::Int
        y::Vector{Float64}
    end
    @params begin
        mu::Float64
        sigma = param(Float64; lower=0.0)
    end
    @logjoint begin
        target += normal_lpdf(mu, 0.0, 10.0)
        target += exponential_lpdf(sigma, 1.0)
        for i in 1:N
            target += normal_lpdf(y[i], mu, sigma)
        end
    end
end

y_data = randn(100) .* 2.0 .+ 3.0
d = NormalModelData(N=100, y=y_data)
m = make(d)

ch = sample(m, 2000; warmup=1000, chains=4)
mean(ch, :mu)    # posterior mean
ci(ch, :mu)      # 95% credible interval
```

</div>
````
