```@raw html
---
layout: page
sidebar: false
---
```

````@raw html
<div class="ps-home">

<div class="ps-hero">
  <h1 class="ps-hero-name">PhaseSkate</h1>
  <p class="ps-hero-text">Bayesian Inference In Julia</p>
  <p class="ps-hero-tagline">Fast sampling built for complex models on your laptop.</p>

  <div class="ps-hero-actions">
    <a class="ps-btn ps-btn-brand" href="/getting_started">Getting Started</a>
    <a class="ps-btn ps-btn-alt" href="/api">API Reference</a>
  </div>

  <div class="ps-features">
    <div class="ps-feature">
      <div class="ps-feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      </div>
      <div class="ps-feature-body">
        <strong>High Performance</strong>
        <span>PhaseSkate was built for sampling large models on your laptop, quickly!</span>
      </div>
    </div>
    <div class="ps-feature">
      <div class="ps-feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
      </div>
      <div class="ps-feature-body">
        <strong>Targeted</strong>
        <span>PhaseSkate was built to excel at HMC. Guides available on how to sample discrete models.</span>
      </div>
    </div>
    <div class="ps-feature">
      <div class="ps-feature-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
      </div>
      <div class="ps-feature-body">
        <strong>Real Time Sampling Analytics</strong>
        <span>The PhaseSkate TUI provides real-time chain traces and analytics so issues can be spotted early.</span>
      </div>
    </div>
  </div>
</div>

<div class="ps-showcase vp-doc showcase-toggle">

<input type="radio" name="showcase" id="showcase-casestudy">
<input type="radio" name="showcase" id="showcase-ide" checked>

<div class="showcase-bar">
  <label for="showcase-ide">Real-Time Interface</label>
  <label for="showcase-casestudy">Example: Hierarchical AFT Survival</label>
</div>

<div class="showcase-panels">

<div class="showcase-panel showcase-panel-ide">

<h1 class="case-study-title">PhaseSkate IDE</h1>
<p class="case-study-subtitle">A 1,000-group hierarchical normal model (the classic "Eight Schools" structure scaled up with random data) sampled live in the PhaseSkate TUI.</p>
<ul class="ide-notes">
  <li>The TUI can be loaded with a prewritten model, or launched from an active REPL session.</li>
  <li>Enzyme compiles the gradient on the first call (brief delay), but subsequent calls are instantaneous as shown below.</li>
</ul>

<div class="ide-demo-card">
  <div id="ide-player"></div>
</div>

</div> <!-- end showcase-panel-ide -->

<div class="showcase-panel showcase-panel-casestudy">

<h1 class="case-study-title">Example: Hierarchical Accelerating Failure Time Survival</h1>
<p class="case-study-subtitle">Sample a complex, hierarchical survival model with thousands of observations in approximately a minute with Enzyme powered dense mass matrix NUTS sampling.</p>

<table class="model-summary">
  <thead><tr><th>Observations</th><th>Hospitals</th><th>Covariates</th><th>Parameters</th><th>Chains</th><th>Draws / chain</th></tr></thead>
  <tbody><tr><td>5,000</td><td>100</td><td>8</td><td>214</td><td>4</td><td>2,000</td></tr></tbody>
</table>

<div class="case-study-grid">

<div class="plot-card">
  <img src="/assets/survival_example.svg"
       alt="Hierarchical survival model: posterior predictive curves, scaling benchmark, and posterior agreement" />
</div>

<div class="model-toggle">
  <input type="radio" name="model-view" id="toggle-desc" checked>
  <input type="radio" name="model-view" id="toggle-code">

  <div class="toggle-bar">
    <label for="toggle-desc">Description</label>
    <label for="toggle-code">Code</label>
  </div>

  <div class="panels-wrapper">
  <div class="panel-desc">
    <ul>
      <li><b>Weibull accelerated failure time (AFT) likelihood</b> for right-censored time-to-event data across <i>N</i> patients</li>
      <li><b>8 patient-level covariates</b> with regularised regression coefficients (&beta; ~ Normal(0, 1))</li>
      <li><b>Correlated site-level random effects</b> &mdash; hospital intercepts and treatment slopes share a bivariate correlation &rho; &isin; [&minus;1, 1] via non-centred parameterisation</li>
      <li><b>Right-censoring</b> handled natively: observed events contribute the Weibull log-pdf, censored observations contribute the log-CCDF</li>
      <li><b>Scales to hundreds of sites</b>: total parameter dimension grows as 13 + 2<i>H</i></li>
      <li><b>4 chains &times; 2 000 draws</b> with Enzyme LLVM autodiff &mdash; full posterior in seconds on a laptop</li>
    </ul>
  </div>

  <div class="panel-code">

```julia
@skate SurvivalFrailty begin
    @constants begin
        N::Int; P::Int; H::Int
        X::Matrix{Float64}; trt::Vector{Float64}; times::Vector{Float64}
        hosp::Vector{Int}; obs_idx::Vector{Int}; cens_idx::Vector{Int}
    end
    @params begin
        log_alpha::Float64; beta_0::Float64
        trt_effect::Float64
        beta = param(Vector{Float64}, P)
        log_sigma_int::Float64; log_sigma_slope::Float64
        rho = param(Float64; lower=-1.0, upper=1.0)
        z_int = param(Vector{Float64}, H)
        z_slope = param(Vector{Float64}, H)
    end
    @logjoint begin
        alpha = exp(log_alpha)
        sigma_int = exp(log_sigma_int)
        sigma_slope = exp(log_sigma_slope)
        sqrt_1mrho2 = sqrt(1.0 - rho * rho)

        target += normal_lpdf(log_alpha, 0.0, 0.5)
        target += normal_lpdf(beta_0, 2.0, 2.0)
        target += normal_lpdf(trt_effect, 0.0, 1.0)
        target += multi_normal_diag_lpdf(beta, 0.0, 1.0)
        target += normal_lpdf(log_sigma_int, -1.0, 1.0)
        target += normal_lpdf(log_sigma_slope, -1.0, 1.0)
        target += log(1.0 - rho * rho)
        target += multi_normal_diag_lpdf(z_int, 0.0, 1.0)
        target += multi_normal_diag_lpdf(z_slope, 0.0, 1.0)

        @for log_scale = beta_0 .+ (X * beta)
            .+ sigma_int .* z_int[hosp]
            .+ trt .* (trt_effect .+ sigma_slope
                .* (rho .* z_int[hosp]
                .+ sqrt_1mrho2 .* z_slope[hosp]))

        target += weibull_logsigma_lpdf_sum(times, alpha, log_scale, obs_idx)
        target += weibull_logsigma_lccdf_sum(times, alpha, log_scale, cens_idx)
    end
end
```

  </div>
  </div>
</div>

</div> <!-- end case-study-grid -->

</div> <!-- end showcase-panel-casestudy -->

</div> <!-- end showcase-panels -->

</div> <!-- end ps-showcase -->

</div> <!-- end ps-home -->

<script setup>
import { onMounted } from 'vue'
import { withBase } from 'vitepress'

onMounted(() => {
  // Fix internal links for production base path
  document.querySelectorAll('.ps-home a[href^="/"]').forEach(a => {
    a.setAttribute('href', withBase(a.getAttribute('href')))
  })

  // Load CSS
  if (!document.querySelector('link[href*="asciinema-player"]')) {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = 'https://unpkg.com/asciinema-player@3.15.1/dist/bundle/asciinema-player.css'
    document.head.appendChild(link)
  }

  // Load JS, then create player when IDE tab is first shown
  const script = document.createElement('script')
  script.src = 'https://unpkg.com/asciinema-player@3.15.1/dist/bundle/asciinema-player.min.js'
  script.onload = () => {
    const radio = document.getElementById('showcase-ide')
    const castUrl = withBase('/assets/demo.cast')

    function initPlayer() {
      const el = document.getElementById('ide-player')
      if (!el || el.hasChildNodes()) return
      window.AsciinemaPlayer.create(
        castUrl,
        el,
        { autoPlay: true, loop: true, speed: 2, theme: 'monokai', fit: 'width', startAt: 34, endAt: 61 }
      )
    }

    // If IDE tab is already active, init now; otherwise wait for click
    if (radio && radio.checked) {
      initPlayer()
    }
    radio.addEventListener('change', () => { setTimeout(initPlayer, 50) })
    // Also catch label clicks
    document.querySelector('label[for="showcase-ide"]')
      ?.addEventListener('click', () => { setTimeout(initPlayer, 100) })
  }
  document.head.appendChild(script)
})
</script>
````
