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

---
```

````@raw html
<style>
/* ── Widen VitePress layout ────────────────────────────────── */
:root { --vp-layout-max-width: 1800px; }

/* ── Hero layout ───────────────────────────────────────────── */
.VPHome .VPHero .image { display: none; }

/* ── Hero + Case Study side by side on wide screens ────────── */
.VPHome {
  display: grid !important;
  grid-template-columns: 1fr;
  max-width: 1800px;
  margin: 0 auto;
  padding: 0 24px;
}
@media (min-width: 1200px) {
  .VPHome {
    grid-template-columns: auto 1fr;
    align-items: start;
    gap: 2rem;
  }
  .VPHome > .VPHero {
    grid-column: 1;
    grid-row: 1;
    max-width: 480px;
    position: sticky;
    top: calc(var(--vp-nav-height) + 1rem);
  }
  /* raw HTML slot — may have extra wrapper divs */
  .VPHome > .vp-doc,
  .VPHome > div:last-child {
    grid-column: 2;
    grid-row: 1;
  }
}

/* ── Case study section ────────────────────────────────────── */
.case-study {
  width: 100%;
  max-width: 100%;
  margin: 0;
  padding: 2.5rem 3rem 2.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
}
.case-study-title {
  font-size: 2rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem 0;
  line-height: 1.2;
}
.case-study-subtitle {
  font-size: 1rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin: 0 0 2rem 0;
  max-width: 70ch;
}

/* ── Model summary table ─────────────────────────────────── */
.model-summary {
  border-collapse: collapse;
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
  color: var(--vp-c-text-1);
  width: auto;
}
.model-summary th,
.model-summary td {
  padding: 0.4rem 1.2rem;
  border: 1px solid var(--vp-c-divider);
  text-align: left;
}
.model-summary th {
  background: var(--vp-c-bg-soft);
  font-weight: 600;
}

/* ── Stacked: plot on top, toggle below ───────────────────── */
.case-study-grid {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* ── Toggle switch ───────────────────────────────────────── */
.model-toggle input[type="radio"] { display: none; }

.toggle-bar {
  display: inline-flex;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 1.2rem;
}
.toggle-bar label {
  padding: 0.45rem 1.2rem;
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  color: var(--vp-c-text-2);
  transition: background 0.15s, color 0.15s;
  user-select: none;
}
.toggle-bar label:not(:last-child) {
  border-right: 1px solid var(--vp-c-divider);
}
#toggle-desc:checked ~ .toggle-bar label[for="toggle-desc"],
#toggle-code:checked ~ .toggle-bar label[for="toggle-code"] {
  background: var(--vp-c-brand);
  color: #fff;
}
.dark #toggle-desc:checked ~ .toggle-bar label[for="toggle-desc"],
.dark #toggle-code:checked ~ .toggle-bar label[for="toggle-code"] {
  background: var(--vp-dark-green);
  color: #fff;
}

/* ── Toggle content visibility ───────────────────────────── */
#toggle-desc:checked ~ .panels-wrapper > .panel-desc { visibility: visible; }
#toggle-code:checked ~ .panels-wrapper > .panel-code { visibility: visible; }

/* ── Keep toggle panels same size (overlay so height doesn't jump on toggle) */
.model-toggle {
  min-width: 0;
  position: relative;
}
.panels-wrapper {
  display: grid;
}
.panels-wrapper > .panel-desc,
.panels-wrapper > .panel-code {
  grid-column: 1;
  grid-row: 1;
  visibility: hidden;
}

/* ── Bullet list styling ─────────────────────────────────── */
.model-toggle .panel-desc ul {
  padding-left: 1.25rem;
  line-height: 1.75;
}
.model-toggle .panel-desc li {
  margin-bottom: 0.4rem;
}

/* ── Code panel font ─────────────────────────────────────── */
.model-toggle .panel-code pre,
.model-toggle .panel-code code {
  font-size: 0.75rem !important;
  line-height: 1.5 !important;
}

/* ── Plot card ───────────────────────────────────────────── */
.plot-card {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  background: #1b1b1f;
}
.plot-card img {
  display: block;
  width: 100%;
  height: auto;
}
</style>

<div class="vp-doc case-study">

<h1 class="case-study-title">Case Study: Multi-site Hierarchical Survival Model</h1>
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
  <input type="radio" name="model-view" id="toggle-desc" checked />
  <input type="radio" name="model-view" id="toggle-code" />

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

</div>

</div>
````
