data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> H;
  matrix[N, P] X;
  vector[N] trt;
  vector<lower=0>[N] times;
  array[N] int<lower=1, upper=H> hosp;
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  array[N_obs] int<lower=1, upper=N> obs_idx;
  array[N_cens] int<lower=1, upper=N> cens_idx;
}
parameters {
  real log_alpha;
  real beta_0;
  real trt_effect;
  vector[P] beta;
  real log_sigma_int;
  real log_sigma_slope;
  real<lower=-1, upper=1> rho;
  vector[H] z_int;
  vector[H] z_slope;
}
model {
  real alpha = exp(log_alpha);
  real sigma_int = exp(log_sigma_int);
  real sigma_slope = exp(log_sigma_slope);
  real sqrt_1mrho2 = sqrt(1.0 - rho * rho);

  log_alpha ~ normal(0, 0.5);
  beta_0 ~ normal(2, 2);
  trt_effect ~ normal(0, 1);
  beta ~ normal(0, 1);
  log_sigma_int ~ normal(-1, 1);
  log_sigma_slope ~ normal(-1, 1);
  target += log(1 - rho * rho);
  z_int ~ std_normal();
  z_slope ~ std_normal();

  vector[N] log_scale = beta_0 + X * beta
    + sigma_int * z_int[hosp]
    + trt .* (trt_effect + sigma_slope * (rho * z_int[hosp] + sqrt_1mrho2 * z_slope[hosp]));

  for (i in 1:N_obs) {
    int idx = obs_idx[i];
    target += weibull_lpdf(times[idx] | alpha, exp(log_scale[idx]));
  }
  for (i in 1:N_cens) {
    int idx = cens_idx[i];
    target += weibull_lccdf(times[idx] | alpha, exp(log_scale[idx]));
  }
}
