# Benchmark raw gradient evaluation time for CmdStan
library(cmdstanr)
library(jsonlite)

bench_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
if (is.null(bench_dir)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    bench_dir <- dirname(sub("--file=", "", file_arg))
  } else {
    bench_dir <- "benchmarks/survival"
  }
}

data_path <- file.path(bench_dir, "data", "survival_data.json")
stan_path <- file.path(bench_dir, "survival_frailty.stan")

data <- fromJSON(data_path)
stan_data <- list(
  N = as.integer(data$N), P = as.integer(data$P), H = as.integer(data$H),
  X = as.matrix(data$X), trt = as.numeric(data$trt),
  times = as.numeric(data$times), hosp = as.integer(data$hosp),
  N_obs = as.integer(data$N_obs), N_cens = as.integer(data$N_cens),
  obs_idx = as.integer(data$obs_idx), cens_idx = as.integer(data$cens_idx)
)

# Compile model
model <- cmdstan_model(stan_path)

# Run 1 warmup + 1 sample just to get a fit object with grad_log_prob
fit <- model$sample(
  data = stan_data, chains = 1,
  iter_warmup = 1, iter_sampling = 1,
  seed = 42, refresh = 0
)

# Get unconstrained params from the single draw
upar <- fit$unconstrain_draws(format = "draws_matrix")[1,]
cat(sprintf("Stan unconstrained dim = %d\n", length(upar)))

# Benchmark gradient
N_eval <- 1000
t0 <- proc.time()
for (i in 1:N_eval) {
  fit$grad_log_prob(upar)
}
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Stan gradient: %.4f ms per eval (%d evals in %.2f s)\n",
            as.numeric(elapsed) / N_eval * 1000, N_eval, as.numeric(elapsed)))
