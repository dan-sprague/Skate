# benchmarks/survival/bench_stan.R
# Benchmark CmdStan on the Weibull survival frailty model using CmdStanR.
#
# Prerequisites:
#   install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
#   cmdstanr::install_cmdstan()
#   install.packages("jsonlite")
#
# Usage: Rscript benchmarks/survival/bench_stan.R

library(cmdstanr)
library(jsonlite)

# -- Configuration ------------------------------------------------------------

NUM_CHAINS  <- 4
NUM_WARMUP  <- 1000
NUM_SAMPLES <- 2000
SEED        <- 42

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

data_path   <- file.path(bench_dir, "data", "survival_data.json")
stan_path   <- file.path(bench_dir, "survival_frailty.stan")
result_path <- file.path(bench_dir, "results", "stan_results.json")

# -- Load data ----------------------------------------------------------------

cat("Loading data from:", data_path, "\n")
data <- fromJSON(data_path)

stan_data <- list(
  N        = as.integer(data$N),
  P        = as.integer(data$P),
  H        = as.integer(data$H),
  X        = as.matrix(data$X),
  trt      = as.numeric(data$trt),
  times    = as.numeric(data$times),
  hosp     = as.integer(data$hosp),
  N_obs    = as.integer(data$N_obs),
  N_cens   = as.integer(data$N_cens),
  obs_idx  = as.integer(data$obs_idx),
  cens_idx = as.integer(data$cens_idx)
)

# -- Compile model ------------------------------------------------------------

cat("Compiling Stan model:", stan_path, "\n")
model <- cmdstan_model(stan_path)

# -- Sample -------------------------------------------------------------------

cat(sprintf("Sampling: %d warmup, %d samples, %d chains\n",
            NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS))

t_start <- proc.time()

fit <- model$sample(
  data            = stan_data,
  chains          = NUM_CHAINS,
  parallel_chains = NUM_CHAINS,
  iter_warmup     = NUM_WARMUP,
  iter_sampling   = NUM_SAMPLES,
  seed            = SEED,
  refresh         = 500
)

t_elapsed <- (proc.time() - t_start)["elapsed"]

# -- Diagnostics --------------------------------------------------------------

cat("\n")
fit$summary()

diag <- fit$diagnostic_summary()
n_divergent <- sum(diag$num_divergent)

summ <- fit$summary()
ess_bulk_vals <- summ$ess_bulk
rhat_vals     <- summ$rhat

valid <- !is.na(ess_bulk_vals) & !is.na(rhat_vals)
min_ess  <- min(ess_bulk_vals[valid])
max_rhat <- max(rhat_vals[valid])

sampling_time <- sum(fit$time()$chains$sampling)
total_time    <- as.numeric(t_elapsed)

cat(sprintf("\n-- Results --------------------------------------------------\n"))
cat(sprintf("  Total wall time:  %.1f s\n", total_time))
cat(sprintf("  Sampling time:    %.1f s\n", sampling_time))
cat(sprintf("  Min ESS (bulk):   %.0f\n", min_ess))
cat(sprintf("  Max Rhat:         %.4f\n", max_rhat))
cat(sprintf("  ESS/s:            %.1f\n", min_ess / total_time))
cat(sprintf("  Divergences:      %d\n", n_divergent))

# -- Save results -------------------------------------------------------------

dir.create(file.path(bench_dir, "results"), showWarnings = FALSE)

results <- list(
  backend         = "CmdStan",
  num_chains      = NUM_CHAINS,
  num_warmup      = NUM_WARMUP,
  num_samples     = NUM_SAMPLES,
  total_time_s    = total_time,
  sampling_time_s = sampling_time,
  min_ess_bulk    = min_ess,
  max_rhat        = max_rhat,
  ess_per_sec     = min_ess / total_time,
  divergences     = n_divergent
)

writeLines(toJSON(results, auto_unbox = TRUE, pretty = TRUE), result_path)
cat(sprintf("\nResults saved to: %s\n", result_path))
