#===============================================================================
library(ggplot2)
library(rstan)
library(bayesplot)
library(coda)
library(stats)

#===============================================================================
# Question 1

# Set random seed for reproducibility
set.seed(42)

# Define parameters
N <- 200  # Number of observations
alpha <- 0.5  # Difficulty parameter
beta <- 1.5   # Discrimination parameter
gamma <- 0.2  # Guessing parameter

# Generate x, y, data and logistic model
x <- runif(N, -2, 2)  

logit <- gamma + (1 - gamma) * (1 / (1 + exp(-beta * (x - alpha))))

y <- rbinom(N, size = 1, prob = logit)

data <- data.frame(x = x, y = y)

head(data)

data$y <- as.factor(data$y)
summary(data)

# Count 0's and 1's
table(data$y)

# Proportions
prop.table(table(data$y))

#===============================================================================
# Question 2

# Gaussian priors

set.seed(123)
N <- 200       
p <- 3        
beta_true <- c(0.1, 1.1, -0.9)  
X <- matrix(runif(N * p, -2, 2), ncol = p)
logit <- X %*% beta_true
prob <- 1 / (1 + exp(-logit))
y <- rbinom(N, size = 1, prob = prob)


prior_mean <- rep(0, p)

prior_sd <- rep(1, p)

# Posterior distribution

log_posterior <- function(beta, X, y, prior_mean, prior_sd) {
  log_lik <- sum(y * (X %*% beta) - log(1 + exp(X %*% beta)))
  log_prior <- sum(dnorm(beta, mean = prior_mean, sd = prior_sd, log = TRUE))
  return(log_lik + log_prior)
}


# Metropolis-Hastings Algorithm
metropolis_hastings <- function(init, X, y, prior_mean, prior_sd, n_iter, sigma) {
  p <- length(init)
  samples <- matrix(NA, nrow = n_iter, ncol = p)
  accept <- 0
  beta <- init
  
  for (i in 1:n_iter) {
    proposal <- beta + rnorm(p, mean = 0, sd = sigma)
    log_accept_ratio <- log_posterior(proposal, X, y, prior_mean, prior_sd) - 
      log_posterior(beta, X, y, prior_mean, prior_sd)
    
    if (log(runif(1)) < log_accept_ratio) {
      beta <- proposal
      accept <- accept + 1
    }
    samples[i, ] <- beta
  }
  acceptance_rate <- accept / n_iter
  return(list(samples = samples, acceptance_rate = acceptance_rate))
}

# Run MCMC for a single chain
set.seed(42)
n_iter <- 2500
sigma <- 0.314  # Tune this to achieve ~23.4% acceptance rate
init <- rnorm(p, mean = 0, sd = 1)
mh_results <- metropolis_hastings(init, X, y, prior_mean, prior_sd, n_iter, sigma)

# Acceptance rate
print(paste("Acceptance Rate:", mh_results$acceptance_rate))

# Trace plots for all parameters
par(mfrow = c(1, 3))  # Adjust layout for 9 parameters
for (i in 1:p) {
  plot(mh_results$samples[, i], type = "l", col = "blue",
       main = paste("Trace Plot for Beta", i - 1),
       xlab = "Iteration", ylab = paste("Beta", i - 1))
}

# Remove burn-in
burn_in <- 2000
post_burn_in_samples <- mh_results$samples[(burn_in + 1):n_iter, ]

# Posterior histograms with true values
par(mfrow = c(3, 1))
posterior_means <- colMeans(post_burn_in_samples)

for (i in 1:p) {
  hist(post_burn_in_samples[, i], breaks = 30, col = "lightblue",
       main = paste("Histogram for Beta", i - 1),
       xlab = paste("Beta", i - 1))
  abline(v = posterior_means[i], col = "blue", lwd = 2, lty = 2)  # Posterior mean
  abline(v = beta_true[i], col = "red", lwd = 2, lty = 2)         # True value
}

# Run multiple chains
set.seed(123)
M <- 20
chains <- list()
for (m in 1:M) {
  init <- rnorm(p, mean = 0, sd = 1)
  chains[[m]] <- metropolis_hastings(init, X, y, prior_mean, prior_sd, n_iter, sigma)$samples
}

# Convert to mcmc.list object
mcmc_chains <- mcmc.list(lapply(chains, function(x) mcmc(x[(burn_in + 1):n_iter, ])))

# Gelman-Rubin diagnostic
gelman_diag <- gelman.diag(mcmc_chains)
print(gelman_diag)

# Plot Gelman-Rubin statistic over iterations
gelman.plot(mcmc_chains)


#===============================================================================
# Question 3
generate_logistic_data_9d <- function(N, beta) {
  x <- matrix(runif(N * 9, -2, 2), ncol = 9)
  linear_predictor <- beta[1] + rowSums(x * beta[2:10])
  probabilities <- 1 / (1 + exp(-linear_predictor))
  y <- rbinom(N, size = 1, prob = probabilities)
  return(data.frame(x, y))
}

# Parameters for the logistic model
beta_true <- c(0.1, 1.1, -0.9, rep(0, 7))  # beta_1 = 0.1, beta_2 = 1.1, beta_3 = -0.9, rest are 0
N <- 200

# Generate data
data_9d <- generate_logistic_data_9d(N, beta_true)


metropolis_hastings_9d <- function(data, n_iter, init, proposal_sd, target_accept_rate = 0.234) {
  current <- init
  samples <- matrix(0, nrow = n_iter, ncol = length(init))
  acceptances <- 0
  
  likelihood <- function(beta, data) {
    x <- data[, 1:9]
    y <- data$y
    linear_predictor <- beta[1] + rowSums(x * beta[2:10])
    prob <- 1 / (1 + exp(-linear_predictor))
    prod(prob^y * (1 - prob)^(1 - y))
  }
  
  prior <- function(beta) {
    prod(dnorm(beta, 0, 1))
  }
  
  posterior <- function(beta, data) {
    likelihood(beta, data) * prior(beta)
  }
  
  for (i in 1:n_iter) {
    proposal <- rnorm(length(current), mean = current, sd = proposal_sd)
    accept_prob <- min(1, posterior(proposal, data) / posterior(current, data))
    if (runif(1) < accept_prob) {
      current <- proposal
      acceptances <- acceptances + 1
    }
    samples[i, ] <- current
  }
  
  acceptance_rate <- acceptances / n_iter
  list(samples = samples, acceptance_rate = acceptance_rate)
}

# Run Metropolis-Hastings for the 9-dimensional model
n_iter <- 5000
proposal_sd <- 0.5  # Initial scaling of the proposal
init <- rnorm(10, 0, 1)  # Start from prior
mh_results_9d <- metropolis_hastings_9d(data_9d, n_iter, init, proposal_sd)

# Adjust proposal_sd to achieve target_accept_rate
while (abs(mh_results_9d$acceptance_rate - 0.234) > 0.01) {
  proposal_sd <- proposal_sd * ifelse(mh_results_9d$acceptance_rate < 0.234, 0.9, 1.1)
  mh_results_9d <- metropolis_hastings_9d(data_9d, n_iter, init, proposal_sd)
}

cat("Final acceptance rate:", mh_results_9d$acceptance_rate, "\n")


# Plot trace plots
plot_trace_9d <- function(samples, burn_in = 1000) {
  par(mfrow = c(3, 4))
  for (i in 1:ncol(samples)) {
    plot(samples[, i], type = "l", col = "blue", main = paste("Trace plot for Beta", i - 1),
         xlab = "Iteration", ylab = paste("Beta", i - 1))
    abline(v = burn_in, col = "red", lty = 2)  # Mark burn-in
  }
}

burn_in <- 1000  # Adjust based on trace plot inspection
plot_trace_9d(mh_results_9d$samples, burn_in)

# Remove burn-in samples
posterior_samples_9d <- mh_results_9d$samples[-(1:burn_in), ]

# Posterior summaries
posterior_means_9d <- colMeans(posterior_samples_9d)
posterior_sds_9d <- apply(posterior_samples_9d, 2, sd)
cat("Posterior Means:", posterior_means_9d, "\n")
cat("Posterior SDs:", posterior_sds_9d, "\n")

# Histograms for each parameter
par(mfrow = c(3, 4))
for (i in 1:ncol(posterior_samples_9d)) {
  hist(posterior_samples_9d[, i], breaks = 30, col = "lightblue",
       main = paste("Posterior Histogram for Beta", i - 1),
       xlab = paste("Beta", i - 1))
  abline(v = posterior_means_9d[i], col = "blue", lty = 2, lwd = 2)  # Posterior mean
  abline(v = beta_true[i], col = "red", lty = 2, lwd = 2)  # True value
}


#===============================================================================


# Logistic function
logistic <- function(x) {
  return(1 / (1 + exp(-x)))
}

# Log posterior function
log_posterior <- function(beta, X, y) {
  # Prior term: Gaussian N(0, 1)
  log_prior <- sum(dnorm(beta, mean = 0, sd = 1, log = TRUE))
  
  # Likelihood term
  eta <- X %*% beta
  log_likelihood <- sum(y * log(logistic(eta)) + (1 - y) * log(1 - logistic(eta)))
  
  return(log_prior + log_likelihood)
}

# Random-Walk Metropolis-Hastings
metropolis_hastings <- function(start, X, y, n_iter, proposal_sd) {
  beta <- start
  samples <- matrix(NA, nrow = n_iter, ncol = length(start))
  acceptance_count <- 0
  
  for (i in 1:n_iter) {
    proposal <- beta + rnorm(length(beta), mean = 0, sd = proposal_sd)
    log_acceptance_ratio <- log_posterior(proposal, X, y) - log_posterior(beta, X, y)
    
    if (log(runif(1)) < log_acceptance_ratio) {
      beta <- proposal
      acceptance_count <- acceptance_count + 1
    }
    
    samples[i, ] <- beta
  }
  
  acceptance_rate <- acceptance_count / n_iter
  return(list(samples = samples, acceptance_rate = acceptance_rate))
}

# Metropolis-within-Gibbs
metropolis_within_gibbs <- function(start, X, y, n_iter, proposal_sd) {
  beta <- start
  p <- length(start)
  samples <- matrix(NA, nrow = n_iter, ncol = p)
  acceptance_counts <- rep(0, p)
  
  for (i in 1:n_iter) {
    for (j in 1:p) {
      # Propose new value for beta[j]
      proposal <- beta
      proposal[j] <- beta[j] + rnorm(1, mean = 0, sd = proposal_sd[j])
      
      # Calculate acceptance ratio
      log_acceptance_ratio <- log_posterior(proposal, X, y) - log_posterior(beta, X, y)
      
      if (log(runif(1)) < log_acceptance_ratio) {
        beta[j] <- proposal[j]
        acceptance_counts[j] <- acceptance_counts[j] + 1
      }
    }
    samples[i, ] <- beta
  }
  
  acceptance_rates <- acceptance_counts / n_iter
  return(list(samples = samples, acceptance_rates = acceptance_rates))
}

# Generate synthetic data
set.seed(123)
n <- 200
p <- 9
X <- cbind(1, matrix(rnorm(n * (p - 1)), nrow = n))
true_beta <- c(0.1, 1.1, -0.9, 1.0, -1.9, 1.1, -0.5, 1.7, -0.9)
y <- rbinom(n, size = 1, prob = logistic(X %*% true_beta))

# Run Metropolis-Hastings
n_iter <- 10000
start <- c(0.1, 1.1, -0.9, 1.0, -1.9, 1.1, -0.5, 1.7, -0.9)
proposal_sd <- 0.1  # Tune for 23.4% acceptance
mh_result <- metropolis_hastings(start, X, y, n_iter, proposal_sd)

# Run Metropolis-within-Gibbs
proposal_sd_gibbs <- rep(0.05, p)  # Tune for ~15% acceptance per parameter
gibbs_result <- metropolis_within_gibbs(start, X, y, n_iter, proposal_sd_gibbs)

# Trace plots
samples_mh <- mh_result$samples
samples_gibbs <- gibbs_result$samples

par(mfrow = c(3, 3))
for (j in 1:p) {
  plot(samples_mh[, j], type = 'l', main = paste0("Trace plot for beta[", j, "] (MH)"), 
       xlab = "Iteration", ylab = paste0("beta[", j, "]"))
  abline(h = true_beta[j], col = "red", lty = 2)
}

par(mfrow = c(3, 3))
for (j in 1:p) {
  plot(samples_gibbs[, j], type = 'l', main = paste0("Trace plot for beta[", j, "] (Gibbs)"), 
       xlab = "Iteration", ylab = paste0("beta[", j, "]"))
  abline(h = true_beta[j], col = "red", lty = 2)
}

# Burn-in and posterior histograms
burn_in <- 2000
post_burn_samples_mh <- samples_mh[(burn_in + 1):n_iter, ]
post_burn_samples_gibbs <- samples_gibbs[(burn_in + 1):n_iter, ]

par(mfrow = c(3, 3))
for (j in 1:p) {
  hist(post_burn_samples_mh[, j], breaks = 30, probability = TRUE,
       main = paste0("Histogram for beta[", j, "] (MH)"),
       xlab = paste0("beta[", j, "]"), col = "lightblue")
  abline(v = mean(post_burn_samples_mh[, j]), col = "red", lwd = 2, lty = 2)
  abline(v = true_beta[j], col = "green", lwd = 2, lty = 2)
}

par(mfrow = c(3, 3))
for (j in 1:p) {
  hist(post_burn_samples_gibbs[, j], breaks = 30, probability = TRUE,
       main = paste0("Histogram for beta[", j, "] (Gibbs)"),
       xlab = paste0("beta[", j, "]"), col = "lightblue")
  abline(v = mean(post_burn_samples_gibbs[, j]), col = "red", lwd = 2, lty = 2)
  abline(v = true_beta[j], col = "green", lwd = 2, lty = 2)
}

# Gelman-Rubin diagnostic
M <- 10
multi_chain_samples_mh <- array(NA, dim = c(n_iter, p, M))
multi_chain_samples_gibbs <- array(NA, dim = c(n_iter, p, M))
for (m in 1:M) {
  start_random <- rnorm(p, mean = 0, sd = 1)
  mh_result_chain <- metropolis_hastings(start_random, X, y, n_iter, proposal_sd)
  gibbs_result_chain <- metropolis_within_gibbs(start_random, X, y, n_iter, proposal_sd_gibbs)
  
  multi_chain_samples_mh[, , m] <- mh_result_chain$samples
  multi_chain_samples_gibbs[, , m] <- gibbs_result_chain$samples
}

# Convert to mcmc.list for Gelman-Rubin
chains_mh <- lapply(1:M, function(m) as.mcmc(multi_chain_samples_mh[, , m]))
chains_gibbs <- lapply(1:M, function(m) as.mcmc(multi_chain_samples_gibbs[, , m]))

chains_gibbs


# Plot Gelman-Rubin statistic
gelman_rubin_mh <- gelman.diag(mcmc.list(chains_mh))
gelman_rubin_gibbs <- gelman.diag(mcmc.list(chains_gibbs))
gelman_rubin_gibbs
gelman.plot(as.mcmc.list(lapply(chains_gibbs, as.mcmc)))


#===============================================================================
# Question 5

# Define the function to run Metropolis-Hastings with random proposal selection
metropolis_hastings_random_proposal <- function(N, beta_start, log_likelihood, log_prior, iterations) {
  beta_samples <- array(NA, c(iterations, N))
  beta_samples[1, ] <- beta_start
  
  for (iter in 2:iterations) {
    # Randomly choose between two proposals (10% or 30% acceptance rate)
    proposal_std <- ifelse(runif(1) < 0.5, 0.1, 0.3)  # 50% chance to pick either
    
    # Propose new sample using random-walk Metropolis-Hastings with chosen proposal
    beta_proposed <- rnorm(N, mean = beta_samples[iter - 1, ], sd = proposal_std)
    
    # Calculate acceptance probability
    log_acceptance_ratio <- log_likelihood(beta_proposed) + log_prior(beta_proposed) - 
      log_likelihood(beta_samples[iter - 1, ]) - log_prior(beta_samples[iter - 1, ])
    acceptance_prob <- min(1, exp(log_acceptance_ratio))
    
    # Accept or reject the proposed sample
    if (runif(1) < acceptance_prob) {
      beta_samples[iter, ] <- beta_proposed
    } else {
      beta_samples[iter, ] <- beta_samples[iter - 1, ]
    }
  }
  return(beta_samples)
}

# Example of how to apply the function
# Define your log-likelihood and log-prior functions based on the 9-dimensional problem

log_likelihood <- function(beta) {
  # Your log-likelihood function
}

log_prior <- function(beta) {
  # Your log-prior function (e.g., Gaussian prior)
}

# Run the sampler
N <- 9
beta_start <- rep(0, N)  # Initial values
iterations <- 10000
M <- 20  # Number of chains

# Collect samples for M chains
samples_list <- vector("list", M)
for (i in 1:M) {
  samples_list[[i]] <- metropolis_hastings_random_proposal(N, beta_start, log_likelihood, log_prior, iterations)
}


# Convert each chain into mcmc object
mcmc_chains <- lapply(samples_list, as.mcmc)

# Combine into an mcmc.list
mcmc_list <- as.mcmc.list(mcmc_chains)

# Calculate the Gelman-Rubin statistic
gelman_result <- gelman.diag(mcmc_list)
print(gelman_result)


# Convert to mcmc.list for Gelman-Rubin diagnostics
mcmc <- as.mcmc(samples_list)

# Calculate the Gelman-Rubin statistic
gelman_result <- gelman.diag(mcmc_list)
print(gelman_result)
