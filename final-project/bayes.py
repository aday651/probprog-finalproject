import torch
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from bitcoin import BitcoinOTC
import pandas as pd


def bayes_logistic_reg(model_num=1, num_samples=1000, warmup_steps=200):
    r"""Implements the logistic regression models described in the report
    writeup.

    Args:
        model_num (int, optional) : Specifies the model we want to fit. If
            this isn't equal to 1 or 2, there'll be an error somewhere.
        num_samples (int, optional) : The number of draws from the posterior
            to sample from via MCMC.
        warmup_steps (int, optional) : The number of steps used as part of
            the warmup phase in order to tune the underlying MCMC algorithm.
    Returns:
        hmc_samples (dict) : Returns a dictionary with keys corresponding to
            the names of the parameters, whose items are ndarray's of length
            num_samples.
    """

    # Define the model
    def model_1(r, sdg, sdr, sdt, gt):
        beta_r = pyro.sample("beta_r", dist.Normal(0, 1))
        beta_sdg = pyro.sample("beta_sdg", dist.Normal(0, 1))
        beta_sdr = pyro.sample("beta_sdr", dist.Normal(0, 1))
        beta_sdt = pyro.sample("beta_sdt", dist.Normal(0, 1))

        log_prob = beta_r*r + beta_sdg*sdg + beta_sdr*sdr + beta_sdt*sdt

        with pyro.plate("data", len(gt)):
            y = pyro.sample("obs", dist.Bernoulli(logits=log_prob), obs=gt)

        return y

    def model_2(r, sdg, sdr, sdt, gt):
        beta_r = pyro.sample("beta_r", dist.Normal(0, 1))
        beta_rsdg = pyro.sample("beta_rsdg", dist.Normal(0, 1))
        beta_rsdr = pyro.sample("beta_rsdr", dist.Normal(0, 1))
        beta_rsdt = pyro.sample("beta_rsdt", dist.Normal(0, 1))

        log_prob = beta_r*r + beta_rsdg*r*sdg + beta_rsdr*r*sdr + \
            beta_rsdt*r*sdt

        with pyro.plate("data", len(gt)):
            y = pyro.sample("obs", dist.Bernoulli(logits=log_prob), obs=gt)

        return y

    # Utility function to print latent sites' quantile information
    def summary(samples):
        site_stats = {}
        for site_name, values in samples.items():
            marginal_site = pd.DataFrame(values)
            describe = marginal_site.describe(
                percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
            site_stats[site_name] = describe[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
        return site_stats

    # Transform variables
    data = BitcoinOTC()

    r = data.in_weight_avg[data.nodes_train] - 0.5
    sdg = 2*data.out_weight_std[data.nodes_train]
    sdg[sdg != sdg] = 0           # set nan entries to zero
    sdr = 2*data.in_weight_std[data.nodes_train]
    sdt = torch.log(data.rate_time_out_std[data.nodes_train] + 1)
    sdt = sdt/sdt.max()
    gt = data.gt[data.nodes_train]

    # Fit via MCMC
    if model_num == 1:
        nuts_kernel = NUTS(model_1)
    elif model_num == 2:
        nuts_kernel = NUTS(model_2)
    else:
        raise ValueError('model_num must be equal to 1 or 2')

    mcmc = MCMC(nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup_steps)
    mcmc.run(r, sdg, sdr, sdt, gt)

    hmc_samples = {k: v.detach().cpu().numpy() for
                   k, v in mcmc.get_samples().items()}

    for site, values in summary(hmc_samples).items():
        print("Site: {}".format(site))
        print(values, "\n")

    # Return samples from MCMC to use further
    return hmc_samples
