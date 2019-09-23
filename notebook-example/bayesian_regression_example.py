# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file
# location. Turn this addition off with the
# DataScience.changeDirOnImportExport setting ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'notebook-example'))
    print(os.getcwd())
except:
    pass
#%%
from IPython import get_ipython

#%% [markdown]
# # Bayesian Regression Example for ML w/ Prob Prog
#%% [markdown]
# Adapted from http://pyro.ai/examples/bayesian_regression_ii.html

#%%
from __future__ import absolute_import, division, print_function

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, JitTrace_ELBO
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
import pyro.optim as optim

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('0.4.1')


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)


#%%
rugged_data = pd.read_csv('rugged_data.csv', encoding='ISO-8859-1')

#%% [markdown]
# ## Bayesian Linear Regression
# 
# Our goal is once again to predict log GDP per capita of a nation as a function of two features from the dataset - whether the nation is in Africa, and its Terrain Ruggedness Index, but we will explore more expressive guides.
#%% [markdown]
# ## Model + Guide
# 
# We will write out the model again, similar to that in [Part I](bayesian_regression.ipynb), but explicitly without the use of `nn.Module`.  We will write out each term in the regression, using the same priors. `bA` and `bR` are regression coefficients corresponding to  `is_cont_africa` and `ruggedness`, `a` is the intercept, and `bAR` is the correlating factor between the two features.
# 
# Writing down a guide will proceed in close analogy to the construction of our model, with the key difference that the guide parameters need to be trainable. To do this we register the guide parameters in the ParamStore using `pyro.param()`. Note the positive constraints on scale parameters.

#%%
# Prepare training data
df = rugged_data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
train = torch.tensor(df.values, dtype=torch.float)

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

#%%
def model(is_cont_africa, ruggedness, log_gdp):
    beta_0 = pyro.sample("beta_0", dist.Normal(0, 1))
    beta_1 = pyro.sample("beta_1", dist.Normal(0, 1))
    beta_2 = pyro.sample("beta_2", dist.Normal(0, 1))
    beta_3 = pyro.sample("beta_3", dist.Normal(0, 1))
    
    sigma = pyro.sample("sigma", dist.Gamma(1, 1))
    mean = beta_0 + beta_1 * is_cont_africa + beta_2 * ruggedness + beta_3 * is_cont_africa * ruggedness
    
    with pyro.plate("data", len(log_gdp)):
        y = pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)
    
    return y


#%%
model(is_cont_africa, ruggedness, log_gdp)


#%%
log_gdp


#%%
def guide(is_cont_africa, ruggedness, log_gdp):
    weights_loc = pyro.param('weights_loc', torch.randn(4))
    weights_scale = pyro.param('weights_scale', torch.ones(4), constraint=constraints.positive)        
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.0), constraint=constraints.positive)
    
    beta_0 = pyro.sample("beta_0", dist.Normal(weights_loc[0], weights_scale[0]))
    beta_1 = pyro.sample("beta_1", dist.Normal(weights_loc[1], weights_scale[1]))
    beta_2 = pyro.sample("beta_2", dist.Normal(weights_loc[2], weights_scale[2]))
    beta_3 = pyro.sample("beta_3", dist.Normal(weights_loc[3], weights_scale[3]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    
    mean = beta_0 +            beta_1 * is_cont_africa +            beta_2 * ruggedness +            beta_3 * is_cont_africa * ruggedness


#%%
# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

#%% [markdown]
# ## SVI
# 
# As before, we will use SVI to perform inference.

#%%
svi = SVI(model, 
          guide, 
          optim.Adam({"lr": .005}), 
          loss=JitTrace_ELBO(), 
          num_samples=1000)

pyro.clear_param_store()
num_iters = 10000 if not smoke_test else 2
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))


#%%
svi_diagnorm_posterior = svi.run(log_gdp, is_cont_africa, ruggedness)

#%% [markdown]
# Let us observe the posterior distribution over the different latent variables in the model.

#%%
sites = ["beta_0", "beta_1", "beta_2", "beta_3", "sigma"]

svi_samples = {site: EmpiricalMarginal(svi_diagnorm_posterior, sites=site)
                     .enumerate_support().detach().cpu().numpy()
               for site in sites}

for site, values in summary(svi_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")

#%% [markdown]
# ## HMC
# 
# In contrast to using variational inference which gives us an approximate posterior over our latent variables, we can also do exact inference using [Markov Chain Monte Carlo](http://docs.pyro.ai/en/dev/mcmc.html) (MCMC), a class of algorithms that in the limit, allow us to draw unbiased samples from the true posterior. The algorithm that we will be using is called the No-U Turn Sampler (NUTS) \[1\], which provides an efficient and automated way of running Hamiltonian Monte Carlo.  It is slightly slower than variational inference, but provides an exact estimate.

#%%
nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc_run = mcmc.run(is_cont_africa, ruggedness, log_gdp)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}


#%%
for site, values in summary(hmc_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")

#%% [markdown]
# ## Comparing Posterior Distributions
# 
# Let us compare the posterior distribution of the latent variables that we obtained from variational inference with those from Hamiltonian Monte Carlo. As can be seen below, for Variational Inference, the marginal distribution of the different regression coefficients is under-dispersed w.r.t. the true posterior (from HMC). This is an artifact of the *KL(q||p)* loss (the KL divergence of the true posterior from the approximate posterior) that is minimized by Variational Inference.
# 
# This can be better seen when we plot different cross sections from the joint posterior distribution overlaid with the approximate posterior from variational inference. Note that since our variational family has diagonal covariance, we cannot model any correlation between the latents and the resulting approximation is overconfident (under-dispersed)

#%%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    try:
        site = sites[i]
        sns.distplot(hmc_samples[site], ax=ax, label="HMC")        
        sns.distplot(svi_samples[site], ax=ax, label="SVI (DiagNormal)")
        ax.set_title(site)
    except:
        pass
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');


#%%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
sns.kdeplot(hmc_samples["beta_1"], hmc_samples["beta_2"], ax=axs[0], shade=True, label="HMC")
sns.kdeplot(svi_samples["beta_1"], svi_samples["beta_2"], ax=axs[0], label="SVI (DiagNormal)")
axs[0].set(xlabel="beta_1", ylabel="beta_2", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(hmc_samples["beta_2"], hmc_samples["beta_3"], ax=axs[1], shade=True, label="HMC")
sns.kdeplot(svi_samples["beta_2"], svi_samples["beta_3"], ax=axs[1], label="SVI (DiagNormal)")
axs[1].set(xlabel="beta_2", ylabel="beta_3", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');

#%% [markdown]
# ## MultivariateNormal Guide
# 
# As comparison to the previously obtained results from Diagonal Normal guide, we will now use a guide that generates samples from a Cholesky factorization of a multivariate normal distribution.  This allows us to capture the correlations between the latent variables via a covariance matrix.  If we wrote this manually, we would need to combine all the latent variables so we could sample a Multivarite Normal jointly.

#%%
from pyro.infer.autoguide.initialization import init_to_mean

guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)

svi = SVI(model, 
          guide, 
          optim.Adam({"lr": .005}), 
          loss=Trace_ELBO(), 
          num_samples=1000)

pyro.clear_param_store()
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))

#%% [markdown]
# We will discuss why we are rerunning this optimization again later in class.

#%%
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))

#%% [markdown]
# Let's look at the shape of the posteriors again.  You can see the multivariate guide is able to capture more of the true posterior.

#%%
svi_mvn_posterior = svi.run(log_gdp, is_cont_africa, ruggedness)
svi_mvn_samples = {site: EmpiricalMarginal(svi_mvn_posterior, sites=site).enumerate_support()
                         .detach().cpu().numpy()
                   for site in sites}
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    try:
        site = sites[i]
        sns.distplot(hmc_samples[site], ax=ax, label="HMC")        
        sns.distplot(svi_mvn_samples[site], ax=ax, label="SVI (Multivariate Normal)")
        ax.set_title(site)
    except:
        pass
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');

#%% [markdown]
# Now let's compare the posterior computed by the Diagonal Normal guide vs the Multivariate Normal guide.  Note that the multivariate distribution is more dispresed than the Diagonal Normal.
#%% [markdown]
# and the Multivariate guide with the posterior computed by HMC.  Note that the Multivariate guide better captures the true posterior.

#%%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
sns.kdeplot(hmc_samples["beta_1"], hmc_samples["beta_2"], ax=axs[0], shade=True, label="HMC")
sns.kdeplot(svi_mvn_samples["beta_1"], svi_mvn_samples["beta_2"], ax=axs[0], label="SVI (Multivariate Normal)")
axs[0].set(xlabel="beta_1", ylabel="beta_2", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(hmc_samples["beta_2"], hmc_samples["beta_3"], ax=axs[1], shade=True, label="HMC")
sns.kdeplot(svi_mvn_samples["beta_2"], svi_mvn_samples["beta_3"], ax=axs[1], label="SVI (Multivariate Normal)")
axs[1].set(xlabel="beta_2", ylabel="beta_3", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');

#%% [markdown]
# ## References
# [1] Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research 15.1 (2014): 1593-1623.  https://arxiv.org/abs/1111.4246.

