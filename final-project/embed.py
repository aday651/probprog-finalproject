import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from bitcoin import BitcoinOTC


def guide(data, sample_args, embed_dim=2):
    r"""Defines a variational family to use to fit an approximate posterior
    distribution for the probability model defined in model.
    """
    # Parameters governing the priors on the embedding vectors
    omega_loc = pyro.param('omega_loc',
                           torch.randn(embed_dim, data.num_nodes))
    omega_scale = pyro.param('omega_scale', torch.tensor(1.0),
                             constraint=constraints.positive)

    # Parameters governing the prior for the linear regression
    beta_loc = pyro.param('beta_loc', 0.5*torch.randn(embed_dim))
    beta_scale = pyro.param('beta_scale', torch.tensor(1.0),
                            constraint=constraints.positive)
    mu_loc = pyro.param('mu_loc', torch.randn(1))
    mu_scale = pyro.param('mu_scale', torch.tensor(1.0),
                          constraint=constraints.positive)

    # Sample the coefficient vector and intercept for linear regression
    beta = pyro.sample('beta', dist.MultivariateNormal(
        loc=beta_loc, covariance_matrix=(beta_scale**2)*torch.eye(embed_dim)
    ))
    mu = pyro.sample('mu', dist.Normal(mu_loc, mu_scale))

    # Subsample vertices now in the guide
    subsample = data.subsample(**sample_args)

    # (Sub)sample embedding vectors
    for i in pyro.plate('nodes', size=data.num_nodes,
                        subsample=subsample['node_ind']):
        pyro.sample('omega_{}'.format(i),
                    dist.MultivariateNormal(
            loc=omega_loc[:, i],
            covariance_matrix=(omega_scale**2)*torch.eye(embed_dim)
        )
        )

    # Define plate for the edge subsampling to pass to model object
    with pyro.plate('edges', size=data.num_edges,
                    subsample=subsample['edge_ind']):
        # Note: we use pyro plate here as we need to keep the subsampling
        # persistent in both our model and guide functions, we so we
        # define an empty plate here just so we can use the same call
        # in the model.
        pass

    return beta, mu


def model(data, omega_scale=1, obs_scale=1, embed_dim=2):
    r"""Defines a probabilistic model for the observed network data."""
    # Define priors on the regression coefficients
    mu = pyro.sample('mu', dist.Normal(torch.tensor(0), torch.tensor(2)))
    beta = pyro.sample('beta', dist.MultivariateNormal(
        loc=torch.zeros(embed_dim), covariance_matrix=4*torch.eye(embed_dim)
    ))

    # Define prior on the embedding vectors, do subsampling for the
    # embedding vector and then the likelihood term for the observed nodes
    omega = [0 for i in range(data.num_nodes)]

    for i in pyro.plate('nodes', size=data.num_nodes):
        # Embedding vectors
        omega[i] = pyro.sample('omega_{}'.format(i), dist.MultivariateNormal(
            loc=torch.zeros(embed_dim),
            covariance_matrix=(omega_scale**2)*torch.eye(embed_dim)
        ))

        # Draw Bernoulli, with or without data depending on if it is observed
        logit = mu + torch.dot(beta, omega[i])
        if i in data.nodes_train:
            pyro.sample('trust_{}'.format(i), dist.Bernoulli(logits=logit),
                        data=data.gt[i])

    # Draw terms corresponding to the edges
    for i in pyro.plate('nodes', size=data.num_edges):
        logit_rating = 0.05 + 0.9*data.edge_weight[i]
        logit_rating = torch.log((logit_rating)/(1 - logit_rating))
        emip = torch.dot(omega[data.edge_index[0, i]],
                         omega[data.edge_index[1, i]])
        pyro.sample('a_{}'.format(i), dist.Normal(emip, obs_scale),
                    data=logit_rating)
