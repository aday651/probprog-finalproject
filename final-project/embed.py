import time
import logging
import torch
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
import numpy as np
from torch.distributions import constraints
from pyro.infer import SVI, TraceGraph_ELBO


def embed(data, embed_args_dict, sample_args_dict):
    r"""Fits a combined matrix factorization and linear regression model on
    the observed network data.

    Args:
        data: This is expected to be the data object returned from
            bitcoin.BitcoinOTC(), and this script probably won't work
            otherwise.

        embed_args_dict (dict): A dictionary which is expected to contain
            the following keys:

            embed_dim (int): An integer specifying the embedding dimension.
            omega_model_scale (float): A positive float which specifies the
                prior variance on the embedding vectors.
            obs_scale (float): A positive float which specifies the variance
                for the observed logit scaled ratings.
            krein (bool): Specifies whether to use a Krein style inner
                product (a difference of two inner products; krein = True)
                or a regular inner product (krein = False) between
                embedding vectors for the matrix factorization part
                of the model. This implicitly assumes that the embedding
                dimension is even; if not, then an error will likely pop up
                somewhere.
            learning_rate (float): Learning rate for the ADAM optimizer.
            num_iters (int): Number of iterations to perform SVI.
            logging (bool): If True, then ELBO updates and the time ellapsed
                are output every 500 iterations, unless num_iters <= 1000
                in which case it is set to 100.

        sample_args_dict (dict): A dictionary with handles the subsampling
            procedure used on the data object. Check the docstring for the
            data object for the relevant details.

    Returns:
        logits (): The estimated logit probability of belonging to the
            truthfull class
    """
    # Extract keys from embed_args_dict
    embed_dim = embed_args_dict['embed_dim']
    omega_model_scale = embed_args_dict['omega_model_scale']
    obs_scale = embed_args_dict['obs_scale']
    learning_rate = embed_args_dict['learning_rate']
    num_iters = embed_args_dict['num_iters']
    logging_ind = embed_args_dict['logging']

    # Set logging printing rate
    if (num_iters > 1000):
        log_update = 500
    else:
        log_update = 100

    # Make sure logging actually works
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # Define guide object for embedding model
    def guide(data, node_ind, edge_ind, edge_list):
        r"""Defines a variational family to use to fit an approximate posterior
        distribution for the probability model defined in model."""
        # Deleting arguments not used in the guide for linting purposes
        del edge_ind, edge_list

        # Parameters governing the priors on the embedding vectors
        # omega_loc should have shape [embed_dum, data.num_nodes]
        omega_loc = pyro.param('omega_loc',
                               lambda: torch.randn(
                                   embed_dim, data.num_nodes
                               )/np.sqrt(embed_dim)
                               )
        # omega_scale should be a single positive tensor
        omega_scale = pyro.param('omega_scale',
                                 torch.tensor(1.0),
                                 constraint=constraints.positive)

        # Paramaeters governing the prior fr the linear regression
        # beta_loc should be of shape [embed_dim]
        beta_loc = pyro.param('beta_loc',
                              0.5*torch.randn(embed_dim))
        # beta_scale should be a single positive tensor
        beta_scale = pyro.param('beta_scale',
                                torch.tensor(1.0),
                                constraint=constraints.positive)
        # mu_loc should be a single tensor
        mu_loc = pyro.param('mu_loc',
                            torch.tensor([0.0]))
        # mu_scale should be a single positive tensor
        mu_scale = pyro.param('mu_scale',
                              torch.tensor(1.0),
                              constraint=constraints.positive)

        # Sample the coefficient vector and intercept for linear regression
        beta = pyro.sample('beta',
                           dist.Normal(loc=beta_loc,
                                       scale=beta_scale*torch.ones(embed_dim)
                                       ).to_event(1))
        mu = pyro.sample('mu',
                         dist.Normal(mu_loc, mu_scale).to_event(1))

        # Handle the subsampling of the embedding vectors
        with poutine.scale(scale=data.num_nodes/len(node_ind)):
            omega = pyro.sample('omega',
                                dist.Normal(loc=omega_loc[:, node_ind],
                                            scale=omega_scale).to_event(2))

        return beta, mu, omega

    # Defines the model to use for SVI when using the usual inner product
    def model_ip(data, node_ind, edge_ind, edge_list):
        r"""Defines a probabilistic model for the observed network data."""
        # Define priors on the regression coefficients
        mu = pyro.sample('mu', dist.Normal(
            torch.tensor([0.0]), torch.tensor([2.0])
        ).to_event(1))

        beta = pyro.sample('beta', dist.Normal(
            loc=torch.zeros(embed_dim), scale=torch.tensor(2.0)
        ).to_event(1))

        # Define prior on the embedding vectors, with subsampling
        with poutine.scale(scale=data.num_nodes/len(node_ind)):
            omega = pyro.sample('omega', dist.Normal(
                loc=torch.zeros(embed_dim, len(node_ind)),
                scale=omega_model_scale).to_event(2)
            )

        # Before proceeding further, define a list t which acts as the
        # inverse function of node_ind - i.e it takes a number in node_ind
        # to its index location
        t = torch.zeros(node_ind.max() + 1, dtype=torch.long)
        t[node_ind] = torch.arange(len(node_ind))

        # Create mask corresponding to entries of ind which lie within the
        # training set (i.e data.train_nodes)
        gt_data = data.gt[node_ind]
        obs_mask = np.isin(node_ind, data.nodes_train).tolist()
        gt_data[gt_data != gt_data] = 0.0
        obs_mask = torch.tensor(obs_mask, dtype=torch.bool)

        # Compute logits, compute relevant parts of sample
        if sum(obs_mask) != 0:
            logit_prob = mu + torch.mv(omega.t(), beta)
            with poutine.scale(scale=len(data.nodes_train)/sum(obs_mask)):
                pyro.sample('trust',
                            dist.Bernoulli(
                                logits=logit_prob[obs_mask]
                            ).independent(1),
                            obs=gt_data[obs_mask]
                            )

        # Begin extracting the relevant components of the gram matrix
        # formed by omega. Note that to extract the relevant indices,
        # we need to account for the change in indexing induced by
        # subsampling omega
        gram = torch.mm(omega.t(), omega)
        gram_sample = gram[t[edge_list[0, :]], t[edge_list[0, :]]]

        # Finally draw terms corresponding to the edges
        with poutine.scale(scale=data.num_edges/len(edge_ind)):
            pyro.sample('a', dist.Normal(
                loc=gram_sample, scale=obs_scale).to_event(1),
                obs=data.edge_weight_logit[edge_ind])

    # Defines the model to use for SVI when using the usual inner product
    def model_krein(data, node_ind, edge_ind, edge_list):
        r"""Defines a probabilistic model for the observed network data."""
        # Define priors on the regression coefficients
        mu = pyro.sample('mu', dist.Normal(
            torch.tensor([0.0]), torch.tensor([2.0])
        ).to_event(1))

        beta = pyro.sample('beta', dist.Normal(
            loc=torch.zeros(embed_dim), scale=torch.tensor(2.0)
        ).to_event(1))

        # Define prior on the embedding vectors, with subsampling
        with poutine.scale(scale=data.num_nodes/len(node_ind)):
            omega = pyro.sample('omega', dist.Normal(
                loc=torch.zeros(embed_dim, len(node_ind)),
                scale=omega_model_scale).to_event(2)
            )

        # Before proceeding further, define a list t which acts as the
        # inverse function of node_ind - i.e it takes a number in node_ind
        # to its index location
        t = torch.zeros(node_ind.max() + 1, dtype=torch.long)
        t[node_ind] = torch.arange(len(node_ind))

        # Create mask corresponding to entries of ind which lie within the
        # training set (i.e data.train_nodes)
        gt_data = data.gt[node_ind]
        obs_mask = np.isin(node_ind, data.nodes_train).tolist()
        gt_data[gt_data != gt_data] = 0.0
        obs_mask = torch.tensor(obs_mask, dtype=torch.bool)

        # Compute logits, compute relevant parts of sample
        if sum(obs_mask) != 0:
            logit_prob = mu + torch.mv(omega.t(), beta)
            with poutine.scale(scale=len(data.nodes_train)/sum(obs_mask)):
                pyro.sample('trust',
                            dist.Bernoulli(
                                logits=logit_prob[obs_mask]
                            ).independent(1),
                            obs=gt_data[obs_mask]
                            )

        # Begin extracting the relevant components of the gram matrix
        # formed by omega. Note that to extract the relevant indices,
        # we need to account for the change in indexing induced by
        # subsampling omega
        gram_pos = torch.mm(omega[:int(embed_dim/2), :].t(),
                            omega[:int(embed_dim/2), :])
        gram_neg = torch.mm(omega[int(embed_dim/2):, :].t(),
                            omega[int(embed_dim/2):, :])
        gram = gram_pos - gram_neg
        gram_sample = gram[t[edge_list[0, :]], t[edge_list[0, :]]]

        # Finally draw terms corresponding to the edges
        with poutine.scale(scale=data.num_edges/len(edge_ind)):
            pyro.sample('a', dist.Normal(
                loc=gram_sample, scale=obs_scale).to_event(1),
                obs=data.edge_weight_logit[edge_ind])

    # Define SVI object depending on if we're using a positive definite
    # bilinear form on embedding vectors or the Krein inner product
    if embed_args_dict['krein']:
        svi = SVI(model_krein, guide, optim.Adam({"lr": learning_rate}),
                  loss=TraceGraph_ELBO())
    else:
        svi = SVI(model_ip, guide, optim.Adam({"lr": learning_rate}),
                  loss=TraceGraph_ELBO())

    # Begin optimization
    # Keep track of time/optizing if desired
    if logging_ind:
        time_store = []
        t0 = time.time()
        elbo = []

    pyro.clear_param_store()
    for i in range(num_iters):
        # Really bad error handling for when the subsampling code for the
        # random walk decides to break
        count = 0
        while (count < 20):
            try:
                subsample_dict = data.subsample(**sample_args_dict)
                count = 30
            except IndexError:
                count += 1

        elbo_val = svi.step(data, **subsample_dict)
        if logging_ind & (i % log_update == 0) & (i > 0):
            elbo.append(elbo_val)
            t1 = time.time()
            time_store.append(t1-t0)
            logging.info('Elbo loss: {}'.format(elbo_val))
            logging.info('Expected completion time: {}s'.format(
                int(np.average(time_store)*(num_iters - i)/log_update)
            ))
            t0 = time.time()

    # Extract the variational parameters and return them
    vp_dict = {}
    vp_dict['mu_loc'] = pyro.param('mu_loc')
    vp_dict['beta_loc'] = pyro.param('beta_loc')
    vp_dict['omega_loc'] = pyro.param('omega_loc')
    vp_dict['mu_scale'] = pyro.param('mu_scale')
    vp_dict['beta_scale'] = pyro.param('beta_scale')
    vp_dict['omega_scale'] = pyro.param('omega_scale')

    if 'elbo' in locals():
        return vp_dict, elbo
    else:
        return vp_dict
