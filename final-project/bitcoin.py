import torch
import numpy as np
from torch_cluster import random_walk


class BitcoinOTC(object):
    r"""The Bitcoin OTC dataset from https://cs.stanford.edu/%7Esrijan/rev2/,
    consisting of one graph with directed edges with two attributes, "score"
    and "time". A subset of nodes also are labelled +1 or 0 corresponding
    to whether the underlying user is trustworthy; None is used to indicate
    that there is no a-priori information

    Attributes:
        train_num (int, optional) : The number of nodes used to form the train
            and test data split for the per node trustworthiness ratings.
        rng_seed (int, optional) : The seed used to randomly select indices
            for the train and test splits for the nodes.
    """

    def __init__(self, train_num=158, rng_seed=651):
        r"""The constructor for the BitcoinOTC class.

        Parameters:
           train_num (int): The number of training samples to take from the
                labelled part of the dataset. Should be between 1 and 315.
                Defaults to 158 (=50% of the dataset).
            rng_seed (int): Set the rng seed to create the test/train split.
                If None, then the default seeding is used.
            edge_index (LongTensor): Lists the edges in the
                BitcoinOTC network, has shape [2, num of edges].
            num_nodes (int): Number of nodes in the network.
            num_edges (int): Number of edges in the network.
            edge_weight (Tensor): Gives weights of edges corresponding to each
                edge of edge_index, has shape [num of edges].
            time_stamp (Tensor): Gives the time in days the edge was formed,
                from the time at which the first edge was created. Has shape
                [2, num of edges].
            out_degree (Tensor): Gives the out degree of each node. Has shape
                [num_nodes].
            in_degree (Tensor): Gives the in degree of each node. Has shape
                [num_nodes].
            nodes_pos_degree_mask (Tensor) : Gives the locations of the nodes
                which have positive in and out degree.
            edges_pos_degree_mask (Tensor) : Gives the locations of the edges,
                with respect to edge_index, whose start and end nodes lie
                within nodes_pos_degree_mask.
            out_weight_avg (Tensor): Gives the average of the out weights to
                a node. Has shape [num_nodes]. Equals nan if the
                corresponding out_degree equals zero.
            out_weight_std (Tensor): Gives the standard deviation of the out
                weights to a node. Has shape [num_nodes]. Equals zero if the
                corresponding out_degree equals one; equals nan if the
                corresponding out_degree equals zero.
            in_weight_avg (Tensor): Gives the average of the in weights to
                a node. Has shape [num_nodes]. Equals nan if the
                corresponding out_degree equals zero.
            in_weight_std (Tensor): Gives the standard deviation of the in
                weights to a node. Has shape [num_nodes]. Equals zero if the
                corresponding out_degree equals one; equals nan if the
                corresponding out_degree equals zero.
            rate_time_out_std (Tensor): Gives the standard deviation of the
                rating times for the edges leaving a node.
                Has shape [num_nodes]. Equals zero if the corresponding
                out_degree equals one; equals nan if the corresponding
                out_degree equals zero.
            rate_time_in_std (Tensor): Gives the standard deviation of the
                rating times for the edges entering a node.
                Has shape [num_nodes]. Equals zero if the corresponding
                out_degree equals one; equals nan if the corresponding
                out_degree equals zero.
            gt (Tensor): Denotes whether a node corresponds to a trustworthy
                user (=1), a not trustworthy user (=0), or the latent state
                is unknown (=nan). Has shape [num_nodes]
            nodes_with_gt (list): Lists the nodes for which we know the
            underlying latent state. Has shape [316].
            nodes_train (list)): Lists nodes corresponding to the training data
            split. Has shape [train_num].
            nodes_test (list): Lists nodes corresponding to the test data
            split. Has shape [316 - train_num].
        """

        self.train_num = train_num
        self.rng_seed = rng_seed

        if rng_seed is not None:
            np.random.seed(rng_seed)

        # Read edge information
        with open('data/otc_network.csv', 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[x for x in line.split(',')] for line in data]

            # Create list of edges
            edge_index = [[int(line[0])-1, int(line[1])-1] for line in data]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            self.edge_index = edge_index.t().contiguous()
            self.num_nodes = edge_index.max().item() + 1
            self.num_edges = self.edge_index.shape[1]

            # Create edge weights, transform to lie in [0, 1]
            edge_weight = [(float(line[2])+10)/20 for line in data]
            self.edge_weight = torch.tensor(edge_weight, dtype=torch.float)

            # Create time stamps for edges, transform to be in terms of
            # days after the first transaction on the network
            time_stamp = [int(float(line[3])) for line in data]
            time_stamp = torch.tensor(time_stamp, dtype=torch.float)
            time_stamp = time_stamp - time_stamp.min()
            self.time_stamp = time_stamp/86400

            # Create per node summary statistics (in/out degree, etc)
            out_degree = [0 for i in range(self.num_nodes)]
            in_degree = [0 for i in range(self.num_nodes)]
            out_weights = [[] for i in range(self.num_nodes)]
            in_weights = [[] for i in range(self.num_nodes)]
            time_in_ratings = [[] for i in range(self.num_nodes)]
            time_out_ratings = [[] for i in range(self.num_nodes)]

            for line in data:
                out_degree[int(line[0])-1] += 1
                in_degree[int(line[1])-1] += 1
                out_weights[int(line[0])-1].append(
                    (float(line[2])+10)/20
                )
                in_weights[int(line[1])-1].append(
                    (float(line[2])+10)/20
                )
                time_in_ratings[int(line[0])-1].append(
                    float(line[3])/86400
                )
                time_out_ratings[int(line[1])-1].append(
                    float(line[3])/86400
                )

            self.out_degree = torch.tensor(out_degree, dtype=torch.float)
            self.in_degree = torch.tensor(in_degree, dtype=torch.float)

            # Find nodes with positive in and out degree
            self.nodes_pos_degree_mask = torch.where(
                (self.out_degree > 0) & (self.in_degree > 0)
            )[0]

            self.edges_pos_degree_mask = np.where(
                np.isin(self.edge_index[0, :].numpy(),
                        self.nodes_pos_degree_mask.numpy())
                & np.isin(self.edge_index[1, :].numpy(),
                          self.nodes_pos_degree_mask.numpy())
            )[0]
            self.edges_pos_degree_mask = torch.from_numpy(
                self.edges_pos_degree_mask
            )

            # Give summary statistics for each node
            self.out_weight_avg = torch.tensor(
                [np.mean(item) for item in out_weights],
                dtype=torch.float
            )
            self.in_weight_avg = torch.tensor(
                [np.mean(item) for item in in_weights],
                dtype=torch.float
            )
            self.out_weight_std = torch.tensor(
                [np.std(item) for item in out_weights],
                dtype=torch.float
            )
            self.in_weight_std = torch.tensor(
                [np.std(item) for item in in_weights],
                dtype=torch.float
            )
            self.rate_time_out_std = torch.tensor(
                [np.std(item) for item in time_out_ratings],
                dtype=torch.float
            )
            self.rate_time_in_std = torch.tensor(
                [np.std(item) for item in time_in_ratings],
                dtype=torch.float
            )

        # Read ground truth data
        with open('data/otc_gt.csv', 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[x for x in line.split(',')] for line in data]

            # Create node indicator marker
            gt = [float('nan') for i in range(self.num_nodes)]
            for line in data:
                if line[1] == '+1':
                    gt[int(line[0])-1] = 1
                elif line[1] == '-1':
                    gt[int(line[0])-1] = 0
            self.gt = torch.tensor(gt, dtype=torch.float)

            # Create test/train splits
            self.nodes_with_gt = np.array([int(line[0])-1 for line in data])
            self.nodes_train = np.random.choice(
                self.nodes_with_gt,
                size=self.train_num,
                replace=False
            )
            self.nodes_test = np.setdiff1d(
                self.nodes_with_gt, self.nodes_train
            )
            self.nodes_with_gt = self.nodes_with_gt.tolist()
            self.nodes_train = self.nodes_train.tolist()
            self.nodes_test = self.nodes_test.tolist()

    def subsample(self, sample_str=None, sample_args=None):
        r"""Draws a subsample from the observed network with respect to
        some user specified sampling scheme.

        Args:
            sample_str (str) : Specifies the type of subsampling scheme to use.
                Must be one of the following:
                    'p-sampling' : Samples edges uniformly at random.
                    'random-walk' : Performs a random walk on the undirected
                        network, starting from a random location each time.
            sample_args (dict): Specifies arguments to use for the type of
                subsampling scheme specified by sample_str. See the docstring
                for the different choices of sampling scheme for more info.

        Returns:
            subsample_dict (dict) : A dictionary with the following three keys:
                'node_ind' : The indices of the nodes which are part of the
                    subsampled graph.
                'edge_ind' : The indices of the edges which are part of the
                    subsampled graph, with respect to the ordering given in
                    self.edge_index and self.edge_weight.
                'edge_list' : Pairs of edges which are part of the subsampled
                    graph, stored as a Tensor of shape
                    [2, num of subsampled edges].
        """
        if sample_str == 'p-sampling':
            subsample_dict = self.subsample_psampling(sample_args=sample_args)
        elif sample_str == 'random-walk':
            subsample_dict = self.subsample_walk(sample_args=sample_args)
        else:
            raise ValueError('sample_str not valid')

        return subsample_dict

    def subsample_psampling(self, sample_args=None):
        r"""Takes a subsample of the graph by randomly selecting observed edges
        with a given user probability.

        Args:
            sample_args (dict) : Dictionary with the following keys/items:
                sample_prob (float) : Probability of edge being selected.
                Must be between 0 and 1.

        Returns:
            subsample_dict (dict) : A dictionary with the following three keys:
                'node_ind' : The indices of the nodes which are part of the
                    subsampled graph.
                'edge_ind' : The indices of the edges which are part of the
                    subsampled graph, with respect to the ordering given in
                    self.edge_index and self.edge_weight.
                'edge_list' : Pairs of edges which are part of the subsampled
                    graph, stored as a Tensor of shape
                    [2, num of subsampled edges].
            """
        # Extract sampling probability
        sample_prob = sample_args['sample_prob']

        # Select eges randomly with probability sample_prob
        ind = np.random.binomial(1, sample_prob, size=self.num_edges)
        edge_ind = np.where(ind == 1)
        edge_ind = torch.as_tensor(edge_ind, dtype=torch.long)[0]

        # Extract the relevant edges and nodes
        edge_list = self.edge_index[:, edge_ind]
        node_ind = torch.unique(edge_list)

        subsample_dict = {'node_ind': node_ind, 'edge_ind': edge_ind,
                          'edge_list': edge_list}

        return subsample_dict

    def subsample_walk(self, sample_args=None):
        r"""Takes a subsample of the graph by performing random walks at
        various randomly selected vertices of particular lengths.

        Args:
            sample_args (dict) : Dictionary with the following keys/items:
                start_num (int) : Decides the number of randomly selected
                    nodes to begin random walks at.
                walk_length (int) : Decides the length of each random walk
                    to perform.

        Returns:
            subsample_dict (dict) : A dictionary with the following three keys:
                'node_ind' : The indices of the nodes which are part of the
                    subsampled graph.
                'edge_ind' : The indices of the edges which are part of the
                    subsampled graph, with respect to the ordering given in
                    self.edge_index and self.edge_weight.
                'edge_list' : Pairs of edges which are part of the subsampled
                    graph, stored as a Tensor of shape
                    [2, num of subsampled edges].
        """
        # Extract values from sample_dict
        start_num = sample_args['start_num']
        walk_length = sample_args['walk_length']

        # Choose random starting locations
        start_loc = torch.as_tensor(
            np.random.choice(self.nodes_pos_degree_mask.numpy(),
                             size=start_num, replace=False),
            dtype=torch.long
        )

        # Perform random walks; note that walks has shape
        # [start_num, walk_length + 1]
        walks = random_walk(row=self.edge_index[0, self.edges_pos_degree_mask],
                            col=self.edge_index[1, self.edges_pos_degree_mask],
                            start=start_loc,
                            walk_length=walk_length,
                            coalesced=True)

        # Convert into edge list
        edge_list = torch.zeros(size=(2, start_num*walk_length),
                                dtype=torch.long)

        for i in range(start_num):
            ind = i*walk_length + np.arange(walk_length)
            ind = list(ind)
            edge_list[0, ind] = walks[i, :-1]
            edge_list[1, ind] = walks[i, 1:]

        # Get rid of duplicates in edge list
        edge_list = torch.unique(edge_list, dim=1)

        # Extract nodes and the indices of edge_list with respect to
        # self.edge_index
        node_ind = torch.unique(edge_list)
        edge_ind = torch.zeros(size=[edge_list.shape[1]], dtype=torch.long)

        for i in range(edge_list.shape[1]):
            edge_ind[i] = torch.where(
                (self.edge_index[0, :] == edge_list[0, i])
                & (self.edge_index[1, :] == edge_list[1, i])
            )[0][0]

        subsample_dict = {'node_ind': node_ind, 'edge_ind': edge_ind,
                          'edge_list': edge_list}

        return subsample_dict
