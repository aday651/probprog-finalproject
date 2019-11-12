import torch
import numpy as np


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
        r"""
        The constructor for the BitcoinOTC class.

        Parameters:
           train_num (int): The number of training samples to take from the
                labelled part of the dataset. Should be between 1 and 315.
                Defaults to 158 (=50% of the dataset).
            rng_seed (int): Set the rng seed to create the test/train split.
                If None, then the default seeding is used.
            edge_index (LongTensor): Lists the edges in the
                BitcoinOTC network, has shape [2, num of edges].
            num_nodes (int): Number of nodes in the network.
            edge_weight (Tensor): Gives weights of edges corresponding to each
                edge of edge_index, has shape [2, num of edges].
            time_stamp (Tensor): Gives the time in days the edge was formed,
                from the time at which the first edge was created. Has shape
                [2, num of edges].
            out_degree (Tensor): Gives the out degree of each node. Has shape
                [num_nodes].
            in_degree (Tensor): Gives the in degree of each node. Has shape
                [num_nodes].
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
