import torch
from torch_geometric.data import InMemoryDataset, Data


class BitcoinOTC(InMemoryDataset):
    r"""The Bitcoin OTC dataset from https://cs.stanford.edu/%7Esrijan/rev2/,
    consisting of one graph with directed edges with two attributes, "score"
    and "time". A subset of nodes also are labelled +1 or 0 corresponding
    to whether the underlying user is trustworthy; None is used to indicate
    that there is no a-priori information

    Args:
        train_num (int, optional) : The number of nodes used to form the train
            and test data split for the per node trustworthiness ratings.
        rng_seed (int, optional) : The seed used to randomly select indices
            for the train and test splits for the nodes.
    """

    def __init__(self,
                 train_num=200,
                 rng_seed=651):
                
    

    with open(self.raw_paths[0], 'r') as f:
        data = f.read().split('\n')[:-1]
        data = [[x for x in line.split(',')] for line in data]

        edge_index = [[int(line[0]), int(line[1])] for line in data]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index - edge_index.min()
        edge_index = edge_index.t().contiguous()
        num_nodes = edge_index.max().item() + 1

        edge_attr = [float(line[2]) for line in data]
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        stamps = [int(float(line[3])) for line in data]
        stamps = [datetime.datetime.fromtimestamp(x) for x in stamps]
