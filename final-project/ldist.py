import matplotlib.pyplot as plt
import numpy as np


def plot_logit_distribution(gt, pop_logits, train_set,
                            test_set, fname=None, display=False):
    r""" Given a dictionary of posterior evaluations of logits from a training
    and test set, along with ground truth labels, displays a calibration curve
    using the posterior mean logits (and samples from the posterior disribution
    to capture uncertainity).

    Args:
        gt (ndarray): Array of ground truth labels.
        pop_logits (ndarray): Array of estimated logits, expected to have
            size (number of data points)*(number of posterior samples).
        train_set (list): List of entries corresponding to entries in the
            training set.
        test_set (list): List of entries corresponding to entries in the
            test set.
        fname (str, optional) : Path/filename to save the ROC curve. If None,
            the ROC curve is not saved.
        display (bool, optional) : Determines whether the ROC curve is
            displayed (if True) or not (if False).
    """

    # Define sigmoid function to use
    def sigmoid(x):
        return np.exp(x)/(1+np.exp(x))

    pop_logits_mean = np.mean(pop_logits)

    plt.figure(figsize=(10, 8))

    for i in range(pop_logits.shape[1]):
        if i % 20 == 0:
            plt.plot(sigmoid(np.sort(pop_logits[:, i])),
                     np.linspace(0, 1, len(pop_logits[:, i]),
                                 endpoint=False),
                     color='#0173B2', lw=2, alpha=0.1)

    plt.plot(sigmoid(np.sort(pop_logits_mean)),
             np.linspace(0, 1, len(pop_logits_mean), endpoint=False),
             color='#0173B2', lw=3)

    plt.hlines(0, 0, 1, color='#949494', lw=3, linestyle='--', alpha=0.5)
    plt.hlines(1, 0, 1, color='#949494', lw=3, linestyle='--', alpha=0.5)
    plt.vlines(0.2, 0, 1, color='#D55E00', lw=3, linestyle='-')

    plt.scatter(sigmoid(pop_logits_mean[train_set]),
                gt[train_set].numpy(),
                color='k', marker='|', lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Probability of user being trustworthy')
    plt.ylabel('Cumulative frequency in the population')
    plt.title(('Distribution of estimated trustworthiness probabilities '
               'of all users in the network'))

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    if display:
        plt.show()

    plt.close()
