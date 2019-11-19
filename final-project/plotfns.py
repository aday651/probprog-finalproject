import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


def sigmoid(x):
    r"""Returns the sigmoid of the input (applied elementwise for arrays)."""
    return np.exp(x)/(1+np.exp(x))


def plot_calibration_curve(gt, logits, fname=None, display=False):
    r""" Given a dictionary of posterior evaluations of logits from a training
    and test set, along with ground truth labels, displays a calibration curve
    using the posterior mean logits (and samples from the posterior disribution
    to capture uncertainity).

    Args:
        gt (dict): Dictionary of ground truth labels, expected to be of the
            form {'train': ndarray of length (train set size),
                  'test': ndarray of length (test set size)}.
        logits (dict) : Dictionary of logits, expected to be of the form
            {'train': ndarray of size (train set size)*(num of samples),
             'test': ndarray of size (test set size)*(num of samples)}.
        fname (str, optional) : Path/filename to save the ROC curve. If None,
            the ROC curve is not saved.
        display (bool, optional) : Determines whether the ROC curve is
            displayed (if True) or not (if False).
    Returns:
        None.
    """

    # Begin creating the plot
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot calibration curves for different draws from the posterior
    for i in range(logits['test'].shape[1]):
        if i % 20 == 0:
            frac_positive, mean_pred = calibration_curve(
                gt['test'],
                sigmoid(logits['test'][:, i]),
                n_bins=10)
            plt.plot(mean_pred, frac_positive,
                     's-', color='#0173B2', alpha=0.1)

    # Plot calibration curve using the posterior mean of the logits
    frac_positive, mean_pred = calibration_curve(
        gt['test'],
        sigmoid(np.mean(logits['test'], axis=1)),
        n_bins=10)
    plt.plot(mean_pred, frac_positive,
             's-', color='#0173B2',
             label='Estimated probabilities (test set)')

    plt.scatter(sigmoid(np.mean(logits['test'], axis=1)), gt['test'],
                color='k', marker='|')

    plt.xlabel('Actual values')
    plt.ylabel('Fraction of positives')
    plt.ylim([-0.04, 1.04])
    plt.legend(loc="lower right")
    plt.title('Calibration curve')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    if display:
        plt.show()

    plt.close()


def plot_roc_curve(gt, logits, fname=None, display=False):
    r""" Given a dictionary of posterior evaluations of logits from a training
    and test set, displays a ROC curve along with an estimate of the AOC
    (with uncertainities).

    Args:
        gt (dict): Dictionary of ground truth labels, expected to be of the
            form {'train': ndarray of length (train set size),
                  'test': ndarray of length (test set size)}.
        logits (dict) : Dictionary of logits, expected to be of the form
            {'train': ndarray of size (train set size)*(num of samples),
             'test': ndarray of size (test set size)*(num of samples)}.
        fname (str, optional) : Path/filename to save the ROC curve. If None,
            the ROC curve is not saved.
        display (bool, optional) : Determines whether the ROC curve is
            displayed (if True) or not (if False).
    Returns:
        None.
    """

    # Keeping track of ROC and AOC values, storing e.g colours
    num_of_samples = logits['train'].shape[1]
    color_dict = {'train': '#0173B2', 'test': '#D55E00'}
    aoc_dict = {'train': [], 'test': []}

    # Figure plotting
    plt.figure(figsize=(7, 5))

    # For training and test sets,
    for key in ['train', 'test']:
        # Begin by computing AOC for each sample, plotting a subset of them
        for i in range(num_of_samples):
            # Calculate AOC for the sample
            fpr, tpr, _ = roc_curve(
                y_true=gt[key],
                y_score=sigmoid(logits[key][:, i])
            )
            aoc_dict[key].append(auc(fpr, tpr))

            if i % 20 == 0:
                plt.plot(fpr, tpr, color=color_dict[key], lw=2, alpha=0.1)

        # Plot posterior mean for train/test, give AOC with +\- std
        fpr, tpr, _ = roc_curve(
            y_true=gt[key],
            y_score=sigmoid(np.mean(logits[key], axis=1))
        )

        label = r'{} AOC: {:.4f} $\pm$ {:.4f}'.format(
            key, np.mean(aoc_dict[key]), np.std(aoc_dict[key])
        )
        plt.plot(fpr, tpr, color=color_dict[key], lw=4, label=label)

    # Add axis labels, plot/save the figure
    plt.plot([0, 1], [0, 1], color='#949494', lw=3, linestyle='--')
    plt.xlim([0.0, 1])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for training and test sets')
    plt.legend(loc="lower right")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    if display:
        plt.show()

    plt.close()


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
    Returns:
        None
    """

    pop_logits_mean = np.mean(pop_logits, axis=1)

    plt.figure(figsize=(7, 5))

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

    plt.scatter(sigmoid(pop_logits_mean[test_set]),
                gt[test_set].numpy(),
                color='#cc78bc', marker='*', lw=2, alpha=0.5)

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
