import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
    """

    # Define sigmoid function to convert logits to probabilities
    def sigmoid(x):
        return np.exp(x)/(1+np.exp(x))

    # Keeping track of ROC and AOC values, storing e.g colours
    num_of_samples = logits['train'].shape(1)
    color_dict = {'train': '#0173B2', 'test': '#D55E00'}
    aoc_dict = {'train': [], 'test': []}

    # Figure plotting
    plt.figure(figsize=(10, 10))

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
                plt.plot(fpr, tpr, color=color_dict[key], lw=2, alpha=0.3)

        # Plot posterior mean for train/test, give AOC with +\- std
        fpr, tpr, _ = roc_curve(
            y_true=gt[key],
            y_score=sigmoid(np.mean(logits[key], axis=1))
        )

        label = r'{} AOC: {} $\pm$ {}'.format(
            key, np.mean(aoc_dict[key]), np.std(aoc_dict[key])
        )
        plt.plot(fpr, tpr, color=color_dict[key], lw=3, label=label)

    # Add axis labels, plot/save the figure
    plt.plot([0, 1], [0, 1], color='#949494', lw=3, linestyle='--')
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for training and test sets')
    plt.legend(loc="lower right")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    if display:
        plt.show()
