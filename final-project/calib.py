import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


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
    """

    # Define sigmoid function to use
    def sigmoid(x):
        return np.exp(x)/(1+np.exp(x))

    # Begin creating the plot
    plt.figure(figsize=(10, 8))
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
             's-', color='#0173B2', label='Logistic regression')

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
