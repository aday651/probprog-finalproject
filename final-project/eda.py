import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bitcoin import BitcoinOTC


# Converting torch tensors to numpy arrays...
def plot_summary_stat(fname=None, display=False):
    r""" Plots summary statistics of the Bitcoin OTC dataset.

    Args:
        fname (str, optional) : Path/filename to save the plot. If fname = None
            the plot is not displayed.
        display (bool, optional) : Determines whether the plot is displayed.
    """
    data = BitcoinOTC()
    out_degree = data.out_degree.numpy()
    in_degree = data.in_degree.numpy()
    out_weight_avg = data.out_weight_avg.numpy()
    out_weight_std = data.out_weight_std.numpy()
    in_weight_avg = data.in_weight_avg.numpy()
    in_weight_std = data.in_weight_std.numpy()
    rate_time_out_std = data.rate_time_out_std.numpy()
    rate_time_in_std = data.rate_time_in_std.numpy()
    gt = data.gt.numpy()

    # Define colour/labels, axis limits to use for the plots
    colors = ['#949494', '#0173B2', '#D55E00']
    labels = ['All users', 'Known trustworthy users',
              'Known un-trustworthy users']
    title_label = ['Distribution of number of ratings given',
                   'Distribution of number of ratings received',
                   'Distribution of average rating given',
                   'Distribution of standard deviation of ratings given',
                   'Distribution of average rating received',
                   'Distribution of standard deviation of ratings received',
                   'Difference in number of ratings given and received',
                   'Difference in average rating given and received',
                   'Std. of times of giving ratings',
                   'Std. of times of recieving ratings']

    xlim_list = [[0, 50], [0, 50],
                 [0, 1], [0, np.sqrt(0.25)],
                 [0, 1], [0, np.sqrt(0.25)],
                 [-50, 50], [-1, 1],
                 [0, 250], [0, 250]]

    # Define data, relevant masks
    stats = [out_degree, in_degree,
             out_weight_avg, out_weight_std,
             in_weight_avg, in_weight_std,
             out_degree - in_degree, out_weight_avg - in_weight_avg,
             rate_time_out_std, rate_time_in_std]
    masks = [range(data.num_nodes), (gt == 1), (gt == 0)]

    # Begin plotting
    f = plt.figure(figsize=(12, 30))

    for i, stat in enumerate(stats):
        ax = f.add_subplot(5, 2, i+1)

        for j, mask in enumerate(masks):
            temp_data = stat[mask]
            temp_data = temp_data[~np.isnan(temp_data)]

            sns.distplot(temp_data,
                         hist=True,
                         kde=True,
                         bins=20,
                         kde_kws={'linewidth': 3,
                                  'clip': xlim_list[i]},
                         label=labels[j],
                         color=colors[j],
                         ax=ax)

            ax.set_title(title_label[i])
            ax.set_ylabel('Density')
            ax.set_xlim(xlim_list[i])
            ax.legend(loc='upper right')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    if display:
        plt.show()
