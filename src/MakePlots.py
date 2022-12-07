"""Make plots for interpreting results and presentation."""

import matplotlib.pyplto as plt
import numpy as np
import sys


def plot_loss(loss_path, ax, g_color="C0", c_color="C1",
              g_label="Generator Loss", c_label="Critic Loss"):
    """Plot the generator and critic loss."""
    # Load the loss functions
    loss_generator = np.loadtxt("{}/loss_generator.txt".format(loss_path))
    loss_critic = np.loadtxt("{}/loss_critic.txt".format(loss_path))

    # Plot the loss functions as a function of epoch
    ax[0].plot(loss_generator, color=g_color, label=g_label)
    ax[1].plot(loss_critic, color=c_color, label=c_label)


def create_loss_plot(loss_paths, output_path, labels=[]):
    """Initialize and save loss function plots."""
    # Initialize plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # If the loss_paths is a list then plot each one
    if isinstance(loss_paths, list):
        # Plot the loss functions for each path
        for i, loss_path in enumerate(loss_paths):
            # Check if labels were given
            if labels:
                plot_loss(loss_path, ax, g_color="C{}".format(2*i),
                          c_color="C{}".format(2*i+1), g_label=labels[i][0],
                          c_label=labels[i][1])

            else:
                plot_loss(loss_path, ax, g_color="C{}".format(2*i),
                          c_color="C{}".format(2*i+1))
    # If not a list of loss_paths then only plot one set
    else:
        plot_loss(loss_path, ax)

    ax[0].set_title("Generator Loss")
    ax[0].set_ylabel("Epoch")
    ax[1].set_title("Critic Loss")
    ax[1].set_ylabel("Epoch")

    if labels:
        ax[0].legend()
        ax[1].legend()

    plt.savefig(output_path)


path_root = ("/home/mj3chapm/projects/rrg-wperciva/woodfiaj/"
             "PHYS-449-midterm-project/results")

learning_rates = ["1e-5", "5e-5", "1e-4", "5e-4", "1e-3"]
for i in range(len(learning_rates)):
    loss_paths = []
    labels = []
    for j in range(len(learning_rates)):
        loss_paths.append("{}/learning_rate_tests/C{}-"
                          "G{}".format(path_root, learning_rates[i],
                                       learning_rates[j]))
        labels.append("C{}-G{}".format(learning_rates[i], learning_rates[j]))
    create_loss_plot(loss_paths,
                     "{}/learning_rate_tests/C{}_variable-G_loss_plot"
                     ".pdf".format(path_root, learning_rates[i]),
                     labels=labels)

    loss_paths = []
    labels = []
    for j in range(len(learning_rates)):
        loss_paths.append("{}/learning_rate_tests/C{}-"
                          "G{}".format(path_root, learning_rates[j],
                                       learning_rates[i]))
        labels.append("C{}-G{}".format(learning_rates[j], learning_rates[i]))
    create_loss_plot(loss_paths,
                     "{}/learning_rate_tests/variable-C_G{}_loss_plot.pdf",
                     labels=labels)
    create_loss_plot(loss_paths,
                     "{}/learning_rate_tests/variable-C_G{}_loss_plot"
                     ".pdf".format(path_root, learning_rates[i]),
                     labels=labels)

    print("Finished plots for learning rate = {}".format(learning_rates[i]))

create_loss_plot("{}/longEpochRunTest".format(path_root),
                 "{}/longEpochRunTest/loss_plot.pdf".format(path_root))
print("Finished plot for longEpochRunTest")
create_loss_plot("{}/longEpochRun2".format(path_root),
                 "{}/longEpochRun2/loss_plot.pdf".format(path_root))
print("Finished plot for longEpochRun2")
create_loss_plot("{}/longEpochRun".format(path_root),
                 "{}/longEpochRun/loss_plot.pdf".format(path_root))
print("Finished plot for longEpochRun")
