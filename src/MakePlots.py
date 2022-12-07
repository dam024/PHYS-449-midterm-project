"""Make plots for interpreting results and presentation."""

import matplotlib.pyplto as plt
import numpy as np
import sys


loss_path = sys.argv[1]
loss_generator = np.loadtxt("{}/loss_generator.txt".format(loss_path))
loss_critic = np.loadtxt("{}/loss_critic.txt".format(loss_path))

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(loss_generator, color="C0")
ax[0].set_title("Generator Loss")
ax[0].set_ylabel("Epoch")

ax[1].plot(loss_critic, color="C1")
ax[1].set_title("Critic Loss")
ax[1].set_ylabel("Epoch")

plt.savefig("{}/loss_plot.pdf".format(loss_path))
