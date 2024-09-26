import os

import matplotlib.pyplot as plt
import numpy as np


class RocPlotter:
    def __init__(self, save_file, nb_columns=3) -> None:
        self.filename = save_file
        self.nb_lines = 2
        self.nb_columns = nb_columns
        self.fig, self.axs = plt.subplots(
            nrows=self.nb_lines,
            ncols=self.nb_columns,
            sharex=False,
            sharey=False,
            figsize=(15, 15),
        )
        self.nb_plotted = 0
        self.next_y = 0
        self.next_iteration = 0
        self.log_next = False

    def _update_coordinates(self):
        self.next_y += 1
        if self.next_y >= self.nb_columns:
            self.log_next = False

    def set_next_plot(self, iteration: int, log_next: bool):
        self.next_iteration = iteration
        self.log_next = log_next

    def plot_roc(self, fpr, tpr, thresholds, roc_auc, losses_train, losses_test):
        if not self.log_next or self.next_y >= self.nb_columns:
            return  # We do not plot anything

        axs_roc: plt.axes.Axes = self.axs[0, self.next_y]

        axs_roc.plot(fpr, tpr, "b")

        # current_axs.legend(loc="lower right")
        axs_roc.plot([0, 1], [0, 1], "r--")
        axs_roc.set_xlim([0, 1])
        axs_roc.set_ylim([0, 1])

        axs_roc.set_ylabel("True Positive Rate")
        axs_roc.set_xlabel("False Positive Rate")
        axs_roc.set_title(f"{self.next_iteration} - AUC = {roc_auc:.2f}")

        # Plot the loss histograms
        # Define number of bins and bin sizes
        bins = 30

        data_min = min(losses_train.min(), losses_test.min())
        data_max = max(losses_train.max(), losses_test.max())

        # Create common bin edges using the combined range
        bin_edges = np.linspace(data_min, data_max, bins + 1)

        bin_width = bin_edges[1] - bin_edges[0]

        # Compute histograms using numpy
        hist_train, _ = np.histogram(
            losses_train,
            bins=bin_edges,
            density=False,
        )
        hist_test, _ = np.histogram(
            losses_test,
            bins=bin_edges,
            density=False,
        )

        # Normalize histograms (manual normalization)
        hist_train_normalized = hist_train / hist_train.sum()
        hist_test_normalized = hist_test / hist_test.sum()

        # Get the axes
        axs_losses_train: plt.axes.Axes = self.axs[1, self.next_y]
        axs_losses_train.set_title(f"{self.next_iteration}")

        axs_losses_train.tick_params(axis="y", labelcolor="b")
        axs_losses_train.set_ylabel("Train set", color="b")
        axs_losses_train.set_xlabel("Loss")

        split_axis = False
        if split_axis:
            axs_losses_test = (
                axs_losses_train.twinx()
            )  # instantiate a second Axes that shares the same x-axis
            axs_losses_test.tick_params(axis="y", labelcolor="r")
            axs_losses_test.set_ylabel("Test set", color="r")
        else:
            axs_losses_test = axs_losses_train
            axs_losses_test.set_ylabel("Density")

        # Plot train histogram
        axs_losses_train.bar(
            bin_edges[:-1],
            hist_train_normalized,
            width=bin_width,
            alpha=0.5,
            label="Train set",
            color="b",
            zorder=2,
        )

        # Plot second histogram
        axs_losses_test.bar(
            bin_edges[:-1],
            hist_test_normalized,
            width=bin_width,
            alpha=0.5,
            label="Test set",
            color="r",
            zorder=1,
        )

        if not split_axis:
            axs_losses_train.legend()

        self._update_coordinates()
        self.fig.tight_layout()
        self.fig.suptitle("Iterations ROC and losses distributions")
        self.fig.savefig(self.filename)
        # if self.next_x >= self.nb_lines:
        #     print(f"Saved ROC curves at {self.filename}")
        #     plt.close(self.fig)
