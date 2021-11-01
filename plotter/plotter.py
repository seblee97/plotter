import math
import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotter import constants, plot_functions


class Plotter:
    """Class for plotting scalar data."""

    def __init__(
        self, save_folder: str, logfile_path: str, smoothing: int, xlabel: str
    ):
        self._save_folder = save_folder
        self._logfile_path = logfile_path
        self._smoothing = smoothing
        self._xlabel = xlabel

        self._plot_tags: List[str]
        self._tag_grouping: List[Union[str, List[str]]]

        self._log_df: pd.DataFrame
        self._scaling: int

    def load_data(self) -> None:
        """Read in data logged to path."""
        self._log_df = pd.read_csv(self._logfile_path)
        self._plot_tags = list(self._log_df.columns)
        self._scaling = len(self._log_df)

    def add_tag_groups(self, tag_groups: List[Tuple[str, List[str]]]) -> None:
        """Method for assining tag groups.

        This is to allow for plotting subsets of tags in one plot,
        i.e. for more direct comparison.

        Args:
            tag_groups: list of tuples of tag groups.
            The first element of the tuple is the group name.
            The second element of the tuple is the list of tags under the group.
        """
        self._plot_tags.extend(tag_groups)

    def plot_learning_curves(self) -> None:
        """Main call function,
        calls smoothed and non-smoothed version.
        """
        # unsmoothed
        self._plot_learning_curves(smoothing=None)
        # smoothed
        self._plot_learning_curves(smoothing=self._smoothing)

    def _plot_learning_curves(self, smoothing: Union[None, int]) -> None:
        """Plot each tag data.

        Args:
            smoothing: moving average window.
        """
        num_graphs = len(self._plot_tags)

        default_layout = (
            math.ceil(np.sqrt(num_graphs)),
            math.ceil(np.sqrt(num_graphs)),
        )
        graph_layout = constants.GRAPH_LAYOUTS.get(num_graphs, default_layout)

        num_rows = graph_layout[0]
        num_columns = graph_layout[1]

        self.fig, self.spec = plot_functions.get_figure_skeleton(
            height=4, width=5, num_columns=num_columns, num_rows=num_rows
        )

        for row in range(num_rows):
            for col in range(num_columns):

                graph_index = (row) * num_columns + col

                if graph_index < num_graphs:

                    print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))
                    self._plot_scalar(
                        row=row,
                        col=col,
                        data_tag=self._plot_tags[graph_index],
                        smoothing=smoothing,
                    )

        if smoothing is not None:
            save_path = os.path.join(self._save_folder, constants.PLOT_PDF)
        else:
            save_path = os.path.join(self._save_folder, constants.RAW_PLOT_PDF)
        plt.tight_layout()
        self.fig.savefig(save_path, dpi=100)
        plt.close()

    def _plot_scalar(
        self,
        row: int,
        col: int,
        data_tag: Union[str, Tuple[str, List[str]]],
        smoothing: int,
    ):
        """Plot for specific tag.

        Args:
            row: subplot row id.
            col: subplot column id.
            data_tag: name of tag or tag group tuple.
            smoothing: moving average window.
        """
        fig_sub = self.fig.add_subplot(self.spec[row, col])

        # labelling
        fig_sub.set_xlabel(self._xlabel)

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which="major", linestyle="-", linewidth="0.5", color="red", alpha=0.2
        )
        fig_sub.grid(
            which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.4
        )

        # plot data
        if isinstance(data_tag, str) and data_tag in self._log_df.columns:
            sub_fig_tags = [data_tag]
            sub_fig_data = [self._log_df[data_tag].dropna()]
            fig_sub.set_ylabel(data_tag)
        elif isinstance(data_tag, tuple):
            group_label = data_tag[0]
            fig_sub.set_ylabel(group_label)
            sub_fig_tags = data_tag[1]
            sub_fig_data = [self._log_df[tag].dropna() for tag in sub_fig_tags]

        if smoothing is not None:
            smoothed_data = [
                plot_functions.smooth_data(
                    data=data.to_numpy(), window_width=min(len(data), smoothing)
                )
                for data in sub_fig_data
            ]
        else:
            smoothed_data = sub_fig_data

        x_data = [
            (self._scaling / len(data)) * np.arange(len(data)) for data in smoothed_data
        ]

        for x, y, label in zip(x_data, smoothed_data, sub_fig_tags):
            fig_sub.plot(x, y, label=label)

        if isinstance(data_tag, tuple):
            fig_sub.legend()
        row_index = row + 1
        title = f"{str(col + 1)}{chr(ord('`')+row_index)}"
        fig_sub.set_title(title)
