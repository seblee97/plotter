import math
import os
from typing import List, Tuple, Union

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from plotter import constants


def get_figure_skeleton(
    height: Union[int, float],
    width: Union[int, float],
    num_columns: int,
    num_rows: int,
) -> Tuple:
    """Get overall figure structure and specfication.

    Args:
        height: height of each subfigure in plot.
        width: width of each subfigure in plot.
        num_columns: total number of columns in plot.
        num_rows: total number of rows in plot.

    Returns:
        fig: matplotlib figure object
        spec: matplotlib gridspec for fig organisation.
    """
    fig = plt.figure(
        constrained_layout=False, figsize=(num_columns * width, num_rows * height)
    )

    heights = [height for _ in range(num_rows)]
    widths = [width for _ in range(num_columns)]

    spec = gridspec.GridSpec(
        nrows=num_rows,
        ncols=num_columns,
        width_ratios=widths,
        height_ratios=heights,
    )

    return fig, spec


def smooth_data(data: List[float], window_width: int) -> List[float]:
    """Calculates moving average of list of values

    Args:
        data: raw, un-smoothed data.
        window_width: width over which to take moving averags

    Returns:
        smoothed_values: averaged data
    """

    def _smooth(single_dataset):
        cumulative_sum = np.cumsum(single_dataset, dtype=np.float32)
        cumulative_sum[window_width:] = (
            cumulative_sum[window_width:] - cumulative_sum[:-window_width]
        )
        # explicitly forcing data type to 32 bits avoids floating point errors
        # with constant dat.
        smoothed_values = np.array(
            cumulative_sum[window_width - 1 :] / window_width, dtype=np.float32
        )
        return smoothed_values

    if all(isinstance(d, list) for d in data):
        smoothed_data = []
        for dataset in data:
            smoothed_data.append(_smooth(dataset))
    elif all(
        (isinstance(d, float) or isinstance(d, int) or isinstance(d, np.int64))
        for d in data
    ):
        smoothed_data = _smooth(data)
    else:
        raise ValueError(
            "data is incorrect type, should be either list of lists or list."
        )

    return smoothed_data


def _get_cmap(colormap: str) -> Tuple:
    """Obtain matplotlig colormap.

    Either returns a continuous colormap or discrete cycle.

    Args:
        colormap: name of map.

    Returns:
        cmap: colormap obect or color cycle.
        cmap_type: continuous or discrete.
    """
    cmap = cm.get_cmap(colormap)

    if isinstance(cmap, mpl.colors.ListedColormap):
        cmap_type = constants.DISCRETE
        color_cycle = mpl.cycler(color=cmap.colors)
        mpl.rcParams["axes.prop_cycle"] = color_cycle
        return cmap, cmap_type
    else:
        cmap_type = constants.CONTINUOUS
        return cmap, cmap_type


def plot_multi_seed_run(
    fig,
    tag: str,
    cmap,
    cmap_type: str,
    relevant_experiments: List[str],
    experiment_folders: List[str],
    window_width: int,
    linewidth: int,
    legend=True,
):
    """Plots for multiple runs each with multiple seeds.

    Args:
        fig: matplotlib figure object.
        tag: name of tag in data to plot.
        cmap: colormap.
        cmap_type: colormap type (discrete or continuous).
        relevant_experiments: subset of experiments to include.
        experiment_folders: all experiment folders from which to plot data.
        window_width: smoothing parameter.
        linewidth: width of line on plot.
        legend: whether or not to inlcude legend.

    Returns:
        fig: figure with plots.
    """
    for exp_i, exp in enumerate(relevant_experiments):

        if cmap_type == constants.DISCRETE:
            color = cmap(exp_i / len(relevant_experiments))

        attribute_data = []
        seed_folders = [
            f
            for f in os.listdir(experiment_folders[exp])
            if os.path.isdir(os.path.join(experiment_folders[exp], f))
        ]

        for seed in seed_folders:
            df = pd.read_csv(
                os.path.join(experiment_folders[exp], seed, "data_logger.csv")
            )
            tag_data = df[tag].dropna()
            attribute_data.append(tag_data)

        if len(attribute_data):
            mean_attribute_data = np.mean(attribute_data, axis=0)
            std_attribute_data = np.std(attribute_data, axis=0)
            smooth_mean_data = smooth_data(
                mean_attribute_data, window_width=window_width
            )
            smooth_std_data = smooth_data(std_attribute_data, window_width=window_width)
            if len(smooth_mean_data):
                scaled_x = (len(df) / len(smooth_mean_data)) * np.arange(
                    len(smooth_mean_data)
                )
                kwargs = {"linewidth": linewidth, "label": exp}
                if cmap_type == constants.DISCRETE:
                    kwargs["color"] = color
                plt.plot(scaled_x, smooth_mean_data, **kwargs)
                kwargs = {"alpha": 0.3}
                if cmap_type == constants.DISCRETE:
                    kwargs["color"] = color
                plt.fill_between(
                    scaled_x,
                    smooth_mean_data - smooth_std_data,
                    smooth_mean_data + smooth_std_data,
                    **kwargs,
                )
    if legend:
        plt.legend(
            bbox_to_anchor=(
                1.01,
                1.0,
            ),
            loc="upper left",
            ncol=1,
            borderaxespad=0.0,
        )
    plt.xlabel(constants.TIME_UNIT)
    plt.ylabel(tag)

    # fig.tight_layout()
    return fig


def plot_multi_seed_multi_run(
    folder_path: str,
    exp_names: List[str],
    window_width: int,
    linewidth: int = 3,
    colormap: Union[str, None] = None,
) -> None:
    """Plot all data in separate figures for each tag.

    Expected structure of folder_path is:

    - folder_path
        |_ run_1
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ run_2
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ ...
        |_ ...
        |_ run_N
            |_ seed_0
            |_ seed_1
            |_ ...
            |_ seed_M

    with a file called data_logger.csv in each leaf folder.

    Args:
        folder_path: path to data folder.
        exp_names: list of experiment names within folder path.
        window_width: moving average smoothing parameter.
        linewidth: width of line on plot.
        colormap: name of colormap to use.
    """
    experiment_folders = {
        exp_name: os.path.join(folder_path, exp_name) for exp_name in exp_names
    }

    tag_set = {}

    # arbitrarily select one seed's dataframe for each run to find set of column names
    for exp, exp_path in experiment_folders.items():
        ex_seed = [
            f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))
        ][0]

        ex_df = pd.read_csv(os.path.join(exp_path, ex_seed, "data_logger.csv"))
        tag_subset = list(ex_df.columns)
        for tag in tag_subset:
            if tag not in tag_set:
                tag_set[tag] = []
            tag_set[tag].append(exp)

    cmap, cmap_type = _get_cmap(colormap)

    for tag, relevant_experiments in tag_set.items():
        if tag not in []:

            print(tag)

            fig = plot_multi_seed_run(
                fig=plt.figure(figsize=(constants.SUMMARY_FIGSIZE)),
                tag=tag,
                relevant_experiments=relevant_experiments,
                experiment_folders=experiment_folders,
                window_width=window_width,
                linewidth=linewidth,
                cmap=cmap,
                cmap_type=cmap_type,
            )

            os.makedirs(os.path.join(folder_path, "figures"), exist_ok=True)
            fig.savefig(
                os.path.join(
                    folder_path, "figures", f"{tag}_plot_multi_seed_multi_run.pdf"
                ),
                dpi=100,
            )
            plt.close()


def plot_all_multi_seed_multi_run(
    folder_path: str,
    exp_names: List[str],
    window_width: int,
    linewidth: int = 3,
    colormap: Union[str, None] = None,
):
    """Plot all data in one single figure (all exps, all repeats.)

    Expected structure of folder_path is:

    - folder_path
        |_ run_1
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ run_2
        |   |_ seed_0
        |   |_ seed_1
        |   |_ ...
        |   |_ seed_M
        |
        |_ ...
        |_ ...
        |_ run_N
            |_ seed_0
            |_ seed_1
            |_ ...
            |_ seed_M

    with a file called data_logger.csv in each leaf folder.

    Args:
        folder_path: path to data folder.
        exp_names: list of experiment names within folder path.
        window_width: moving average smoothing parameter.
        linewidth: width of line on plot.
        colormap: name of colormap to use.
    """
    experiment_folders = {
        exp_name: os.path.join(folder_path, exp_name) for exp_name in exp_names
    }

    tag_set = {}

    # arbitrarily select one seed's dataframe for each run to find set of column names
    for exp, exp_path in experiment_folders.items():
        ex_seed = [
            f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))
        ][0]

        ex_df = pd.read_csv(os.path.join(exp_path, ex_seed, "data_logger.csv"))
        tag_subset = list(ex_df.columns)
        for tag in tag_subset:
            if tag not in tag_set:
                tag_set[tag] = []
            tag_set[tag].append(exp)

    cmap, cmap_type = _get_cmap(colormap)

    num_graphs = len(tag_set)

    default_layout = (
        math.ceil(np.sqrt(num_graphs)),
        math.ceil(np.sqrt(num_graphs)),
    )
    graph_layout = constants.GRAPH_LAYOUTS.get(num_graphs, default_layout)

    num_rows = graph_layout[0]
    num_columns = graph_layout[1]

    fig, spec = get_figure_skeleton(
        height=4, width=5, num_columns=num_columns, num_rows=num_rows
    )

    tag_list = list(tag_set.keys())
    exp_list = list(tag_set.values())

    for row in range(num_rows):
        for col in range(num_columns):

            graph_index = (row) * num_columns + col

            if graph_index < num_graphs:

                print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))

                fig_sub = fig.add_subplot(spec[row, col])

                _ = plot_multi_seed_run(
                    fig=fig_sub,
                    tag=tag_list[graph_index],
                    relevant_experiments=exp_list[graph_index],
                    experiment_folders=experiment_folders,
                    window_width=window_width,
                    linewidth=linewidth,
                    cmap=cmap,
                    cmap_type=cmap_type,
                    legend=graph_index == num_graphs - 1,
                )

    save_path = os.path.join(folder_path, "all_plot.pdf")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
